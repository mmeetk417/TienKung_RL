"""Terminal, JSON, CSV, and run-directory reporting for telemetry."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType

from tienkung_rl.config.load import write_config_snapshot
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json, sha256_fingerprint

from .metrics import METRIC_SCHEMA, TelemetryDiagnostic, TelemetrySnapshot

REPORT_SCHEMA_VERSION = "tienkung_telemetry_report_v1"
RUN_MANIFEST_VERSION = "tienkung_run_manifest_v1"


@dataclass(frozen=True)
class ReportMetadata:
    seed: int
    command: str
    config_fingerprint: str
    asset_fingerprints: Mapping[str, str]
    config_version: str
    randomization_version: str
    fixed_torsional_friction: float
    fixed_rolling_friction: float
    training_seeds: tuple[int, ...] = ()
    evaluation_seeds: tuple[int, ...] = ()

    @classmethod
    def from_config(
        cls,
        config: ResolvedRunConfig,
    ) -> ReportMetadata:
        config.validate()
        asset_fingerprints = {
            item.name: getattr(config.asset, item.name)
            for item in fields(config.asset)
            if item.name.endswith("_sha256")
        }
        return cls(
            seed=config.run.seed,
            command=config.run.command,
            config_fingerprint=sha256_fingerprint(config),
            asset_fingerprints=MappingProxyType(asset_fingerprints),
            config_version=config.run.code_version,
            randomization_version=config.randomization.version,
            fixed_torsional_friction=config.randomization.fixed_torsional_friction,
            fixed_rolling_friction=config.randomization.fixed_rolling_friction,
            training_seeds=config.evaluation.training_seeds,
            evaluation_seeds=config.evaluation.evaluation_seeds,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "seed": self.seed,
            "command": self.command,
            "config_fingerprint": self.config_fingerprint,
            "asset_fingerprints": dict(self.asset_fingerprints),
            "config_version": self.config_version,
            "randomization_version": self.randomization_version,
            "fixed_torsional_friction": self.fixed_torsional_friction,
            "fixed_rolling_friction": self.fixed_rolling_friction,
            "training_seeds": list(self.training_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
        }


def _diagnostic_dict(item: TelemetryDiagnostic) -> dict[str, object]:
    return {
        "kind": item.kind.value,
        "message": item.message,
        "count": item.count,
        "env_id": item.env_id,
        "policy_step": item.policy_step,
        "config_fingerprint": item.config_fingerprint,
        "context": dict(item.context),
    }


def report_payload(snapshot: TelemetrySnapshot, metadata: ReportMetadata) -> dict[str, object]:
    metrics = []
    for name, spec in METRIC_SCHEMA.items():
        aggregate = snapshot.metrics[name]
        metrics.append(
            {
                "name": name,
                "level": spec.level.value,
                "unit": spec.unit,
                "reduction": spec.reduction.value,
                "conditional": spec.conditional,
                "value": aggregate.value,
                "count": aggregate.count,
                "numerator": aggregate.numerator,
                "denominator": aggregate.denominator,
            }
        )
    return {
        "metadata": metadata.to_dict(),
        "iteration": snapshot.iteration,
        "randomization_version": snapshot.randomization_version,
        "bottleneck": snapshot.bottleneck,
        "metrics": metrics,
        "diagnostics": [_diagnostic_dict(item) for item in snapshot.diagnostics],
    }


def write_json_report(
    path: str | Path,
    snapshot: TelemetrySnapshot,
    metadata: ReportMetadata,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(report_payload(snapshot, metadata)) + "\n", encoding="utf-8", newline="\n")
    return target


CSV_FIELDS = (
    "schema_version",
    "iteration",
    "seed",
    "command",
    "config_fingerprint",
    "asset_fingerprints",
    "name",
    "level",
    "unit",
    "reduction",
    "conditional",
    "value",
    "count",
    "numerator",
    "denominator",
    "diagnostics",
)


def write_csv_report(
    path: str | Path,
    snapshot: TelemetrySnapshot,
    metadata: ReportMetadata,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_json = canonical_json([_diagnostic_dict(item) for item in snapshot.diagnostics])
    asset_json = canonical_json(dict(metadata.asset_fingerprints))
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for name, spec in METRIC_SCHEMA.items():
            aggregate = snapshot.metrics[name]
            writer.writerow(
                {
                    "schema_version": REPORT_SCHEMA_VERSION,
                    "iteration": snapshot.iteration,
                    "seed": metadata.seed,
                    "command": metadata.command,
                    "config_fingerprint": metadata.config_fingerprint,
                    "asset_fingerprints": asset_json,
                    "name": name,
                    "level": spec.level.value,
                    "unit": spec.unit,
                    "reduction": spec.reduction.value,
                    "conditional": str(spec.conditional).lower(),
                    "value": format(aggregate.value, ".17g"),
                    "count": aggregate.count,
                    "numerator": format(aggregate.numerator, ".17g"),
                    "denominator": aggregate.denominator,
                    "diagnostics": diagnostic_json,
                }
            )
    return target


@dataclass(frozen=True)
class TerminalSummary:
    text: str
    values: Mapping[str, float]
    counts: Mapping[str, int]
    alerts: tuple[str, ...]


def render_terminal_summary(
    snapshot: TelemetrySnapshot,
    *,
    dangerous_rate_limit: float = 0.10,
) -> TerminalSummary:
    if not 0.0 <= dangerous_rate_limit <= 1.0:
        raise ValueError("dangerous_rate_limit must lie in [0, 1]")
    values = {name: snapshot.metrics[name].value for name in METRIC_SCHEMA}
    counts = {name: snapshot.metrics[name].count for name in METRIC_SCHEMA}
    alerts: list[str] = []
    alert_metrics = (
        ("warning.nonfinite.count", "NONFINITE"),
        ("warning.worker.count", "WORKER_EXCEPTION"),
        ("warning.mujoco.count", "MUJOCO_WARNING"),
        ("warning.solver.count", "SOLVER_ANOMALY"),
        ("warning.capacity.count", "CAPACITY_ANOMALY"),
    )
    for name, label in alert_metrics:
        if values[name] > 0:
            alerts.append(f"{label}: {values[name]:.17g}")
    dangerous = snapshot.metrics["outcome.dangerous_rate"]
    if dangerous.count and dangerous.value > dangerous_rate_limit:
        alerts.append(f"DANGEROUS_RATE_EXCEEDED: {dangerous.value:.17g} > {dangerous_rate_limit:.17g}")
    stalled = snapshot.metrics["curriculum.stalled_rate"]
    if stalled.count and stalled.value > 0:
        alerts.append(f"CURRICULUM_STALLED: {stalled.value:.17g}")
    if snapshot.bottleneck:
        alerts.append(f"BOTTLENECK: {snapshot.bottleneck}")
    for diagnostic in snapshot.diagnostics:
        context = canonical_json(dict(diagnostic.context))
        alerts.append(
            f"DIAGNOSTIC[{diagnostic.kind.value}] x{diagnostic.count}: {diagnostic.message} context={context}"
        )

    lines = [f"iteration={snapshot.iteration}"]
    lines.extend(f"metric {name} value={values[name]:.17g} count={counts[name]}" for name in METRIC_SCHEMA)
    lines.extend(f"ALERT {item}" for item in alerts)
    return TerminalSummary(
        text="\n".join(lines),
        values=MappingProxyType(values),
        counts=MappingProxyType(counts),
        alerts=tuple(alerts),
    )


RUN_REQUIRED_FILES = (
    "config/resolved.yaml",
    "config/canonical.json",
    "config/dependencies.txt",
    "assets/manifest.json",
    "evaluation/final.json",
    "evaluation/final.csv",
    "run_manifest.json",
)
RUN_REQUIRED_PATTERNS = (
    "checkpoints/model_<iteration>.pt",
    "checkpoints/model_<iteration>.json",
    "tensorboard/events.out.tfevents.*",
)


def _configured_asset_manifest(config: ResolvedRunConfig) -> dict[str, object]:
    return {
        "asset_version": config.asset.asset_version,
        "calibration_status": config.asset.calibration_status,
        "paths": {
            item.name: getattr(config.asset, item.name)
            for item in fields(config.asset)
            if item.name.endswith("_path") or item.name.endswith("_directory")
        },
        "fingerprints": {
            item.name: getattr(config.asset, item.name)
            for item in fields(config.asset)
            if item.name.endswith("_sha256")
        },
    }


def prepare_run_directory(
    config: ResolvedRunConfig,
    run_dir: str | Path | None = None,
    *,
    dependencies: Sequence[str] | None = None,
    asset_manifest: Mapping[str, object] | None = None,
    exist_ok: bool = False,
) -> Path:
    """Create the run skeleton and manifest without fabricating runtime output."""

    config.validate()
    target = Path(config.run.output_dir if run_dir is None else run_dir)
    manifest_path = target / "run_manifest.json"
    if target.is_dir() and any(target.iterdir()) and not exist_ok:
        raise FileExistsError(f"run directory is already initialized: {target}")
    for directory in ("config", "assets", "checkpoints", "tensorboard", "evaluation"):
        (target / directory).mkdir(parents=True, exist_ok=True)
    config_fingerprint = write_config_snapshot(config, target / "config", dependencies=dependencies)
    assets = dict(asset_manifest) if asset_manifest is not None else _configured_asset_manifest(config)
    (target / "assets" / "manifest.json").write_text(
        canonical_json(assets) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    run_manifest = {
        "schema_version": RUN_MANIFEST_VERSION,
        "seed": config.run.seed,
        "command": config.run.command,
        "config_fingerprint": config_fingerprint,
        "asset_fingerprint": config.asset.runtime_mjcf_sha256,
        "curriculum_state": {
            "stage": config.curriculum.initial_stage,
            "completed_episodes_in_window": 0,
            "consecutive_passing_windows": 0,
        },
        "required_files": list(RUN_REQUIRED_FILES),
        "required_patterns": list(RUN_REQUIRED_PATTERNS),
        "complete": False,
        "missing": [
            "checkpoints/model_<iteration>.pt",
            "checkpoints/model_<iteration>.json",
            "tensorboard/events.out.tfevents.*",
            "evaluation/final.json",
            "evaluation/final.csv",
        ],
    }
    manifest_path.write_text(canonical_json(run_manifest) + "\n", encoding="utf-8", newline="\n")
    return target


@dataclass(frozen=True)
class RunDirectoryStatus:
    complete: bool
    present: tuple[str, ...]
    missing: tuple[str, ...]
    orphan_checkpoints: tuple[str, ...]

    def require_complete(self) -> None:
        if not self.complete:
            detail = ", ".join((*self.missing, *self.orphan_checkpoints))
            raise FileNotFoundError(f"run directory is incomplete: {detail}")


def validate_run_directory(run_dir: str | Path) -> RunDirectoryStatus:
    target = Path(run_dir)
    present: list[str] = []
    missing: list[str] = []
    for relative in RUN_REQUIRED_FILES:
        path = target / relative
        (present if path.is_file() and path.stat().st_size > 0 else missing).append(relative)

    checkpoint_dir = target / "checkpoints"
    pt_stems = {path.stem for path in checkpoint_dir.glob("model_*.pt") if path.is_file() and path.stat().st_size > 0}
    json_stems = {
        path.stem for path in checkpoint_dir.glob("model_*.json") if path.is_file() and path.stat().st_size > 0
    }
    paired = pt_stems & json_stems
    if paired:
        present.extend(f"checkpoints/{stem}.pt" for stem in sorted(paired))
        present.extend(f"checkpoints/{stem}.json" for stem in sorted(paired))
    else:
        missing.extend(RUN_REQUIRED_PATTERNS[:2])
    orphan_checkpoints = tuple(
        sorted(
            [f"orphan checkpoint binary: checkpoints/{stem}.pt" for stem in pt_stems - json_stems]
            + [f"orphan checkpoint sidecar: checkpoints/{stem}.json" for stem in json_stems - pt_stems]
        )
    )

    tensorboard_files = sorted(
        path
        for path in (target / "tensorboard").glob("events.out.tfevents.*")
        if path.is_file() and path.stat().st_size > 0
    )
    if tensorboard_files:
        present.extend(path.relative_to(target).as_posix() for path in tensorboard_files)
    else:
        missing.append(RUN_REQUIRED_PATTERNS[2])

    status = RunDirectoryStatus(
        complete=not missing and not orphan_checkpoints,
        present=tuple(sorted(set(present))),
        missing=tuple(sorted(set(missing))),
        orphan_checkpoints=orphan_checkpoints,
    )
    return status


def finalize_run_manifest(run_dir: str | Path) -> RunDirectoryStatus:
    """Recompute manifest completeness after real producers write their files."""

    target = Path(run_dir)
    manifest_path = target / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing run manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid run manifest JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != RUN_MANIFEST_VERSION:
        raise ValueError("invalid run manifest schema")
    status = validate_run_directory(target)
    payload["complete"] = status.complete
    payload["missing"] = list(status.missing)
    payload["orphan_checkpoints"] = list(status.orphan_checkpoints)
    payload["present"] = list(status.present)
    manifest_path.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
    return status


def write_final_reports(
    run_dir: str | Path,
    snapshot: TelemetrySnapshot,
    metadata: ReportMetadata,
) -> tuple[Path, Path]:
    evaluation = Path(run_dir) / "evaluation"
    return (
        write_json_report(evaluation / "final.json", snapshot, metadata),
        write_csv_report(evaluation / "final.csv", snapshot, metadata),
    )


__all__ = [
    "CSV_FIELDS",
    "REPORT_SCHEMA_VERSION",
    "RUN_MANIFEST_VERSION",
    "RUN_REQUIRED_FILES",
    "RUN_REQUIRED_PATTERNS",
    "ReportMetadata",
    "RunDirectoryStatus",
    "TerminalSummary",
    "finalize_run_manifest",
    "prepare_run_directory",
    "render_terminal_summary",
    "report_payload",
    "validate_run_directory",
    "write_csv_report",
    "write_final_reports",
    "write_json_report",
]
