"""Deterministic frozen-manifest evaluation and three-seed quality gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, TextIO, cast

import torch
from rsl_rl.modules import ActorCritic

from tienkung_rl.config.models import EvaluationThresholdConfig, ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json
from tienkung_rl.task.curriculum import CurriculumStage, ForcedTargetAuthority
from tienkung_rl.task.environment import ReachThenCloseTask, TaskResetOverrides, TaskStepTransition
from tienkung_rl.task.randomization import ForcedCylinderParameters, ForcedParameterAuthority

from .checkpoint import ValidatedCheckpoint, load_checkpoint_preflight
from .manifests import EvaluationCase, EvaluationManifest, ManifestKind, file_sha256, load_manifest

EVALUATION_REPORT_SCHEMA_VERSION: Final = "tienkung_deterministic_evaluation_v2"
EVALUATION_CSV_SCHEMA_VERSION: Final = "tienkung_deterministic_evaluation_csv_v1"
GATE_SCHEMA_VERSION: Final = "tienkung_three_seed_gate_v1"
FINAL_USAGE_SCHEMA_VERSION: Final = "tienkung_final_manifest_usage_v1"


class EvaluationError(RuntimeError):
    """Evaluation inputs, execution, reports, or formal gates are invalid."""


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    training_seed: int
    evaluation_seed: int
    case_id: str
    target_x_m: float
    target_y_m: float
    radius_m: float
    height_m: float
    mass_kg: float
    hand_sliding_friction: float
    table_sliding_friction: float
    trigger_requested: bool
    option_accepted: bool
    lift_5cm: bool
    success: bool
    unsafe: bool
    primary_reason: str | None
    auxiliary_reasons: tuple[str, ...]
    final_position_error_m: float
    final_orientation_error_rad: float
    episode_steps: int

    def __post_init__(self) -> None:
        if type(self.training_seed) is not int or type(self.evaluation_seed) is not int:
            raise EvaluationError("episode seeds must be integers")
        if not self.case_id or self.episode_steps < 1:
            raise EvaluationError("episode case_id and positive episode length are required")
        floats = (
            self.target_x_m,
            self.target_y_m,
            self.radius_m,
            self.height_m,
            self.mass_kg,
            self.hand_sliding_friction,
            self.table_sliding_friction,
            self.final_position_error_m,
            self.final_orientation_error_rad,
        )
        if not all(math.isfinite(value) for value in floats):
            raise EvaluationError(f"episode {self.case_id} contains non-finite metrics")
        if self.success and self.unsafe:
            raise EvaluationError("evaluation success and unsafe must be mutually exclusive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricRatio:
    value: float
    numerator: int | float
    denominator: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not math.isfinite(float(self.numerator)):
            raise EvaluationError("metric ratios must be finite")
        if self.denominator < 0:
            raise EvaluationError("metric denominator must be non-negative")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    training_seed: int
    manifest_kind: ManifestKind
    manifest_hash: str
    completed_episodes: int
    expected_episodes: int
    option_accepted_rate: MetricRatio
    lift_5cm_rate: MetricRatio
    complete_success_rate: MetricRatio
    dangerous_rate: MetricRatio
    success_given_accepted: MetricRatio
    unsafe_given_accepted: MetricRatio

    @property
    def complete(self) -> bool:
        return self.completed_episodes == self.expected_episodes

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["manifest_kind"] = self.manifest_kind.value
        result["complete"] = self.complete
        return result


@dataclass(frozen=True, slots=True)
class DatasetGate:
    manifest_kind: ManifestKind
    required: bool
    complete: bool
    passed: bool
    training_seeds: tuple[int, ...]
    mean_option_accepted_rate: float
    mean_lift_5cm_rate: float
    mean_complete_success_rate: float
    mean_dangerous_rate: float
    per_seed_dangerous_rates: tuple[float, ...]
    failures: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["manifest_kind"] = self.manifest_kind.value
        return result


@dataclass(frozen=True, slots=True)
class GateDecision:
    gate: str
    passed: bool
    datasets: tuple[DatasetGate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate": self.gate,
            "passed": self.passed,
            "datasets": [item.to_dict() for item in self.datasets],
        }


def _ratio(numerator: int | float, denominator: int) -> MetricRatio:
    return MetricRatio(0.0 if denominator == 0 else float(numerator) / denominator, numerator, denominator)


def aggregate_episodes(
    episodes: Sequence[EpisodeEvaluation],
    *,
    manifest: EvaluationManifest,
    training_seed: int,
) -> EvaluationSummary:
    """Aggregate complete episodes without substituting unconditional means."""

    if type(training_seed) is not int:
        raise EvaluationError("training_seed must be an integer")
    if any(item.training_seed != training_seed for item in episodes):
        raise EvaluationError("episode training seeds differ from the requested aggregate")
    case_ids = tuple(item.case_id for item in episodes)
    if len(set(case_ids)) != len(case_ids):
        raise EvaluationError("evaluation episodes contain duplicate case IDs")
    expected_ids = {item.case_id for item in manifest.cases}
    if not set(case_ids).issubset(expected_ids):
        raise EvaluationError("evaluation episodes contain a case outside the manifest")
    count = len(episodes)
    accepted = sum(item.option_accepted for item in episodes)
    return EvaluationSummary(
        training_seed=training_seed,
        manifest_kind=manifest.kind,
        manifest_hash=manifest.manifest_hash,
        completed_episodes=count,
        expected_episodes=len(manifest.cases),
        option_accepted_rate=_ratio(accepted, count),
        lift_5cm_rate=_ratio(sum(item.lift_5cm for item in episodes), count),
        complete_success_rate=_ratio(sum(item.success for item in episodes), count),
        dangerous_rate=_ratio(sum(item.unsafe for item in episodes), count),
        success_given_accepted=_ratio(sum(item.success and item.option_accepted for item in episodes), accepted),
        unsafe_given_accepted=_ratio(sum(item.unsafe and item.option_accepted for item in episodes), accepted),
    )


def _dataset_gate(
    summaries: Sequence[EvaluationSummary],
    *,
    kind: ManifestKind,
    thresholds: EvaluationThresholdConfig | None,
    required: bool,
    expected_training_seeds: tuple[int, ...],
) -> DatasetGate:
    selected = tuple(item for item in summaries if item.manifest_kind is kind)
    failures: list[str] = []
    seeds = tuple(item.training_seed for item in selected)
    complete = (
        len(selected) == 3
        and len(set(seeds)) == 3
        and set(seeds) == set(expected_training_seeds)
        and all(item.complete for item in selected)
    )
    if not complete:
        failures.append(
            f"{kind.value}: requires complete reports for configured training seeds {expected_training_seeds}"
        )
    hashes = {item.manifest_hash for item in selected}
    if len(hashes) > 1:
        failures.append(f"{kind.value}: training seeds did not consume the same frozen manifest")

    def mean(attribute: str) -> float:
        if not selected:
            return 0.0
        return math.fsum(getattr(item, attribute).value for item in selected) / len(selected)

    accepted = mean("option_accepted_rate")
    lifted = mean("lift_5cm_rate")
    success = mean("complete_success_rate")
    dangerous = mean("dangerous_rate")
    per_seed_dangerous = tuple(item.dangerous_rate.value for item in selected)
    if thresholds is not None and complete:
        comparisons = (
            (accepted, thresholds.option_accepted_rate_min, ">=", "option accepted"),
            (lifted, thresholds.lift_5cm_rate_min, ">=", "Lift5cm"),
            (success, thresholds.complete_success_rate_min, ">=", "complete success"),
            (dangerous, thresholds.dangerous_rate_max, "<=", "dangerous"),
        )
        for value, threshold, relation, label in comparisons:
            tolerance = 1.0e-12
            if (relation == ">=" and value < threshold - tolerance) or (
                relation == "<=" and value > threshold + tolerance
            ):
                failures.append(f"{kind.value}: mean {label} {value:.6f} violates {relation} {threshold:.6f}")
        for seed, value in zip(seeds, per_seed_dangerous, strict=True):
            if value > thresholds.dangerous_rate_max + 1.0e-12:
                failures.append(
                    f"{kind.value}: seed {seed} dangerous rate {value:.6f} exceeds {thresholds.dangerous_rate_max:.6f}"
                )
    # ``required`` describes whether metric thresholds are quality gates.  It
    # never relaxes completeness or the same-frozen-manifest integrity rule.
    passed = complete and not failures
    return DatasetGate(
        manifest_kind=kind,
        required=required,
        complete=complete,
        passed=passed,
        training_seeds=seeds,
        mean_option_accepted_rate=accepted,
        mean_lift_5cm_rate=lifted,
        mean_complete_success_rate=success,
        mean_dangerous_rate=dangerous,
        per_seed_dangerous_rates=per_seed_dangerous,
        failures=tuple(failures),
    )


def preliminary_gate(
    standard: Sequence[EvaluationSummary],
    randomized: Sequence[EvaluationSummary],
    config: ResolvedRunConfig,
) -> GateDecision:
    """Apply stage-F thresholds; randomized final must be complete and public."""

    standard_gate = _dataset_gate(
        standard,
        kind=ManifestKind.FINAL_STANDARD,
        thresholds=config.evaluation.preliminary_thresholds,
        required=True,
        expected_training_seeds=config.evaluation.training_seeds,
    )
    random_gate = _dataset_gate(
        randomized,
        kind=ManifestKind.FINAL_RANDOMIZED,
        thresholds=None,
        required=False,
        expected_training_seeds=config.evaluation.training_seeds,
    )
    return GateDecision("preliminary", standard_gate.passed and random_gate.passed, (standard_gate, random_gate))


def final_gate(
    standard: Sequence[EvaluationSummary],
    randomized: Sequence[EvaluationSummary],
    config: ResolvedRunConfig,
) -> GateDecision:
    """Apply stage-G thresholds independently to both frozen final sets."""

    gates = tuple(
        _dataset_gate(
            summaries,
            kind=kind,
            thresholds=config.evaluation.final_thresholds,
            required=True,
            expected_training_seeds=config.evaluation.training_seeds,
        )
        for summaries, kind in (
            (standard, ManifestKind.FINAL_STANDARD),
            (randomized, ManifestKind.FINAL_RANDOMIZED),
        )
    )
    return GateDecision("final", all(item.passed for item in gates), gates)


def _policy(config: ResolvedRunConfig) -> ActorCritic:
    network = config.ppo.network
    return ActorCritic(
        num_actor_obs=106,
        num_critic_obs=106,
        num_actions=7,
        actor_hidden_dims=network.actor_hidden_dimensions,
        critic_hidden_dims=network.critic_hidden_dimensions,
        activation=network.activation,
        init_noise_std=config.ppo.initial_continuous_std,
        min_std=config.ppo.minimum_continuous_std,
        max_std=config.ppo.maximum_continuous_std,
    )


def preflight_evaluation_inputs(
    config: ResolvedRunConfig,
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
) -> tuple[ValidatedCheckpoint, EvaluationManifest]:
    """Load checkpoint and manifest inputs before constructing MuJoCo state.

    Historical contract, asset, and manifest fingerprints are informational;
    source/configuration edits do not invalidate these inputs.
    """

    manifest = load_manifest(manifest_path, config)
    checkpoint = load_checkpoint_preflight(checkpoint_path, config, mode="inference")
    return checkpoint, manifest


def deterministic_policy_action(policy: ActorCritic, observation: torch.Tensor) -> torch.Tensor:
    """Return the deterministic seven-joint arm action."""

    if observation.device.type != "cpu" or observation.shape != (106,):
        raise EvaluationError("deterministic evaluation observation must be a CPU tensor with shape (106,)")
    with torch.inference_mode():
        batch = observation.to(dtype=torch.float32).unsqueeze(0)
        action = policy.act_inference(batch).squeeze(0).to(dtype=torch.float32, device="cpu")
    if action.shape != (7,) or not bool(torch.isfinite(action).all()):
        raise EvaluationError("deterministic policy produced an invalid action")
    return action


def _overrides(case: EvaluationCase) -> TaskResetOverrides:
    return TaskResetOverrides(
        target_xy_m=case.target_xy_m,
        cylinder=ForcedCylinderParameters(
            radius_m=case.radius_m,
            height_m=case.height_m,
            mass_kg=case.mass_kg,
            hand_sliding_friction=case.hand_sliding_friction,
            table_sliding_friction=case.table_sliding_friction,
        ),
        target_authority=ForcedTargetAuthority.FROZEN_MANIFEST,
        cylinder_authority=ForcedParameterAuthority.FROZEN_MANIFEST,
    )


def replay_case(
    task: ReachThenCloseTask,
    policy: ActorCritic,
    case: EvaluationCase,
    *,
    training_seed: int,
) -> EpisodeEvaluation:
    """Replay exactly one frozen case using the shared task/Option/classifier."""

    task.reseed_reset_streams(case.seed)
    task.curriculum.restore(replace(task.curriculum.state(), stage=CurriculumStage.S3))
    task.reset_overrides = _overrides(case)
    observation = task.reset().values
    requested = False
    accepted = False
    terminal: TaskStepTransition | None = None
    for _ in range(task.config.task.scene.maximum_episode_steps):
        action = deterministic_policy_action(policy, observation)
        result = task.step(action, auto_reset=False)
        transition = result.transition
        requested |= transition.snapshot.trigger_requested
        accepted |= transition.snapshot.option_accepted_ever
        observation = result.observation.values
        if transition.done:
            terminal = transition
            break
    if terminal is None:
        raise EvaluationError(f"case {case.case_id} did not terminate within the configured episode limit")
    snapshot = terminal.snapshot
    applied = snapshot.randomization.get("cylinder")
    if not isinstance(applied, Mapping):
        raise EvaluationError("task snapshot omitted applied cylinder parameters")
    expected = {
        "radius_m": case.radius_m,
        "height_m": case.height_m,
        "mass_kg": case.mass_kg,
        "hand_sliding_friction": case.hand_sliding_friction,
        "table_sliding_friction": case.table_sliding_friction,
    }
    if any(float(applied[name]) != value for name, value in expected.items()):
        raise EvaluationError(f"case {case.case_id} replay substituted sampled cylinder parameters")
    if snapshot.cylinder_reset_position_m[:2] != case.target_xy_m:
        raise EvaluationError(f"case {case.case_id} replay substituted a sampled target")
    return EpisodeEvaluation(
        training_seed=training_seed,
        evaluation_seed=case.seed,
        case_id=case.case_id,
        target_x_m=case.target_xy_m[0],
        target_y_m=case.target_xy_m[1],
        radius_m=case.radius_m,
        height_m=case.height_m,
        mass_kg=case.mass_kg,
        hand_sliding_friction=case.hand_sliding_friction,
        table_sliding_friction=case.table_sliding_friction,
        trigger_requested=requested,
        option_accepted=accepted,
        lift_5cm=snapshot.lift_5cm,
        success=snapshot.success,
        unsafe=snapshot.unsafe,
        primary_reason=snapshot.primary_termination_reason,
        auxiliary_reasons=snapshot.secondary_termination_reasons,
        final_position_error_m=snapshot.position_error_m,
        final_orientation_error_rad=snapshot.orientation_error_rad,
        episode_steps=snapshot.episode_step,
    )


def evaluate_manifest(
    config: ResolvedRunConfig,
    checkpoint: ValidatedCheckpoint,
    manifest: EvaluationManifest,
) -> tuple[tuple[EpisodeEvaluation, ...], EvaluationSummary]:
    policy = _policy(config)
    model_state = checkpoint.payload["model_state_dict"]
    if not isinstance(model_state, Mapping):
        raise EvaluationError("checkpoint model state is not a mapping")
    policy.load_state_dict(cast(Mapping[str, object], model_state), strict=True)
    policy.eval()
    training = checkpoint.sidecar.get("training")
    if not isinstance(training, Mapping) or type(training.get("seed")) is not int:
        raise EvaluationError("checkpoint sidecar has no exact training seed")
    training_seed = cast(int, training["seed"])
    episodes: list[EpisodeEvaluation] = []
    # M12 exposes one task runtime; evaluation intentionally has no optimizer,
    # storage, PPO update, or curriculum gate mutation.
    with ReachThenCloseTask(
        config,
        seed=manifest.generation_seed,
        reset_overrides=_overrides(manifest.cases[0]),
        curriculum_updates_enabled=False,
    ) as task:
        for case in manifest.cases:
            episodes.append(replay_case(task, policy, case, training_seed=training_seed))
    result = tuple(episodes)
    return result, aggregate_episodes(result, manifest=manifest, training_seed=training_seed)


_CSV_FIELDS = (
    *(name for name in EpisodeEvaluation.__dataclass_fields__ if name != "auxiliary_reasons"),
    "auxiliary_reasons_json",
)


def write_evaluation_reports(
    output_directory: str | Path,
    *,
    command: str,
    checkpoint: ValidatedCheckpoint,
    checkpoint_path: str | Path,
    manifest: EvaluationManifest,
    manifest_path: str | Path,
    episodes: Sequence[EpisodeEvaluation],
    summary: EvaluationSummary,
) -> tuple[Path, Path, Path]:
    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    stem = f"{manifest.kind.value}_seed_{summary.training_seed}"
    json_path = target / f"{stem}.json"
    csv_path = target / f"{stem}.csv"
    text_path = target / f"{stem}.md"
    payload = {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "command": command,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "inference": {
            "actions": "actor_latent_mean_then_tanh",
            "option_handoff": "automatic_after_admission_dwell",
            "exploration_noise": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
        },
        "manifest": {
            "path": str(manifest_path),
            "kind": manifest.kind.value,
            "manifest_hash": manifest.manifest_hash,
            "file_sha256": file_sha256(manifest_path),
        },
        "summary": summary.to_dict(),
        "episodes": [item.to_dict() for item in episodes],
        "formal_gate_status": "not_evaluated_requires_three_complete_training_seeds",
    }
    json_path.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for episode in episodes:
            row = episode.to_dict()
            row["auxiliary_reasons_json"] = canonical_json(row.pop("auxiliary_reasons"))
            writer.writerow(row)
    metric_lines = "\n".join(
        f"- {name}: {metric.value:.6f} ({metric.numerator}/{metric.denominator})"
        for name, metric in (
            ("Option accepted", summary.option_accepted_rate),
            ("Lift5cm", summary.lift_5cm_rate),
            ("Complete success", summary.complete_success_rate),
            ("Dangerous", summary.dangerous_rate),
            ("P(success|accepted)", summary.success_given_accepted),
            ("P(unsafe|accepted)", summary.unsafe_given_accepted),
        )
    )
    failures = tuple(item for item in episodes if not item.success)
    unsafe = tuple(item for item in episodes if item.unsafe)
    detail_lines = "\n".join(
        f"- `{item.case_id}`: success={item.success}, unsafe={item.unsafe}, primary={item.primary_reason}, auxiliary={list(item.auxiliary_reasons)}"
        for item in failures
    )
    text_path.write_text(
        f"# Deterministic evaluation: {manifest.kind.value}\n\n"
        f"Command: `{command}`\n\n"
        f"Completed: {summary.completed_episodes}/{summary.expected_episodes}\n\n"
        f"{metric_lines}\n\n"
        f"Failures ({len(failures)}; unsafe {len(unsafe)})\n\n"
        f"{detail_lines or '- None'}\n\n"
        "This single-seed report does not claim a preliminary or final product gate.\n",
        encoding="utf-8",
        newline="\n",
    )
    return json_path, csv_path, text_path


def record_final_manifest_usage(
    path: str | Path,
    *,
    run_id: str,
    manifest: EvaluationManifest,
    checkpoint_bundle_id: str,
    command: str,
) -> str | None:
    """Append an auditable final-use record and warn on repeated dev-run use."""

    if manifest.kind is ManifestKind.VALIDATION:
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = 0
    if target.exists():
        for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"invalid final usage record line {line_number}: {error}") from error
            if not isinstance(item, Mapping) or item.get("schema_version") != FINAL_USAGE_SCHEMA_VERSION:
                raise EvaluationError(f"invalid final usage record schema at line {line_number}")
            if item.get("run_id") == run_id and item.get("manifest_hash") == manifest.manifest_hash:
                previous += 1
    record = {
        "schema_version": FINAL_USAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "manifest_kind": manifest.kind.value,
        "manifest_hash": manifest.manifest_hash,
        "checkpoint_bundle_id": checkpoint_bundle_id,
        "command": command,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "prior_consumptions_for_run_and_manifest": previous,
        "intended_use": "one_time_formal_final_evaluation_not_model_selection",
    }
    with target.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(record) + "\n")
    if previous:
        return (
            "FINAL MANIFEST AUDIT WARNING: this development run has consumed the same final manifest "
            f"{previous + 1} times; final sets must not be used for checkpoint selection or tuning."
        )
    return None


def _command(args: argparse.Namespace) -> str:
    parts = [
        sys.executable,
        "-m",
        "tienkung_rl.evaluate",
        "--checkpoint",
        str(args.checkpoint),
        "--manifest",
        str(args.manifest),
    ]
    for option, destination in (
        ("--config", "config"),
        ("--seed", "seed"),
        ("--output-dir", "output_dir"),
        ("--num-envs", "num_envs"),
        ("--worker-threads", "worker_threads"),
    ):
        value = getattr(args, destination, None)
        if value is not None:
            parts.extend((option, str(value)))
    for override in getattr(args, "config_overrides", ()):
        parts.extend(("--set", str(override)))
    return shlex.join(parts)


def run(config: ResolvedRunConfig, args: argparse.Namespace, *, stream: TextIO | None = None) -> int:
    """CLI runtime.  Preflight finishes before constructing the environment."""

    output = sys.stdout if stream is None else stream
    checkpoint_path = cast(Path, args.checkpoint)
    manifest_path = cast(Path, args.manifest)
    command = _command(args)
    checkpoint, manifest = preflight_evaluation_inputs(
        config,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )
    run_id = cast(str, checkpoint.payload["run_id"])
    # A run and iteration identify the checkpoint without a content hash.
    bundle_id = str(checkpoint.sidecar.get("bundle_id", f"{run_id}:{checkpoint.iteration}"))
    report_directory = Path(config.run.output_dir) / "evaluation"
    warning = record_final_manifest_usage(
        report_directory / "final_manifest_usage.jsonl",
        run_id=run_id,
        manifest=manifest,
        checkpoint_bundle_id=bundle_id,
        command=command,
    )
    if warning is not None:
        print(f"\n{'!' * 88}\n{warning}\n{'!' * 88}\n", file=output)
    episodes, summary = evaluate_manifest(config, checkpoint, manifest)
    paths = write_evaluation_reports(
        report_directory,
        command=command,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        manifest_path=manifest_path,
        episodes=episodes,
        summary=summary,
    )
    print(f"Command: {command}", file=output)
    print(f"Manifest: {manifest.kind.value} {manifest.manifest_hash}", file=output)
    print(f"Completed episodes: {summary.completed_episodes}/{summary.expected_episodes}", file=output)
    print("Reports: " + ", ".join(str(path) for path in paths), file=output)
    print("No three-seed preliminary/final quality gate is claimed by this invocation.", file=output)
    return 0


__all__ = [
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "EpisodeEvaluation",
    "EvaluationError",
    "EvaluationSummary",
    "GateDecision",
    "MetricRatio",
    "aggregate_episodes",
    "deterministic_policy_action",
    "evaluate_manifest",
    "final_gate",
    "preflight_evaluation_inputs",
    "preliminary_gate",
    "record_final_manifest_usage",
    "replay_case",
    "run",
    "write_evaluation_reports",
]
