"""Run a bounded end-to-end M15 CPU rollout/PPO configuration profile."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RSL_RL_ROOT = REPOSITORY_ROOT / "rsl_rl"
for search_path in (REPOSITORY_ROOT, RSL_RL_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import torch  # noqa: E402

from tienkung_rl.config import DEFAULT_CONFIG_PATH, load_run_config  # noqa: E402
from tienkung_rl.config.serialization import sha256_fingerprint  # noqa: E402
from tienkung_rl.config.models import ResolvedRunConfig  # noqa: E402
from tienkung_rl.learning.runner import (  # noqa: E402
    build_training_components,
    collect_rollout,
    gradient_diagnostics,
    seed_all,
)
from tienkung_rl.task import TaskVecEnv  # noqa: E402
from tienkung_rl.telemetry.metrics import TelemetryAggregator  # noqa: E402

PROFILE_SCHEMA_VERSION = "tienkung_end_to_end_training_profile_v1"


@dataclass(frozen=True, slots=True)
class ProfileCase:
    num_envs: int
    worker_threads: int
    rollout_steps_per_env: int
    mini_batches: int


CASES = (
    ProfileCase(1, 1, 4, 1),
    ProfileCase(2, 1, 4, 1),
    ProfileCase(2, 2, 4, 1),
    ProfileCase(2, 2, 8, 1),
    ProfileCase(2, 2, 8, 2),
)


def _proc_value(name: str) -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(":")
        if separator and key == name:
            number, unit = value.strip().split()[:2]
            if unit != "kB":
                raise RuntimeError(f"unexpected {name} unit {unit}")
            return int(number) * 1024
    raise RuntimeError(f"/proc/self/status has no {name}")


def _profile_case(base: ResolvedRunConfig, case: ProfileCase, seed: int) -> dict[str, object]:
    config = replace(
        base,
        platform=replace(base.platform, num_envs=case.num_envs, worker_threads=case.worker_threads),
        ppo=replace(
            base.ppo,
            rollout_steps_per_env=case.rollout_steps_per_env,
            mini_batches=case.mini_batches,
        ),
    )
    config.validate()
    seed_all(seed)
    before_rss = _proc_value("VmRSS")
    construction_start = time.perf_counter()
    components = build_training_components(config)
    with TaskVecEnv(config) as environment:
        observations, _ = environment.reset()
        construction_seconds = time.perf_counter() - construction_start
        sampling_start = time.perf_counter()
        collect_rollout(
            environment,
            components.ppo,
            observations,
            iteration=1,
            telemetry=TelemetryAggregator(),
        )
        sampling_seconds = time.perf_counter() - sampling_start
        learning_start = time.perf_counter()
        losses = components.ppo.update()
        learning_seconds = time.perf_counter() - learning_start
        actor_gradient_norm = gradient_diagnostics(components.policy)
        peak_rss = _proc_value("VmHWM")
    transitions = case.num_envs * case.rollout_steps_per_env
    total_seconds = sampling_seconds + learning_seconds
    return {
        **asdict(case),
        "learning_epochs": config.ppo.learning_epochs,
        "rollout_batch_size": transitions,
        "mini_batch_size": transitions // case.mini_batches,
        "config_fingerprint": sha256_fingerprint(config),
        "construction_and_reset_seconds": construction_seconds,
        "sampling_seconds": sampling_seconds,
        "learning_seconds": learning_seconds,
        "total_update_seconds": total_seconds,
        "rollout_transitions_per_second": transitions / sampling_seconds,
        "end_to_end_transitions_per_second": transitions / total_seconds,
        "physics_steps_per_second": transitions * config.physics.decimation / sampling_seconds,
        "rss_before_bytes": before_rss,
        "sampled_peak_rss_bytes": peak_rss,
        "actor_gradient_norm": actor_gradient_norm,
        "losses": {name: float(value) for name, value in losses.items()},
        "finite": bool(
            torch.isfinite(torch.tensor([*losses.values(), actor_gradient_norm])).all()
        ),
        "bottleneck": "sampling" if sampling_seconds >= learning_seconds else "learning",
    }


def _markdown(report: dict[str, object]) -> str:
    samples = report["samples"]
    assert isinstance(samples, list)
    lines = [
        "# M15 bounded end-to-end CPU training profile",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        (
            "Status: bounded engineering profile only. It is not a 24-hour run, not three-seed training, "
            "and provides no task-quality or acceptance-gate evidence."
        ),
        "",
        (
            "The profile uses the production TaskVecEnv, terminal-observation bootstrap, continuous PPO, "
            "and the configured network."
        ),
        "",
        "| envs | workers | rollout/env | mini-batches | batch | sample trans/s | end-to-end trans/s | sample s | learn s | peak RSS | bottleneck |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for sample in samples:
        assert isinstance(sample, dict)
        lines.append(
            "| {num_envs} | {worker_threads} | {rollout_steps_per_env} | {mini_batches} | "
            "{rollout_batch_size} | {rollout_transitions_per_second:.3f} | "
            "{end_to_end_transitions_per_second:.3f} | {sampling_seconds:.6f} | "
            "{learning_seconds:.6f} | {sampled_peak_rss_bytes} | {bottleneck} |".format(**sample)
        )
    lines.extend(
        [
            "",
            f"Bounded throughput candidate: `{report['bounded_throughput_candidate']}`.",
            "",
            f"Retained production default: `{report['retained_default']}`.",
            "",
            str(report["retained_default_reason"]),
            "",
            "Raw finite losses, actor gradient norms, memory, fingerprints, and timings are preserved in the JSON report.",
        ]
    )
    return "\n".join(lines) + "\n"


def _profile_case_argument(text: str) -> ProfileCase:
    try:
        values = tuple(int(value) for value in text.split(":"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must contain four integers") from error
    if len(values) != 4 or min(values) < 1:
        raise argparse.ArgumentTypeError("case must be ENVS:WORKERS:ROLLOUT_STEPS:MINI_BATCHES")
    case = ProfileCase(*values)
    if case.worker_threads > case.num_envs:
        raise argparse.ArgumentTypeError("case worker count cannot exceed environment count")
    if case.num_envs * case.rollout_steps_per_env % case.mini_batches:
        raise argparse.ArgumentTypeError("case rollout batch must divide evenly into mini-batches")
    return case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        type=_profile_case_argument,
        help="Profile ENVS:WORKERS:ROLLOUT_STEPS:MINI_BATCHES; repeat for multiple cases.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/profiling/m15_training_runner_profile.json"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/profiling/m15_training_runner_profile.md"),
    )
    args = parser.parse_args()
    base = load_run_config(args.config)
    cases = CASES if args.cases is None else tuple(args.cases)
    samples = [_profile_case(base, case, args.seed) for case in cases]
    candidate = max(samples, key=lambda item: cast(float, item["end_to_end_transitions_per_second"]))
    report: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "calibration_status": "bounded_engineering_profile_not_training_quality_evidence",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "machine": platform.machine(),
        "seed": args.seed,
        "cases_are_single_update": True,
        "formal_training_seed_count": 0,
        "twenty_four_hour_run_performed": False,
        "task_quality_measured": False,
        "samples": samples,
        "bounded_throughput_candidate": (
            f"envs={candidate['num_envs']}, workers={candidate['worker_threads']}, "
            f"rollout={candidate['rollout_steps_per_env']}, mini_batches={candidate['mini_batches']}"
        ),
        "retained_default": (
            f"envs={base.platform.num_envs}, workers={base.platform.worker_threads}, "
            f"rollout={base.ppo.rollout_steps_per_env}, mini_batches={base.ppo.mini_batches}"
        ),
        "retained_default_reason": (
            "This short single-update matrix isolates runtime throughput and memory, but does not measure "
            "time-to-quality or the three-seed acceptance gates. It therefore cannot justify changing the "
            "versioned production rollout batch contract."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(f"wrote {args.output_json} and {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
