"""Bounded native-MuJoCo profiling for the M05 parallel baseline.

This module records engineering measurements only.  It does not run PPO and
does not claim the 24-hour training optimization target has been calibrated.
"""

from __future__ import annotations

import gc
import json
import os
import platform
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-untyped]  # MuJoCo 3.11 does not ship a PEP 561 marker.
import numpy as np

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import sha256_fingerprint
from tienkung_rl.sim.parallel import ParallelStepRequest, ThreadPoolParallelExecutor


def _linux_status_value_bytes(name: str) -> int:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("parallel profiling requires Linux /proc/self/status") from error
    prefix = f"{name}:"
    for line in lines:
        if line.startswith(prefix):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise RuntimeError(f"unexpected {name} format in /proc/self/status")
            return int(fields[1]) * 1024
    raise RuntimeError(f"{name} is absent from /proc/self/status")


def _linux_thread_count() -> int:
    try:
        lines = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("parallel profiling requires Linux /proc/self/status") from error
    for line in lines:
        if line.startswith("Threads:"):
            return int(line.split()[1])
    raise RuntimeError("Threads is absent from /proc/self/status")


def _linux_memory_total_bytes() -> int:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError("parallel profiling requires Linux /proc/meminfo") from error
    for line in lines:
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal is absent from /proc/meminfo")


def _cpu_model() -> str:
    try:
        lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return platform.processor() or "unknown"
    for line in lines:
        if line.startswith("model name"):
            return line.split(":", maxsplit=1)[1].strip()
    return platform.processor() or "unknown"


class _PeakSampler:
    """Sample native RSS and thread count while model construction is active."""

    def __init__(self, interval_s: float = 0.005) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="parallel-profile-sampler")
        self.maximum_rss_bytes = _linux_status_value_bytes("VmRSS")
        self.maximum_thread_count = _linux_thread_count()

    def _sample(self) -> None:
        self.maximum_rss_bytes = max(
            self.maximum_rss_bytes,
            _linux_status_value_bytes("VmRSS"),
        )
        self.maximum_thread_count = max(self.maximum_thread_count, _linux_thread_count())

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._sample()

    def __enter__(self) -> _PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


@dataclass(frozen=True, slots=True)
class ParallelProfileSample:
    """Raw resource and throughput measurement for one environment count."""

    environment_count: int
    worker_threads: int
    warmup_policy_steps: int
    measured_policy_steps: int
    physics_substeps_per_policy_step: int
    construction_seconds: float
    reset_seconds: float
    measured_step_seconds: float
    mean_synchronous_step_seconds: float
    policy_transitions_per_second: float
    physics_steps_per_second: float
    rss_before_bytes: int
    rss_after_construction_bytes: int
    rss_after_close_bytes: int
    sampled_peak_rss_bytes: int
    thread_count_before: int
    thread_count_with_pool: int
    sampled_peak_thread_count: int
    thread_count_after_close: int
    buffer_allocation_count: int
    warning_count: int


@dataclass(frozen=True, slots=True)
class ParallelProfileReport:
    """Traceable short baseline and explicitly non-final recommendation."""

    schema_version: str
    generated_at_utc: str
    hostname: str
    cpu_model: str
    logical_cpu_count: int
    memory_total_bytes: int
    python_version: str
    mujoco_version: str
    config_fingerprint: str
    runtime_mjcf_path: str
    runtime_mjcf_sha256: str
    samples: tuple[ParallelProfileSample, ...]
    simulator_only_candidate_num_envs: int
    simulator_only_candidate_worker_threads: int
    simulator_only_candidate_reason: str
    retained_training_default_num_envs: int
    retained_training_default_worker_threads: int
    retained_training_default_reason: str
    calibration_status: str
    configuration_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_one_count(
    config: ResolvedRunConfig,
    *,
    environment_count: int,
    worker_threads: int,
    warmup_policy_steps: int,
    measured_policy_steps: int,
) -> ParallelProfileSample:
    rss_before = _linux_status_value_bytes("VmRSS")
    threads_before = _linux_thread_count()
    construction_started = time.perf_counter()
    with (
        _PeakSampler() as sampler,
        ThreadPoolParallelExecutor(
            config,
            num_envs=environment_count,
            worker_threads=worker_threads,
            seed=config.run.seed,
        ) as executor,
    ):
        construction_seconds = time.perf_counter() - construction_started
        rss_after_construction = _linux_status_value_bytes("VmRSS")
        threads_with_pool = _linux_thread_count()
        reset_started = time.perf_counter()
        initial = executor.reset()
        reset_seconds = time.perf_counter() - reset_started
        requests = tuple(
            ParallelStepRequest(
                result.env_id,
                np.array(result.state.actuator_targets_rad, dtype=np.float64, order="C", copy=True),
            )
            for result in initial.results
        )
        del initial
        gc.collect()
        for _ in range(warmup_policy_steps):
            executor.step(requests)
        measured_started = time.perf_counter()
        warning_count = 0
        for _ in range(measured_policy_steps):
            batch = executor.step(requests)
            warning_count += sum(len(result.diagnostics.warnings) for result in batch.results)
        measured_step_seconds = time.perf_counter() - measured_started
        buffer_allocation_count = executor.buffer_allocation_count
    rss_after_close = _linux_status_value_bytes("VmRSS")
    threads_after_close = _linux_thread_count()
    transitions = environment_count * measured_policy_steps
    physics_steps = transitions * config.physics.decimation
    return ParallelProfileSample(
        environment_count=environment_count,
        worker_threads=worker_threads,
        warmup_policy_steps=warmup_policy_steps,
        measured_policy_steps=measured_policy_steps,
        physics_substeps_per_policy_step=config.physics.decimation,
        construction_seconds=construction_seconds,
        reset_seconds=reset_seconds,
        measured_step_seconds=measured_step_seconds,
        mean_synchronous_step_seconds=measured_step_seconds / measured_policy_steps,
        policy_transitions_per_second=transitions / measured_step_seconds,
        physics_steps_per_second=physics_steps / measured_step_seconds,
        rss_before_bytes=rss_before,
        rss_after_construction_bytes=rss_after_construction,
        rss_after_close_bytes=rss_after_close,
        sampled_peak_rss_bytes=sampler.maximum_rss_bytes,
        thread_count_before=threads_before,
        thread_count_with_pool=threads_with_pool,
        sampled_peak_thread_count=sampler.maximum_thread_count,
        thread_count_after_close=threads_after_close,
        buffer_allocation_count=buffer_allocation_count,
        warning_count=warning_count,
    )


def profile_parallel_executor(
    config: ResolvedRunConfig,
    *,
    environment_counts: Sequence[int] = (1, 4, 128),
    worker_thread_counts: Sequence[int] | None = None,
    warmup_policy_steps: int = 1,
    measured_policy_steps: int = 3,
) -> ParallelProfileReport:
    """Measure a bounded environment/worker matrix with no policy or renderer."""

    counts = tuple(environment_counts)
    if not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("environment_counts must contain positive integers")
    if len(set(counts)) != len(counts):
        raise ValueError("environment_counts must not contain duplicates")
    worker_counts = None if worker_thread_counts is None else tuple(worker_thread_counts)
    if worker_counts is not None:
        if not worker_counts or any(type(count) is not int or count < 1 for count in worker_counts):
            raise ValueError("worker_thread_counts must contain positive integers")
        if len(set(worker_counts)) != len(worker_counts):
            raise ValueError("worker_thread_counts must not contain duplicates")
    if warmup_policy_steps < 0 or measured_policy_steps < 1:
        raise ValueError("profiling step counts must be non-negative/positive")

    samples: list[ParallelProfileSample] = []
    for count in counts:
        workers_for_count = (
            (min(config.platform.worker_threads, count),)
            if worker_counts is None
            else tuple(workers for workers in worker_counts if workers <= count)
        )
        for workers in workers_for_count:
            samples.append(
                _profile_one_count(
                    config,
                    environment_count=count,
                    worker_threads=workers,
                    warmup_policy_steps=warmup_policy_steps,
                    measured_policy_steps=measured_policy_steps,
                )
            )
            gc.collect()
    if not samples:
        raise ValueError("worker_thread_counts must include a value no larger than an environment count")

    memory_total = _linux_memory_total_bytes()
    eligible = tuple(sample for sample in samples if sample.sampled_peak_rss_bytes <= 0.85 * memory_total)
    if not eligible:
        raise RuntimeError("every measured environment count exceeded the 85% memory safety budget")
    simulator_candidate = max(eligible, key=lambda sample: sample.physics_steps_per_second)
    return ParallelProfileReport(
        schema_version="tienkung_parallel_profile_v1",
        generated_at_utc=datetime.now(UTC).isoformat(),
        hostname=platform.node(),
        cpu_model=_cpu_model(),
        logical_cpu_count=os.cpu_count() or 1,
        memory_total_bytes=memory_total,
        python_version=platform.python_version(),
        mujoco_version=mujoco.__version__,
        config_fingerprint=sha256_fingerprint(config),
        runtime_mjcf_path=config.asset.runtime_mjcf_path,
        runtime_mjcf_sha256=config.asset.runtime_mjcf_sha256,
        samples=tuple(samples),
        simulator_only_candidate_num_envs=simulator_candidate.environment_count,
        simulator_only_candidate_worker_threads=simulator_candidate.worker_threads,
        simulator_only_candidate_reason=(
            "Highest measured native physics-step throughput in the bounded environment/worker "
            "matrix whose sampled peak RSS stayed within 85% of host memory."
        ),
        retained_training_default_num_envs=config.platform.num_envs,
        retained_training_default_worker_threads=config.platform.worker_threads,
        retained_training_default_reason=(
            "This bounded simulator-only run does not include policy inference, rollout storage, "
            "PPO batch quality, or time-to-quality. Changing environment count would change the "
            "training batch contract, so the configured training baseline is retained "
            "until the M15 end-to-end PPO profile can justify a versioned change."
        ),
        calibration_status="short_engineering_baseline_not_final_training_calibration",
        configuration_changed=False,
    )


def render_parallel_profile_markdown(report: ParallelProfileReport) -> str:
    """Render the same raw measurements and recommendation for human review."""

    lines = [
        "# M05 native MuJoCo parallel baseline",
        "",
        f"Generated: `{report.generated_at_utc}`",
        "",
        (
            "Status: short engineering baseline only; this is not the final 24-hour training "
            "calibration and it does not change rollout or PPO batch contracts."
        ),
        "",
        "| envs | workers | transitions/s | physics steps/s | mean batch step (s) | peak RSS (bytes) | peak threads | warnings |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in report.samples:
        lines.append(
            f"| {sample.environment_count} | {sample.worker_threads} | "
            f"{sample.policy_transitions_per_second:.6f} | {sample.physics_steps_per_second:.6f} | "
            f"{sample.mean_synchronous_step_seconds:.6f} | {sample.sampled_peak_rss_bytes} | "
            f"{sample.sampled_peak_thread_count} | {sample.warning_count} |"
        )
    lines.extend(
        (
            "",
            f"Simulator-only throughput candidate: `{report.simulator_only_candidate_num_envs}` "
            f"environments with `{report.simulator_only_candidate_worker_threads}` workers.",
            "",
            f"Candidate reason: {report.simulator_only_candidate_reason}",
            "",
            f"Retained training default: `{report.retained_training_default_num_envs}` environments "
            f"with `{report.retained_training_default_worker_threads}` workers.",
            "",
            f"Training-default reason: {report.retained_training_default_reason}",
            "",
            "The profiling command did not change environment, rollout, or PPO batch configuration.",
            "",
        )
    )
    return "\n".join(lines)


def write_parallel_profile(
    report: ParallelProfileReport,
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    """Write matching machine-readable raw data and a concise decision record."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_parallel_profile_markdown(report), encoding="utf-8")


__all__ = [
    "ParallelProfileReport",
    "ParallelProfileSample",
    "profile_parallel_executor",
    "render_parallel_profile_markdown",
    "write_parallel_profile",
]
