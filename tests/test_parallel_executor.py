"""Real-MuJoCo tests for the deterministic in-process parallel executor."""

from __future__ import annotations

import gc
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import (
    ActuatorControlRanges,
    BatchStepResult,
    ContactFrame,
    CylinderParameters,
    EnvironmentBackend,
    MujocoEnvironmentBackend,
    ParallelExecutorClosedError,
    ParallelResetRequest,
    ParallelStepRequest,
    ParallelWorkerError,
    PhysicsState,
    PhysicsStepResult,
    ResetSpec,
    SimulationDiagnostics,
    SimulationWarning,
    SiteHingeKinematics,
    ThreadPoolParallelExecutor,
)
from tienkung_rl.validation.parallel_profile import (
    profile_parallel_executor,
    write_parallel_profile,
)


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config()


def _targets(state: PhysicsState) -> np.ndarray:
    return np.array(state.actuator_targets_rad, dtype=np.float64, order="C", copy=True)


def _thread_names(prefix: str) -> set[str]:
    return {thread.name for thread in threading.enumerate() if thread.name.startswith(prefix)}


@dataclass(slots=True)
class _Controls:
    reset_delay_s: float = 0.0
    step_delay_s: float = 0.0
    fail_next_reset: bool = False
    fail_next_step: bool = False
    fail_close: bool = False
    inject_warning: bool = False
    work_started: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)
    completed_resets: list[int] = field(default_factory=list)
    completion_lock: threading.Lock = field(default_factory=threading.Lock)


class _ControlledMujocoBackend:
    """Fault/delay harness around a genuine independent MuJoCo backend."""

    def __init__(self, config: ResolvedRunConfig, env_id: int, controls: _Controls) -> None:
        self._env_id = env_id
        self._controls = controls
        self._inner = MujocoEnvironmentBackend(config)
        # Expose opaque identities to the executor's mutable-sharing audit.
        self._model: object | None = self._inner._model
        self._data: object | None = self._inner._data

    def reset(self, reset_spec: ResetSpec) -> PhysicsState:
        if self._controls.fail_next_reset:
            self._controls.fail_next_reset = False
            raise RuntimeError("injected reset crash")
        if self._controls.reset_delay_s:
            time.sleep(self._controls.reset_delay_s)
        state = self._inner.reset(reset_spec)
        with self._controls.completion_lock:
            self._controls.completed_resets.append(self._env_id)
        return state

    def apply_targets(self, targets_rad: NDArray[np.float64]) -> None:
        self._inner.apply_targets(targets_rad)

    def step_substeps(self, count: int) -> PhysicsStepResult:
        self._controls.work_started.set()
        if self._controls.step_delay_s:
            time.sleep(self._controls.step_delay_s)
        if self._controls.fail_next_step:
            self._controls.fail_next_step = False
            raise RuntimeError("injected worker crash")
        result = self._inner.step_substeps(count)
        if not self._controls.inject_warning:
            return result
        warning = SimulationWarning(
            code=77,
            name="injected_warning",
            message="warning propagated from worker",
            occurrence_count=1,
            last_info=self._env_id,
            physics_step=result.diagnostics.physics_step,
            config_fingerprint=result.diagnostics.config_fingerprint,
        )
        diagnostics = replace(result.diagnostics, warnings=(*result.diagnostics.warnings, warning))
        return replace(result, diagnostics=diagnostics)

    def read_state(self) -> PhysicsState:
        return self._inner.read_state()

    def read_contacts(self) -> ContactFrame:
        return self._inner.read_contacts()

    def read_warnings(self) -> SimulationDiagnostics:
        return self._inner.read_warnings()

    def read_actuator_control_ranges(self) -> ActuatorControlRanges:
        return self._inner.read_actuator_control_ranges()

    def read_site_hinge_kinematics(self, site_name: str, joint_name: str) -> SiteHingeKinematics:
        return self._inner.read_site_hinge_kinematics(site_name, joint_name)

    def read_arm_site_jacobian(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._inner.read_arm_site_jacobian(site_name)

    def close(self) -> None:
        self._inner.close()
        self._model = None
        self._data = None
        self._controls.closed.set()
        if self._controls.fail_close:
            raise RuntimeError("injected close crash")


class _SampledReset:
    def __init__(
        self,
        delays_s: Mapping[int, float] | None = None,
        *,
        fail_env_id: int | None = None,
    ) -> None:
        self._delays_s = {} if delays_s is None else dict(delays_s)
        self._fail_env_id = fail_env_id

    def __call__(self, env_id: int, rng: np.random.Generator) -> ResetSpec:
        values = rng.uniform(size=5)
        delay = self._delays_s.get(env_id, 0.0)
        if delay:
            time.sleep(delay)
        if env_id == self._fail_env_id:
            raise RuntimeError("reset factory failure after RNG consumption")
        return ResetSpec(
            cylinder=CylinderParameters(
                radius_m=0.010 + 0.005 * float(values[0]),
                height_m=0.080 + 0.040 * float(values[1]),
                mass_kg=0.100 + 0.030 * float(values[2]),
                hand_sliding_friction=0.7 + 0.4 * float(values[3]),
                table_sliding_friction=0.7 + 0.4 * float(values[4]),
            )
        )


def _controlled_factory(
    config: ResolvedRunConfig,
    controls: Mapping[int, _Controls],
) -> Callable[[int], EnvironmentBackend]:
    def factory(env_id: int) -> EnvironmentBackend:
        return _ControlledMujocoBackend(config, env_id, controls[env_id])

    return factory


def test_real_models_data_backends_and_rng_are_independent_and_results_sorted(
    config: ResolvedRunConfig,
) -> None:
    with ThreadPoolParallelExecutor(config, num_envs=2, worker_threads=2, seed=91) as executor:
        identities = executor.environment_identities
        for attribute in (
            "backend_identity",
            "model_identity",
            "data_identity",
            "rng_identity",
        ):
            assert len({getattr(item, attribute) for item in identities}) == 2

        reset = executor.reset(
            (
                ParallelResetRequest(
                    1,
                    ResetSpec(cylinder=CylinderParameters(radius_m=0.015)),
                ),
                ParallelResetRequest(
                    0,
                    ResetSpec(cylinder=CylinderParameters(radius_m=0.012)),
                ),
            )
        )
        assert tuple(result.env_id for result in reset.results) == (0, 1)
        np.testing.assert_array_equal(reset.state.env_ids, (0, 1))
        np.testing.assert_allclose(reset.state.cylinder_parameters[:, 0], (0.012, 0.015))

        before_env1_rng = executor.rng_manifest()[1]
        env1_targets = _targets(reset.results[1].state)
        executor.reset(
            (
                ParallelResetRequest(
                    0,
                    ResetSpec(cylinder=CylinderParameters(radius_m=0.015)),
                ),
            )
        )
        env1_step = executor.step((ParallelStepRequest(1, env1_targets),))
        assert env1_step.results[0].state.cylinder_parameters.radius_m == pytest.approx(0.015)
        assert executor.rng_manifest()[1] == before_env1_rng


def test_barrier_order_readonly_contiguous_double_buffers_and_policy_steps(
    config: ResolvedRunConfig,
) -> None:
    controls = {0: _Controls(step_delay_s=0.04), 1: _Controls()}
    factory = _controlled_factory(config, controls)
    with ThreadPoolParallelExecutor(
        config,
        num_envs=2,
        worker_threads=2,
        backend_factory=factory,
    ) as executor:
        initial = executor.reset()
        initial_positions = initial.state.joint_positions_rad.copy()
        requests = tuple(
            ParallelStepRequest(result.env_id, _targets(result.state)) for result in reversed(initial.results)
        )
        first = executor.step(requests)
        assert tuple(result.env_id for result in first.results) == (0, 1)
        assert all(result.substeps_executed == 4 for result in first.results)
        assert all(result.physics_step == 4 for result in first.results)
        assert first.state.generation > initial.state.generation
        assert executor.buffer_allocation_count == 2
        for value in vars(type(first.state)).get("__slots__", ()):
            array = getattr(first.state, value, None)
            if isinstance(array, np.ndarray):
                assert array.flags.c_contiguous
                assert not array.flags.writeable
        np.testing.assert_array_equal(initial.state.joint_positions_rad, initial_positions)

        del initial
        gc.collect()
        second = executor.step(requests)
        assert executor.buffer_allocation_count == 2
        np.testing.assert_array_equal(first.state.episode_physics_steps, (4, 4))
        np.testing.assert_array_equal(second.state.episode_physics_steps, (8, 8))


def test_completion_order_does_not_change_sorted_results_or_rng_replay(
    config: ResolvedRunConfig,
) -> None:
    completion_order: list[int] = []
    completion_lock = threading.Lock()
    controls = {
        0: _Controls(completed_resets=completion_order, completion_lock=completion_lock),
        1: _Controls(completed_resets=completion_order, completion_lock=completion_lock),
    }
    factory = _controlled_factory(config, controls)
    with ThreadPoolParallelExecutor(
        config,
        num_envs=2,
        worker_threads=2,
        seed=12345,
        backend_factory=factory,
    ) as executor:
        baseline_manifest = executor.rng_manifest()
        first = executor.reset(reset_spec_factory=_SampledReset({0: 0.06, 1: 0.0}))
        assert completion_order[-2:] == [1, 0]
        first_parameters = first.state.cylinder_parameters.copy()
        first_after = executor.rng_manifest()

        executor.restore_rng_manifest(baseline_manifest)
        completion_order.clear()
        second = executor.reset(reset_spec_factory=_SampledReset({0: 0.0, 1: 0.06}))
        assert completion_order == [0, 1]
        np.testing.assert_array_equal(second.state.cylinder_parameters, first_parameters)
        assert executor.rng_manifest() == first_after

        before_failed_reset = executor.rng_manifest()
        with pytest.raises(ParallelWorkerError, match="reset factory failure") as caught:
            executor.reset(
                (ParallelResetRequest(0),),
                reset_spec_factory=_SampledReset(fail_env_id=0),
            )
        assert caught.value.failures[0].env_id == 0
        assert executor.rng_manifest() == before_failed_reset


def test_frequent_partial_resets_and_worker_failure_do_not_cross_contaminate(
    config: ResolvedRunConfig,
) -> None:
    controls = {0: _Controls(), 1: _Controls()}
    factory = _controlled_factory(config, controls)
    with ThreadPoolParallelExecutor(
        config,
        num_envs=2,
        worker_threads=2,
        seed=7,
        backend_factory=factory,
    ) as executor:
        for index in range(12):
            env_id = index % 2
            batch = executor.reset(
                (ParallelResetRequest(env_id),),
                reset_spec_factory=_SampledReset(),
            )
            parameters = batch.results[0].state.cylinder_parameters
            assert 0.010 <= parameters.radius_m <= 0.015
            assert 0.080 <= parameters.height_m <= 0.120
            assert 0.100 <= parameters.mass_kg <= 0.130

        stable = executor.reset()
        controls[1].fail_next_step = True
        with pytest.raises(ParallelWorkerError) as caught:
            executor.step(
                tuple(ParallelStepRequest(result.env_id, _targets(result.state)) for result in stable.results)
            )
        failure = caught.value.failures[0]
        assert failure.env_id == 1
        assert failure.exception_type == "RuntimeError"
        assert failure.physics_step >= 0
        assert failure.config_fingerprint == executor.config_fingerprint
        assert "injected worker crash" in failure.message

        # Env 0 completed despite the peer failure and remains independently usable.
        recovered = executor.reset((ParallelResetRequest(1),))
        assert recovered.results[0].env_id == 1
        assert recovered.results[0].physics_step >= 0


def test_warning_propagation_pending_close_crash_cleanup_and_repeated_lifecycle(
    config: ResolvedRunConfig,
) -> None:
    controls = {0: _Controls(step_delay_s=0.08, inject_warning=True)}
    factory = _controlled_factory(config, controls)
    executor = ThreadPoolParallelExecutor(
        config,
        num_envs=1,
        worker_threads=1,
        backend_factory=factory,
    )
    prefix = executor.thread_name_prefix
    initial = executor.reset()
    outcome: list[object] = []

    def run_pending_step() -> None:
        try:
            outcome.append(executor.step((ParallelStepRequest(0, _targets(initial.results[0].state)),)))
        except Exception as error:  # pragma: no cover - assertion below reports the unexpected error.
            outcome.append(error)

    client = threading.Thread(target=run_pending_step, name="parallel-test-client")
    client.start()
    assert controls[0].work_started.wait(timeout=2.0)
    executor.close()
    client.join(timeout=2.0)
    assert not client.is_alive()
    assert len(outcome) == 1
    result = outcome[0]
    assert not isinstance(result, Exception)
    step_result = cast(BatchStepResult, result)
    warning = step_result.results[0].diagnostics.warnings[-1]
    assert warning.name == "injected_warning"
    assert warning.config_fingerprint == executor.config_fingerprint
    assert controls[0].closed.is_set()
    assert not _thread_names(prefix)
    with pytest.raises(ParallelExecutorClosedError):
        executor.reset()

    failing_controls = {0: _Controls(fail_close=True)}
    failing_factory = _controlled_factory(config, failing_controls)
    crashing = ThreadPoolParallelExecutor(
        config,
        num_envs=1,
        worker_threads=1,
        backend_factory=failing_factory,
    )
    crashing_prefix = crashing.thread_name_prefix
    with pytest.raises(ParallelWorkerError, match="injected close crash"):
        crashing.close()
    assert crashing.is_closed
    assert failing_controls[0].closed.is_set()
    assert not _thread_names(crashing_prefix)

    # A new pool after both normal and exceptional close owns fresh threads.
    with ThreadPoolParallelExecutor(config, num_envs=1, worker_threads=1) as fresh:
        assert fresh.reset().results[0].env_id == 0
        fresh_prefix = fresh.thread_name_prefix
    assert not _thread_names(fresh_prefix)


def test_bounded_profile_records_raw_throughput_memory_threads_and_reason(
    config: ResolvedRunConfig,
    tmp_path: Path,
) -> None:
    report = profile_parallel_executor(
        config,
        environment_counts=(1,),
        warmup_policy_steps=0,
        measured_policy_steps=1,
    )
    assert report.schema_version == "tienkung_parallel_profile_v1"
    assert report.calibration_status == "short_engineering_baseline_not_final_training_calibration"
    assert report.configuration_changed is False
    assert report.simulator_only_candidate_num_envs == 1
    assert report.retained_training_default_num_envs == config.platform.num_envs
    assert report.retained_training_default_worker_threads == config.platform.worker_threads
    sample = report.samples[0]
    assert sample.environment_count == sample.worker_threads == 1
    assert sample.physics_substeps_per_policy_step == 4
    assert sample.physics_steps_per_second > 0.0
    assert sample.policy_transitions_per_second > 0.0
    assert sample.mean_synchronous_step_seconds > 0.0
    assert sample.sampled_peak_rss_bytes >= sample.rss_before_bytes > 0
    assert sample.sampled_peak_thread_count >= sample.thread_count_before > 0
    assert sample.thread_count_after_close <= sample.sampled_peak_thread_count
    assert sample.warning_count == 0

    json_path = tmp_path / "profile.json"
    markdown_path = tmp_path / "profile.md"
    write_parallel_profile(report, json_path=json_path, markdown_path=markdown_path)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["config_fingerprint"] == report.config_fingerprint
    assert raw["runtime_mjcf_sha256"] == config.asset.runtime_mjcf_sha256
    assert raw["samples"][0]["environment_count"] == 1
    decision = markdown_path.read_text(encoding="utf-8")
    assert "short engineering baseline only" in decision
    assert "Simulator-only throughput candidate" in decision
    assert "Retained training default" in decision


def test_bounded_profile_can_measure_an_environment_worker_matrix(
    config: ResolvedRunConfig,
) -> None:
    report = profile_parallel_executor(
        config,
        environment_counts=(2,),
        worker_thread_counts=(1, 2),
        warmup_policy_steps=0,
        measured_policy_steps=1,
    )
    assert tuple((sample.environment_count, sample.worker_threads) for sample in report.samples) == (
        (2, 1),
        (2, 2),
    )


def test_parallel_named_queries_are_sorted_readonly_and_model_local(
    config: ResolvedRunConfig,
) -> None:
    with ThreadPoolParallelExecutor(config, num_envs=3, worker_threads=2) as executor:
        executor.reset()
        ranges = executor.read_actuator_control_ranges((2, 0, 1))
        assert tuple(item.env_id for item in ranges) == (0, 1, 2)
        assert len({id(item.ranges.ranges_rad) for item in ranges}) == 3
        assert all(not item.ranges.ranges_rad.flags.writeable for item in ranges)
        kinematics = executor.read_site_hinge_kinematics(
            "right_tcp_site",
            "elbow_pitch_r_joint",
            (2, 0),
        )
        assert tuple(item.env_id for item in kinematics) == (0, 2)
        assert len({id(item.kinematics.site_jacobian_6d) for item in kinematics}) == 2
        assert all(not item.kinematics.site_jacobian_6d.flags.writeable for item in kinematics)
