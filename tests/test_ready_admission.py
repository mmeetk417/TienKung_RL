"""M08 pure Ready and Option-admission contract tests."""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tienkung_rl.config.load import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.task.admission import (
    ADMISSION_CONDITION_KEYS,
    ADMISSION_DWELL_STEPS,
    ADMISSION_THRESHOLD_CALIBRATION_STATUS,
    ADMISSION_THRESHOLD_VERSION,
    ILLEGAL_CONTACT_IMPULSE_CALIBRATION_KEY,
    POLICY_STEP_S,
    READY_CONDITION_KEYS,
    READY_DWELL_STEPS,
    AdmissionEpisodeState,
    AdmissionInput,
    ReadyEpisodeState,
    ReadyInput,
    evaluate_admission,
    step_admission,
    step_ready,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config(ROOT / "configs/reach_then_close.yaml")


def _admission_input(**overrides: object) -> AdmissionInput:
    values: dict[str, object] = {
        "state_finite": True,
        "target_valid": True,
        "reach_phase": True,
        "position_error_m": 0.0,
        "orientation_error_rad": 0.0,
        "grasp_center_speed_m_s": 0.0,
        "arm_speed_max_rad_s": 0.0,
        "object_displacement_m": 0.0,
        "object_tilt_rad": 0.0,
        "object_linear_speed_m_s": 0.0,
        "object_angular_speed_rad_s": 0.0,
        "illegal_contact_impulse_n_s": 0.0,
    }
    values.update(overrides)
    return AdmissionInput(**values)  # type: ignore[arg-type]


def _ready_input(**overrides: object) -> ReadyInput:
    values = dataclasses.asdict(_admission_input(**overrides))
    values.pop("reach_phase")
    return ReadyInput(**values)


ADMISSION_BOUNDARIES = (
    ("position_error_m", "position_error_m", "position_error"),
    ("orientation_error_rad", "orientation_error_rad", "orientation_error"),
    ("grasp_center_speed_m_s", "grasp_center_speed_m_s", "grasp_center_speed"),
    ("object_tilt_rad", "object_tilt_rad", "object_tilt"),
    ("illegal_contact_impulse_n_s", "illegal_contact_impulse_n_s", "illegal_contact_impulse"),
)

READY_BOUNDARIES = (
    ("position_error_m", "position_error_m", "position_error"),
    ("orientation_error_rad", "orientation_error_rad", "orientation_error"),
    ("grasp_center_speed_m_s", "grasp_center_speed_m_s", "grasp_center_speed"),
    ("arm_speed_max_rad_s", "arm_speed_rad_s", "arm_joint_speed"),
    ("object_displacement_m", "object_displacement_m", "object_displacement"),
    ("object_tilt_rad", "object_tilt_rad", "object_tilt"),
    ("object_linear_speed_m_s", "object_linear_speed_m_s", "object_linear_speed"),
    ("object_angular_speed_rad_s", "object_angular_speed_rad_s", "object_angular_speed"),
    ("illegal_contact_impulse_n_s", "contact_impulse_n_s", "illegal_contact_impulse"),
)


def test_minimal_scalar_inputs_have_no_simulator_handles() -> None:
    admission_fields = tuple(field.name for field in dataclasses.fields(AdmissionInput))
    ready_fields = tuple(field.name for field in dataclasses.fields(ReadyInput))
    assert admission_fields == (
        "state_finite",
        "target_valid",
        "reach_phase",
        "position_error_m",
        "orientation_error_rad",
        "grasp_center_speed_m_s",
        "arm_speed_max_rad_s",
        "object_displacement_m",
        "object_tilt_rad",
        "object_linear_speed_m_s",
        "object_angular_speed_rad_s",
        "illegal_contact_impulse_n_s",
    )
    assert ready_fields == admission_fields[:2] + admission_fields[3:]
    module = inspect.getmodule(evaluate_admission)
    assert module is not None
    assert "mujoco" not in module.__dict__


def test_stable_eight_condition_order_and_error_metrics(config: ResolvedRunConfig) -> None:
    inputs = _admission_input(position_error_m=0.01, orientation_error_rad=0.02)
    result = evaluate_admission(inputs, config.task.admission)
    assert tuple(item.key for item in result.conditions) == ADMISSION_CONDITION_KEYS
    assert len(result.conditions) == 8
    assert all(item.passed for item in result.conditions)
    assert result.admissible
    assert result.option_admissible
    assert result.failure_reasons == ()
    assert result.metrics.position_error_m == 0.01
    assert result.metrics.orientation_error_rad == 0.02
    assert result.condition_passed("position_error")
    assert tuple(result.condition_flags) == ADMISSION_CONDITION_KEYS
    with pytest.raises(TypeError):
        result.condition_flags["position_error"] = False  # type: ignore[index]
    with pytest.raises(KeyError):
        result.condition_passed("unstable_unknown_key")


def test_admission_config_excludes_removed_speed_and_displacement_limits(config: ResolvedRunConfig) -> None:
    admission_fields = {field.name for field in dataclasses.fields(config.task.admission)}
    assert admission_fields.isdisjoint(
        {"arm_speed_rad_s", "object_displacement_m", "object_linear_speed_m_s", "object_angular_speed_rad_s"}
    )


def test_all_failures_are_reported_without_short_circuit(config: ResolvedRunConfig) -> None:
    limits = config.task.admission
    inputs = _admission_input(
        state_finite=False,
        target_valid=False,
        reach_phase=False,
        position_error_m=limits.position_error_m + 1.0,
        orientation_error_rad=limits.orientation_error_rad + 1.0,
        grasp_center_speed_m_s=limits.grasp_center_speed_m_s + 1.0,
        object_tilt_rad=limits.object_tilt_rad + 1.0,
        illegal_contact_impulse_n_s=limits.illegal_contact_impulse_n_s + 1.0,
    )
    result = evaluate_admission(inputs, limits)
    assert not result.admissible
    assert result.failure_reasons == ADMISSION_CONDITION_KEYS
    assert all(not item.passed for item in result.conditions)


@pytest.mark.parametrize(("input_name", "config_name", "condition_key"), ADMISSION_BOUNDARIES)
def test_each_admission_threshold_is_inclusive(
    config: ResolvedRunConfig, input_name: str, config_name: str, condition_key: str
) -> None:
    threshold = float(getattr(config.task.admission, config_name))
    below = math.nextafter(threshold, -math.inf)
    above = math.nextafter(threshold, math.inf)
    for value, expected in ((below, True), (threshold, True), (above, False)):
        result = evaluate_admission(_admission_input(**{input_name: value}), config.task.admission)
        assert result.condition_passed(condition_key) is expected
        assert result.admissible is expected


@pytest.mark.parametrize(
    "input_name",
    ("arm_speed_max_rad_s", "object_linear_speed_m_s", "object_angular_speed_rad_s", "object_displacement_m"),
)
def test_removed_admission_conditions_do_not_gate_admission(config: ResolvedRunConfig, input_name: str) -> None:
    result = evaluate_admission(_admission_input(**{input_name: 1e6}), config.task.admission)
    assert result.admissible
    removed_keys = {"arm_joint_speed", "object_displacement", "object_linear_speed", "object_angular_speed"}
    assert removed_keys.isdisjoint(tuple(item.key for item in result.conditions))


@pytest.mark.parametrize(("input_name", "config_name", "condition_key"), READY_BOUNDARIES)
def test_each_ready_threshold_is_inclusive(
    config: ResolvedRunConfig, input_name: str, config_name: str, condition_key: str
) -> None:
    threshold = float(getattr(config.task.ready, config_name))
    below = math.nextafter(threshold, -math.inf)
    above = math.nextafter(threshold, math.inf)
    for value, expected in ((below, True), (threshold, True), (above, False)):
        result, _ = step_ready(_ready_input(**{input_name: value}), ReadyEpisodeState(), config.task.ready)
        assert result.condition_passed(condition_key) is expected
        assert result.instantaneous is expected


@pytest.mark.parametrize("flag", ("state_finite", "target_valid", "reach_phase"))
def test_each_admission_prerequisite_fails_independently(config: ResolvedRunConfig, flag: str) -> None:
    result = evaluate_admission(_admission_input(**{flag: False}), config.task.admission)
    expected_key = {"state_finite": "finite_state", "target_valid": "valid_target", "reach_phase": "reach_phase"}[flag]
    assert result.failure_reasons == (expected_key,)


@pytest.mark.parametrize("input_name", tuple(item[0] for item in ADMISSION_BOUNDARIES))
@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
def test_nonfinite_metrics_fail_closed_and_keep_diagnostics(
    config: ResolvedRunConfig, input_name: str, nonfinite: float
) -> None:
    result = evaluate_admission(_admission_input(**{input_name: nonfinite}), config.task.admission)
    condition_key = next(key for name, _, key in ADMISSION_BOUNDARIES if name == input_name)
    assert not result.admissible
    assert result.condition_passed("finite_state") is False
    assert result.condition_passed(condition_key) is False
    assert "finite_state" in result.failure_reasons
    assert condition_key in result.failure_reasons


def test_negative_measurement_fails_its_named_condition(config: ResolvedRunConfig) -> None:
    result = evaluate_admission(_admission_input(object_tilt_rad=-1e-12), config.task.admission)
    assert result.failure_reasons == ("object_tilt",)


def test_non_reach_and_invalid_target_return_both_reasons(config: ResolvedRunConfig) -> None:
    result = evaluate_admission(
        _admission_input(target_valid=False, reach_phase=False),
        config.task.admission,
    )
    assert result.failure_reasons == ("valid_target", "reach_phase")


def test_ready_uses_eleven_non_phase_conditions(config: ResolvedRunConfig) -> None:
    result, _ = step_ready(_ready_input(), ReadyEpisodeState(), config.task.ready)
    assert tuple(item.key for item in result.conditions) == READY_CONDITION_KEYS
    assert "reach_phase" not in READY_CONDITION_KEYS
    assert result.instantaneous
    assert not result.ready


def test_ready_nonfinite_and_invalid_prerequisites_fail_together(config: ResolvedRunConfig) -> None:
    result, next_state = step_ready(
        _ready_input(state_finite=False, target_valid=False, object_angular_speed_rad_s=math.nan),
        ReadyEpisodeState(consecutive_steps=14),
        config.task.ready,
    )
    assert result.failure_reasons == ("finite_state", "valid_target", "object_angular_speed")
    assert not result.instantaneous
    assert not result.ready
    assert next_state == ReadyEpisodeState()


def test_ready_requires_fifteen_continuous_steps_and_latches_once(config: ResolvedRunConfig) -> None:
    state = ReadyEpisodeState()
    for step in range(1, READY_DWELL_STEPS):
        result, state = step_ready(_ready_input(), state, config.task.ready)
        assert result.consecutive_steps == step
        assert not result.ready
        assert not result.ever_ready

    result, state = step_ready(_ready_input(), state, config.task.ready)
    assert result.ready
    assert result.newly_ready
    assert result.ever_ready
    assert state == ReadyEpisodeState(consecutive_steps=15, ever_ready=True)

    repeated, state = step_ready(_ready_input(), state, config.task.ready)
    assert repeated.ready
    assert not repeated.newly_ready
    assert repeated.ever_ready
    assert state.consecutive_steps == 16


def test_admission_dwell_automatically_becomes_admissible_once(config: ResolvedRunConfig) -> None:
    state = AdmissionEpisodeState()
    for step in range(1, ADMISSION_DWELL_STEPS):
        result, state = step_admission(_admission_input(), state, config.task.admission)
        assert result.instantaneous_admissible
        assert result.consecutive_steps == step
        assert not result.option_admissible
        assert not result.newly_admissible

    result, state = step_admission(_admission_input(), state, config.task.admission)
    assert result.option_admissible
    assert result.newly_admissible
    assert state == AdmissionEpisodeState(consecutive_steps=15, ever_admissible=True)

    repeated, _ = step_admission(_admission_input(), state, config.task.admission)
    assert repeated.option_admissible
    assert not repeated.newly_admissible


def test_ready_failure_clears_dwell_but_preserves_ever_ready(config: ResolvedRunConfig) -> None:
    state = ReadyEpisodeState(consecutive_steps=22, ever_ready=True)
    result, next_state = step_ready(
        _ready_input(grasp_center_speed_m_s=config.task.ready.grasp_center_speed_m_s + 1e-6),
        state,
        config.task.ready,
    )
    assert not result.instantaneous
    assert not result.ready
    assert not result.newly_ready
    assert result.failure_reasons == ("grasp_center_speed",)
    assert next_state == ReadyEpisodeState(consecutive_steps=0, ever_ready=True)


def test_ready_rejects_policy_timing_drift(config: ResolvedRunConfig) -> None:
    assert POLICY_STEP_S == 0.020
    assert READY_DWELL_STEPS == 15
    with pytest.raises(ValueError, match=r"0\.020"):
        step_ready(_ready_input(), ReadyEpisodeState(), config.task.ready, policy_step_s=0.01)


def test_identical_inputs_are_deterministic_and_immutable(config: ResolvedRunConfig) -> None:
    inputs = _admission_input(position_error_m=0.01)
    before = dataclasses.asdict(inputs)
    first = evaluate_admission(inputs, config.task.admission)
    second = evaluate_admission(inputs, config.task.admission)
    assert first == second
    assert dataclasses.asdict(inputs) == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        inputs.position_error_m = 0.0  # type: ignore[misc]

    ready_inputs = _ready_input(position_error_m=0.001)
    state = ReadyEpisodeState(consecutive_steps=7)
    first_step = step_ready(ready_inputs, state, config.task.ready)
    second_step = step_ready(ready_inputs, state, config.task.ready)
    assert first_step == second_step
    assert state == ReadyEpisodeState(consecutive_steps=7)


def test_result_contract_has_no_control_reward_termination_or_curriculum_outputs(config: ResolvedRunConfig) -> None:
    admission_result = evaluate_admission(_admission_input(), config.task.admission)
    ready_result, _ = step_ready(_ready_input(), ReadyEpisodeState(), config.task.ready)
    forbidden_fragments = ("trigger", "option", "script", "reward", "terminate", "curriculum")
    for result in (admission_result, ready_result):
        field_names = tuple(field.name for field in dataclasses.fields(result))
        assert all(fragment not in name for name in field_names for fragment in forbidden_fragments)


def test_thresholds_remain_versioned_unvalidated_calibration_starts(config: ResolvedRunConfig) -> None:
    result = evaluate_admission(_admission_input(), config.task.admission)
    assert result.threshold_version == ADMISSION_THRESHOLD_VERSION
    assert result.calibration_status == "starting_point_unvalidated"
    assert ADMISSION_THRESHOLD_CALIBRATION_STATUS == "starting_point_unvalidated"
    assert ILLEGAL_CONTACT_IMPULSE_CALIBRATION_KEY == "illegal_contact_impulse_n_s"
    assert config.task.admission.illegal_contact_impulse_n_s == 0.05


def test_default_threshold_changes_fail_closed_until_contract_is_updated(config: ResolvedRunConfig) -> None:
    altered = replace(config.task.admission, illegal_contact_impulse_n_s=0.051)
    with pytest.raises(ValueError, match="FR-08"):
        evaluate_admission(_admission_input(), altered)
    altered_ready = replace(config.task.ready, position_error_m=0.006)
    with pytest.raises(ValueError, match="FR-11"):
        step_ready(_ready_input(), ReadyEpisodeState(), altered_ready)


def test_input_and_state_runtime_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="reach_phase"):
        _admission_input(reach_phase=1)
    with pytest.raises(TypeError, match="position_error_m"):
        _admission_input(position_error_m="0.01")
    with pytest.raises(ValueError, match="non-negative"):
        ReadyEpisodeState(consecutive_steps=-1)
    with pytest.raises(ValueError, match="ever_ready"):
        ReadyEpisodeState(consecutive_steps=15, ever_ready=False)
