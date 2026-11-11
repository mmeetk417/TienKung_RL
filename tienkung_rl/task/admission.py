"""Pure Option-admission and strict-Ready contracts.

This module deliberately accepts only already-reduced task scalars.  It has no
simulator dependency and cannot start an Option, terminate an episode, grant a
reward, or update curriculum state.  AdmissionInput carries measurements shared
with strict Ready and telemetry; only the active admission subset is gated.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tienkung_rl.config.models import AdmissionConfig, ReadyConfig

TASK_CONTRACT_VERSION = "reach_then_close_runtime"

POLICY_STEP_S = 0.020
ADMISSION_DWELL_STEPS = 15
READY_DWELL_STEPS = 15

# Admission thresholds are part of TaskConfig and its component fingerprint.
# The contact value is intentionally not described as validated: FR-08 makes it
# a first MuJoCo calibration starting point.
ADMISSION_THRESHOLD_VERSION = TASK_CONTRACT_VERSION
ADMISSION_THRESHOLD_CALIBRATION_STATUS = "starting_point_unvalidated"
ILLEGAL_CONTACT_IMPULSE_CALIBRATION_KEY = "illegal_contact_impulse_n_s"

ADMISSION_CONDITION_KEYS = (
    "finite_state",
    "valid_target",
    "reach_phase",
    "position_error",
    "orientation_error",
    "grasp_center_speed",
    "object_tilt",
    "illegal_contact_impulse",
)

READY_CONDITION_KEYS = (
    "finite_state",
    "valid_target",
    "position_error",
    "orientation_error",
    "grasp_center_speed",
    "arm_joint_speed",
    "object_displacement",
    "object_tilt",
    "object_linear_speed",
    "object_angular_speed",
    "illegal_contact_impulse",
)

_NUMERIC_INPUT_NAMES = (
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


def _require_bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _require_scalar(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionInput:
    """Reduced task measurements shared by Option admission and telemetry."""

    state_finite: bool
    target_valid: bool
    reach_phase: bool
    position_error_m: float
    orientation_error_rad: float
    grasp_center_speed_m_s: float
    arm_speed_max_rad_s: float
    object_displacement_m: float
    object_tilt_rad: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    illegal_contact_impulse_n_s: float

    def __post_init__(self) -> None:
        for name in ("state_finite", "target_valid", "reach_phase"):
            _require_bool(getattr(self, name), name)
        for name in _NUMERIC_INPUT_NAMES:
            _require_scalar(getattr(self, name), name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadyInput:
    """Strict-Ready inputs; phase is intentionally absent from this diagnostic."""

    state_finite: bool
    target_valid: bool
    position_error_m: float
    orientation_error_rad: float
    grasp_center_speed_m_s: float
    arm_speed_max_rad_s: float
    object_displacement_m: float
    object_tilt_rad: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    illegal_contact_impulse_n_s: float

    def __post_init__(self) -> None:
        for name in ("state_finite", "target_valid"):
            _require_bool(getattr(self, name), name)
        for name in _NUMERIC_INPUT_NAMES:
            _require_scalar(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class AdmissionMetrics:
    """Named error and safety measurements retained for logging."""

    position_error_m: float
    orientation_error_rad: float
    grasp_center_speed_m_s: float
    arm_speed_max_rad_s: float
    object_displacement_m: float
    object_tilt_rad: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    illegal_contact_impulse_n_s: float


@dataclass(frozen=True, slots=True)
class ConditionResult:
    """One stable, independently loggable condition result."""

    key: str
    passed: bool


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Admission conjunction, optionally advanced through the automatic dwell."""

    conditions: tuple[ConditionResult, ...]
    admissible: bool
    metrics: AdmissionMetrics
    failure_reasons: tuple[str, ...]
    instantaneous: bool = False
    newly_admissible: bool = False
    consecutive_steps: int = 0
    dwell_steps_required: int = ADMISSION_DWELL_STEPS
    threshold_version: str = ADMISSION_THRESHOLD_VERSION
    calibration_status: str = ADMISSION_THRESHOLD_CALIBRATION_STATUS

    def __post_init__(self) -> None:
        if self.consecutive_steps < 0:
            raise ValueError("admission consecutive_steps must be non-negative")
        if self.dwell_steps_required != ADMISSION_DWELL_STEPS:
            raise ValueError("admission dwell must require exactly 15 policy steps")
        if self.admissible and not self.instantaneous:
            raise ValueError("dwell-complete admission must also be instantaneous")
        if self.newly_admissible and not self.admissible:
            raise ValueError("newly_admissible requires dwell-complete admission")

    @property
    def option_admissible(self) -> bool:
        """True only after the automatic admission dwell is complete."""

        return self.admissible

    @property
    def instantaneous_admissible(self) -> bool:
        return self.instantaneous

    @property
    def condition_flags(self) -> Mapping[str, bool]:
        return MappingProxyType({condition.key: condition.passed for condition in self.conditions})

    def condition_passed(self, key: str) -> bool:
        for condition in self.conditions:
            if condition.key == key:
                return condition.passed
        raise KeyError(key)


@dataclass(frozen=True, slots=True)
class ReadyEpisodeState:
    """The only episode-local state strict Ready is allowed to update."""

    consecutive_steps: int = 0
    ever_ready: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.consecutive_steps, bool) or not isinstance(self.consecutive_steps, int):
            raise TypeError("consecutive_steps must be an int")
        if self.consecutive_steps < 0:
            raise ValueError("consecutive_steps must be non-negative")
        _require_bool(self.ever_ready, "ever_ready")
        if self.consecutive_steps >= READY_DWELL_STEPS and not self.ever_ready:
            raise ValueError("a dwell-complete state must have ever_ready latched")


@dataclass(frozen=True, slots=True)
class AdmissionEpisodeState:
    """Episode-local consecutive admission count for deterministic triggering."""

    consecutive_steps: int = 0
    ever_admissible: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.consecutive_steps, bool) or not isinstance(self.consecutive_steps, int):
            raise TypeError("admission consecutive_steps must be an int")
        if self.consecutive_steps < 0:
            raise ValueError("admission consecutive_steps must be non-negative")
        _require_bool(self.ever_admissible, "ever_admissible")
        if self.consecutive_steps >= ADMISSION_DWELL_STEPS and not self.ever_admissible:
            raise ValueError("dwell-complete admission state must latch ever_admissible")


@dataclass(frozen=True, slots=True)
class ReadyResult:
    """Current strict-Ready diagnostic; it carries no control/reward outputs."""

    conditions: tuple[ConditionResult, ...]
    instantaneous: bool
    ready: bool
    newly_ready: bool
    consecutive_steps: int
    dwell_steps_required: int
    ever_ready: bool
    metrics: AdmissionMetrics
    failure_reasons: tuple[str, ...]
    threshold_version: str = ADMISSION_THRESHOLD_VERSION
    calibration_status: str = ADMISSION_THRESHOLD_CALIBRATION_STATUS

    @property
    def condition_flags(self) -> Mapping[str, bool]:
        return MappingProxyType({condition.key: condition.passed for condition in self.conditions})

    def condition_passed(self, key: str) -> bool:
        for condition in self.conditions:
            if condition.key == key:
                return condition.passed
        raise KeyError(key)


def _metrics(inputs: AdmissionInput | ReadyInput) -> AdmissionMetrics:
    return AdmissionMetrics(*(float(getattr(inputs, name)) for name in _NUMERIC_INPUT_NAMES))


def _bounded(value: float, maximum: float) -> bool:
    # The inputs are magnitudes.  Negative values are malformed measurements,
    # and Python's ordinary comparisons already fail closed for NaN.
    return math.isfinite(value) and 0.0 <= value <= maximum


def _admission_condition_results(
    inputs: AdmissionInput,
    *,
    position_error_m: float,
    orientation_error_rad: float,
    grasp_center_speed_m_s: float,
    object_tilt_rad: float,
    contact_impulse_n_s: float,
) -> tuple[ConditionResult, ...]:
    # Every supplied measurement must remain finite; only the active subset
    # below has an admission threshold.
    numeric_finite = all(math.isfinite(float(getattr(inputs, name))) for name in _NUMERIC_INPUT_NAMES)
    return (
        ConditionResult("finite_state", inputs.state_finite and numeric_finite),
        ConditionResult("valid_target", inputs.target_valid),
        ConditionResult("reach_phase", inputs.reach_phase),
        ConditionResult("position_error", _bounded(inputs.position_error_m, position_error_m)),
        ConditionResult("orientation_error", _bounded(inputs.orientation_error_rad, orientation_error_rad)),
        ConditionResult("grasp_center_speed", _bounded(inputs.grasp_center_speed_m_s, grasp_center_speed_m_s)),
        ConditionResult("object_tilt", _bounded(inputs.object_tilt_rad, object_tilt_rad)),
        ConditionResult("illegal_contact_impulse", _bounded(inputs.illegal_contact_impulse_n_s, contact_impulse_n_s)),
    )


def _ready_condition_results(
    inputs: ReadyInput,
    *,
    position_error_m: float,
    orientation_error_rad: float,
    grasp_center_speed_m_s: float,
    arm_speed_rad_s: float,
    object_displacement_m: float,
    object_tilt_rad: float,
    object_linear_speed_m_s: float,
    object_angular_speed_rad_s: float,
    contact_impulse_n_s: float,
) -> tuple[ConditionResult, ...]:
    numeric_finite = all(math.isfinite(float(getattr(inputs, name))) for name in _NUMERIC_INPUT_NAMES)
    return (
        ConditionResult("finite_state", inputs.state_finite and numeric_finite),
        ConditionResult("valid_target", inputs.target_valid),
        ConditionResult("position_error", _bounded(inputs.position_error_m, position_error_m)),
        ConditionResult("orientation_error", _bounded(inputs.orientation_error_rad, orientation_error_rad)),
        ConditionResult("grasp_center_speed", _bounded(inputs.grasp_center_speed_m_s, grasp_center_speed_m_s)),
        ConditionResult("arm_joint_speed", _bounded(inputs.arm_speed_max_rad_s, arm_speed_rad_s)),
        ConditionResult("object_displacement", _bounded(inputs.object_displacement_m, object_displacement_m)),
        ConditionResult("object_tilt", _bounded(inputs.object_tilt_rad, object_tilt_rad)),
        ConditionResult("object_linear_speed", _bounded(inputs.object_linear_speed_m_s, object_linear_speed_m_s)),
        ConditionResult(
            "object_angular_speed", _bounded(inputs.object_angular_speed_rad_s, object_angular_speed_rad_s)
        ),
        ConditionResult("illegal_contact_impulse", _bounded(inputs.illegal_contact_impulse_n_s, contact_impulse_n_s)),
    )


def evaluate_admission(inputs: AdmissionInput, config: AdmissionConfig) -> AdmissionResult:
    """Evaluate Option-admission conditions without dwell or early return."""

    config.validate()
    conditions = _admission_condition_results(
        inputs,
        position_error_m=config.position_error_m,
        orientation_error_rad=config.orientation_error_rad,
        grasp_center_speed_m_s=config.grasp_center_speed_m_s,
        object_tilt_rad=config.object_tilt_rad,
        contact_impulse_n_s=config.illegal_contact_impulse_n_s,
    )
    failure_reasons = tuple(condition.key for condition in conditions if not condition.passed)
    return AdmissionResult(
        conditions=conditions,
        admissible=not failure_reasons,
        metrics=_metrics(inputs),
        failure_reasons=failure_reasons,
        instantaneous=not failure_reasons,
    )


def step_admission(
    inputs: AdmissionInput,
    state: AdmissionEpisodeState,
    config: AdmissionConfig,
    *,
    policy_step_s: float = POLICY_STEP_S,
) -> tuple[AdmissionResult, AdmissionEpisodeState]:
    """Require 0.30 s of continuous admission before automatic triggering."""

    config.validate()
    if not math.isfinite(policy_step_s) or not math.isclose(policy_step_s, POLICY_STEP_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("admission policy_step_s must remain exactly 0.020 s")
    dwell_steps = round(config.dwell_time_s / policy_step_s)
    if dwell_steps != ADMISSION_DWELL_STEPS or not math.isclose(
        dwell_steps * policy_step_s, config.dwell_time_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("admission dwell must remain exactly 0.30 s / 15 policy steps")

    instantaneous = evaluate_admission(inputs, config)
    consecutive_steps = state.consecutive_steps + 1 if instantaneous.admissible else 0
    admissible = instantaneous.admissible and consecutive_steps >= dwell_steps
    newly_admissible = admissible and not state.ever_admissible
    ever_admissible = state.ever_admissible or admissible
    next_state = AdmissionEpisodeState(
        consecutive_steps=consecutive_steps,
        ever_admissible=ever_admissible,
    )
    return (
        AdmissionResult(
            conditions=instantaneous.conditions,
            admissible=admissible,
            metrics=instantaneous.metrics,
            failure_reasons=instantaneous.failure_reasons,
            instantaneous=instantaneous.admissible,
            newly_admissible=newly_admissible,
            consecutive_steps=consecutive_steps,
            dwell_steps_required=dwell_steps,
        ),
        next_state,
    )


def step_ready(
    inputs: ReadyInput,
    state: ReadyEpisodeState,
    config: ReadyConfig,
    *,
    policy_step_s: float = POLICY_STEP_S,
) -> tuple[ReadyResult, ReadyEpisodeState]:
    """Advance strict-Ready dwell and return a new immutable episode state."""

    config.validate()
    if not math.isfinite(policy_step_s) or not math.isclose(policy_step_s, POLICY_STEP_S, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("strict Ready policy_step_s must remain exactly 0.020 s")
    dwell_steps = round(config.dwell_time_s / policy_step_s)
    if dwell_steps != READY_DWELL_STEPS or not math.isclose(
        dwell_steps * policy_step_s, config.dwell_time_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("strict Ready dwell must remain exactly 0.30 s / 15 policy steps")

    conditions = _ready_condition_results(
        inputs,
        position_error_m=config.position_error_m,
        orientation_error_rad=config.orientation_error_rad,
        grasp_center_speed_m_s=config.grasp_center_speed_m_s,
        object_tilt_rad=config.object_tilt_rad,
        contact_impulse_n_s=config.contact_impulse_n_s,
        arm_speed_rad_s=config.arm_speed_rad_s,
        object_displacement_m=config.object_displacement_m,
        object_linear_speed_m_s=config.object_linear_speed_m_s,
        object_angular_speed_rad_s=config.object_angular_speed_rad_s,
    )
    failure_reasons = tuple(condition.key for condition in conditions if not condition.passed)
    instantaneous = not failure_reasons
    consecutive_steps = state.consecutive_steps + 1 if instantaneous else 0
    ready = instantaneous and consecutive_steps >= dwell_steps
    newly_ready = ready and not state.ever_ready
    ever_ready = state.ever_ready or ready
    next_state = ReadyEpisodeState(consecutive_steps=consecutive_steps, ever_ready=ever_ready)
    return (
        ReadyResult(
            conditions=conditions,
            instantaneous=instantaneous,
            ready=ready,
            newly_ready=newly_ready,
            consecutive_steps=consecutive_steps,
            dwell_steps_required=dwell_steps,
            ever_ready=ever_ready,
            metrics=_metrics(inputs),
            failure_reasons=failure_reasons,
        ),
        next_state,
    )


__all__ = [
    "ADMISSION_CONDITION_KEYS",
    "ADMISSION_DWELL_STEPS",
    "ADMISSION_THRESHOLD_CALIBRATION_STATUS",
    "ADMISSION_THRESHOLD_VERSION",
    "ILLEGAL_CONTACT_IMPULSE_CALIBRATION_KEY",
    "POLICY_STEP_S",
    "READY_CONDITION_KEYS",
    "READY_DWELL_STEPS",
    "AdmissionEpisodeState",
    "AdmissionInput",
    "AdmissionMetrics",
    "AdmissionResult",
    "ConditionResult",
    "ReadyEpisodeState",
    "ReadyInput",
    "ReadyResult",
    "evaluate_admission",
    "step_admission",
    "step_ready",
]
