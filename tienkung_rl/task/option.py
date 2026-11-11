"""Deterministic Reach-to-grasp Option state machine.

The module is intentionally simulator independent.  A task orchestrator owns
the policy/physics ordering and supplies one observation for every physics
substep.  MuJoCo remains the sole source of finger-mimic motion: this module
outputs only the six active-hand actuator targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray

from tienkung_rl.config.models import OptionConfig

from .contracts import (
    ACTIVE_HAND_DIMENSION,
    ACTIVE_HAND_JOINT_NAMES,
    ARM_POLICY_DIMENSION,
    MAXIMUM_ARM_TARGET_STEP_RAD,
    PHYSICAL_ACTION_DIMENSION,
    PhysicalAction,
)

FloatArray = NDArray[np.float64]

POLICY_STEP_S = 0.020
SETTLE_STEPS = 25
CLOSE_STEPS = 175
HOLD_STEPS = 50
LIFT_STEPS = 100
LIFT_HOLD_STEPS = 250
TOTAL_SCRIPT_STEPS = 600
CONTACT_FORCE_THRESHOLD_N = 0.20
BILATERAL_PERSISTENCE_STEPS = 3
SUPPORTED_LIFT_PERSISTENCE_STEPS = 5
LIFT_EVENT_HEIGHT_M = 0.05
LIFT_TRAJECTORY_VERSION = "elbow_biased_cartesian_lift_v5"
LIFT_DLS_DAMPING_SQUARED = 1.0e-6
LIFT_MAXIMUM_JOINT_CORRECTION_RAD = 0.03
# This limit lets the 100-step smoothstep reach the larger elbow bias.
LIFT_MAXIMUM_COMMAND_STEP_RAD = 0.006
# The right elbow's negative pitch direction raises the TCP in this asset.
# These joint-space offsets deliberately allow a small path/orientation
# deviation so the grasp gets additional mechanical lift instead of relying
# on a strict +Z Cartesian target alone.
LIFT_ELBOW_BIAS_RAD = -0.40
LIFT_WRIST_PITCH_COMPENSATION_RAD = 0.10
# Compensate for position-actuator compliance while keeping the task contract
# and the 5 cm success event unchanged.


class OptionPhase(StrEnum):
    """Stable phase names used by observations, reports, and checkpoints."""

    Reach = "Reach"
    ScriptPending = "ScriptPending"
    Settle = "Settle"
    Close = "Close"
    Hold = "Hold"
    Lift = "Lift"
    LiftHold = "LiftHold"
    Success = "Success"
    Failure = "Failure"


_ACTIVE_SCRIPT_PHASES = frozenset(
    {
        OptionPhase.ScriptPending,
        OptionPhase.Settle,
        OptionPhase.Close,
        OptionPhase.Hold,
        OptionPhase.Lift,
        OptionPhase.LiftHold,
    }
)


class OptionEvent(StrEnum):
    """Stable, one-transition event keys."""

    TriggerRequested = "trigger_requested"
    TriggerAccepted = "trigger_accepted"
    TriggerRejected = "trigger_rejected"
    Lift5cm = "lift_5cm"
    Toppled = "toppled"
    Dropped = "dropped"
    IllegalDisplacement = "illegal_displacement"
    IllegalContact = "illegal_contact"
    NonFinite = "nonfinite"
    ScriptFailure = "script_failure"
    CompleteSuccess = "complete_success"


_HARD_SAFETY_EVENT_ORDER = (
    OptionEvent.NonFinite,
    OptionEvent.Toppled,
    OptionEvent.Dropped,
    OptionEvent.IllegalContact,
)


@dataclass(frozen=True, slots=True)
class OptionTiming:
    """The exact policy-step realization of the versioned FR-10 timing."""

    settle_steps: int = SETTLE_STEPS
    close_steps: int = CLOSE_STEPS
    hold_steps: int = HOLD_STEPS
    lift_steps: int = LIFT_STEPS
    lift_hold_steps: int = LIFT_HOLD_STEPS

    def __post_init__(self) -> None:
        if self.phase_steps != (25, 175, 50, 100, 250):
            raise ValueError("Option timing must remain exactly 25/175/50/100/250 policy steps")

    @classmethod
    def from_config(cls, config: OptionConfig, policy_step_s: float = POLICY_STEP_S) -> OptionTiming:
        config.validate()
        if not math.isclose(policy_step_s, POLICY_STEP_S, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("Option policy step must remain exactly 0.020 s")
        durations = (
            config.settle_time_s,
            config.closing_time_s,
            config.hold_time_s,
            config.lift_time_s,
            config.lift_hold_time_s,
        )
        exact_steps = tuple(duration / policy_step_s for duration in durations)
        rounded = tuple(round(value) for value in exact_steps)
        if any(
            not math.isclose(value, step, rel_tol=0.0, abs_tol=1.0e-12)
            for value, step in zip(exact_steps, rounded, strict=True)
        ):
            raise ValueError("every Option duration must be an exact integer number of policy steps")
        result = cls(*rounded)
        if result.phase_steps != (25, 175, 50, 100, 250) or result.total_steps != TOTAL_SCRIPT_STEPS:
            raise ValueError("Option timing must be 25/175/50/100/250 = 600 policy steps")
        return result

    @property
    def phase_steps(self) -> tuple[int, int, int, int, int]:
        return (
            self.settle_steps,
            self.close_steps,
            self.hold_steps,
            self.lift_steps,
            self.lift_hold_steps,
        )

    @property
    def total_steps(self) -> int:
        return sum(self.phase_steps)

    def steps_for(self, phase: OptionPhase) -> int:
        mapping = {
            OptionPhase.Settle: self.settle_steps,
            OptionPhase.Close: self.close_steps,
            OptionPhase.Hold: self.hold_steps,
            OptionPhase.Lift: self.lift_steps,
            OptionPhase.LiftHold: self.lift_hold_steps,
        }
        try:
            return mapping[phase]
        except KeyError as error:
            raise ValueError(f"phase {phase.value} has no scripted duration") from error


@dataclass(frozen=True, slots=True)
class LiftPreparation:
    """Measured right-arm and TCP kinematics for one closed-loop Lift step."""

    measured_arm_positions_rad: tuple[float, ...]
    tcp_world_position_m: tuple[float, float, float]
    tcp_world_quaternion_wxyz: tuple[float, float, float, float]
    tcp_jacobian_6x7: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if len(self.measured_arm_positions_rad) != ARM_POLICY_DIMENSION:
            raise ValueError("measured arm positions must contain seven values")
        if len(self.tcp_world_position_m) != 3 or len(self.tcp_world_quaternion_wxyz) != 4:
            raise ValueError("TCP position/quaternion dimensions are invalid")
        if len(self.tcp_jacobian_6x7) != 6 or any(
            len(row) != ARM_POLICY_DIMENSION for row in self.tcp_jacobian_6x7
        ):
            raise ValueError("TCP Jacobian must have shape 6 x 7")
        values = (
            *self.measured_arm_positions_rad,
            *self.tcp_world_position_m,
            *self.tcp_world_quaternion_wxyz,
            *(value for row in self.tcp_jacobian_6x7 for value in row),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("LiftPreparation values must be finite")
        quaternion_norm = float(np.linalg.norm(self.tcp_world_quaternion_wxyz))
        if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("TCP quaternion must be normalized")


@dataclass(frozen=True, slots=True)
class OptionSubstepObservation:
    """Already-reduced contact and hard-safety values for one physics substep."""

    thumb_side_force_n: float = 0.0
    opposing_side_force_n: float = 0.0
    net_hand_force_n: float = 0.0
    object_lift_height_m: float = 0.0
    toppled: bool = False
    dropped: bool = False
    illegal_displacement: bool = False
    illegal_contact: bool = False
    state_finite: bool = True

    def __post_init__(self) -> None:
        for name in ("toppled", "dropped", "illegal_displacement", "illegal_contact", "state_finite"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "thumb_side_force_n",
            "opposing_side_force_n",
            "net_hand_force_n",
            "object_lift_height_m",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real scalar")
            if math.isfinite(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class OptionEpisodeState:
    """Immutable episode state; transition flags refer only to the latest step."""

    phase: OptionPhase = OptionPhase.Reach
    phase_step: int = 0
    script_steps_completed: int = 0
    trigger_requested: bool = False
    trigger_accepted: bool = False
    trigger_rejected: bool = False
    request_count: int = 0
    rejection_count: int = 0
    acceptance_count: int = 0
    acceptance_latched: bool = False
    latched_arm_targets_rad: tuple[float, ...] | None = None
    lift_start_arm_targets_rad: tuple[float, ...] | None = None
    lift_final_arm_targets_rad: tuple[float, ...] | None = None
    lift_start_tcp_position_m: tuple[float, float, float] | None = None
    lift_start_tcp_quaternion_wxyz: tuple[float, float, float, float] | None = None
    bilateral_contact_steps: int = 0
    supported_lift_steps: int = 0
    maximum_supported_lift_steps: int = 0
    bilateral_contact_ever: bool = False
    bilateral_persistent_ever: bool = False
    supported_lift_ever: bool = False
    lift_5cm_event: bool = False
    hard_safety_events: tuple[OptionEvent, ...] = ()
    transition_events: tuple[OptionEvent, ...] = ()
    trajectory_version: str = LIFT_TRAJECTORY_VERSION

    def __post_init__(self) -> None:
        if self.phase_step < 0 or self.script_steps_completed < 0:
            raise ValueError("Option step counters must be non-negative")
        if self.script_steps_completed > TOTAL_SCRIPT_STEPS:
            raise ValueError("script_steps_completed cannot exceed 600")
        if min(self.request_count, self.rejection_count, self.acceptance_count) < 0:
            raise ValueError("Option event counters must be non-negative")
        if self.acceptance_count > 1:
            raise ValueError("an episode can accept the Option at most once")
        if self.acceptance_latched != (self.acceptance_count == 1):
            raise ValueError("acceptance latch and count disagree")
        if self.acceptance_latched and self.latched_arm_targets_rad is None:
            raise ValueError("an accepted Option requires a latched arm target")
        for targets in (
            self.latched_arm_targets_rad,
            self.lift_start_arm_targets_rad,
            self.lift_final_arm_targets_rad,
        ):
            if targets is not None and (
                len(targets) != ARM_POLICY_DIMENSION or not all(math.isfinite(x) for x in targets)
            ):
                raise ValueError("stored arm targets must contain seven finite values")
        if self.lift_start_tcp_position_m is not None and (
            len(self.lift_start_tcp_position_m) != 3
            or not all(math.isfinite(x) for x in self.lift_start_tcp_position_m)
        ):
            raise ValueError("stored Lift TCP position must contain three finite values")
        if self.lift_start_tcp_quaternion_wxyz is not None and (
            len(self.lift_start_tcp_quaternion_wxyz) != 4
            or not all(math.isfinite(x) for x in self.lift_start_tcp_quaternion_wxyz)
        ):
            raise ValueError("stored Lift TCP quaternion must contain four finite values")
        if min(self.bilateral_contact_steps, self.supported_lift_steps, self.maximum_supported_lift_steps) < 0:
            raise ValueError("contact persistence counters must be non-negative")
        if self.supported_lift_steps > self.maximum_supported_lift_steps:
            raise ValueError("current support count cannot exceed its episode maximum")

    @property
    def script_active(self) -> bool:
        return self.phase in _ACTIVE_SCRIPT_PHASES

    @property
    def script_progress(self) -> float:
        return min(1.0, self.script_steps_completed / TOTAL_SCRIPT_STEPS)

    @property
    def support_progress(self) -> float:
        return min(1.0, self.supported_lift_steps / SUPPORTED_LIFT_PERSISTENCE_STEPS)


def smoothstep(fraction: float) -> float:
    """Return the inherited bounded C1 interpolation fraction."""

    if not math.isfinite(fraction):
        raise ValueError("smoothstep fraction must be finite")
    clamped = min(1.0, max(0.0, fraction))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _quaternion_rotation_error(
    target_wxyz: tuple[float, float, float, float],
    current_wxyz: tuple[float, float, float, float],
) -> FloatArray:
    """Return the shortest world-frame rotation vector from current to target."""

    target = np.asarray(target_wxyz, dtype=np.float64)
    current = np.asarray(current_wxyz, dtype=np.float64)
    cw, cx, cy, cz = current
    tw, tx, ty, tz = target
    error = np.asarray(
        (
            tw * cw + tx * cx + ty * cy + tz * cz,
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ),
        dtype=np.float64,
    )
    if error[0] < 0.0:
        error *= -1.0
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(error[0])))
    return np.asarray(angle * error[1:] / vector_norm, dtype=np.float64)


def solve_cartesian_lift_target(
    preparation: LiftPreparation,
    *,
    start_tcp_position_m: tuple[float, float, float],
    start_tcp_quaternion_wxyz: tuple[float, float, float, float],
    waypoint_height_m: float,
    previous_arm_targets_rad: FloatArray,
) -> FloatArray:
    """Compute one bounded DLS target that raises TCP in world Z without pose drift."""

    if not math.isfinite(waypoint_height_m) or waypoint_height_m < 0.0:
        raise ValueError("Lift waypoint height must be finite and non-negative")
    previous = _strict_vector(previous_arm_targets_rad, ARM_POLICY_DIMENSION, "previous_arm_targets_rad")
    measured = np.asarray(preparation.measured_arm_positions_rad, dtype=np.float64)
    desired_position = np.asarray(start_tcp_position_m, dtype=np.float64).copy()
    desired_position[2] += waypoint_height_m
    position_error = desired_position - np.asarray(preparation.tcp_world_position_m, dtype=np.float64)
    rotation_error = _quaternion_rotation_error(
        start_tcp_quaternion_wxyz,
        preparation.tcp_world_quaternion_wxyz,
    )
    cartesian_error = np.concatenate((position_error, rotation_error))
    jacobian = np.asarray(preparation.tcp_jacobian_6x7, dtype=np.float64)
    normal = jacobian @ jacobian.T + LIFT_DLS_DAMPING_SQUARED * np.eye(6, dtype=np.float64)
    joint_delta = jacobian.T @ np.linalg.solve(normal, cartesian_error)
    maximum_requested_step = float(np.max(np.abs(joint_delta)))
    if maximum_requested_step > LIFT_MAXIMUM_JOINT_CORRECTION_RAD:
        joint_delta *= LIFT_MAXIMUM_JOINT_CORRECTION_RAD / maximum_requested_step
    desired = measured + joint_delta
    command_delta = desired - previous
    maximum_command_step = float(np.max(np.abs(command_delta)))
    if maximum_command_step > LIFT_MAXIMUM_COMMAND_STEP_RAD:
        command_delta *= LIFT_MAXIMUM_COMMAND_STEP_RAD / maximum_command_step
    return np.ascontiguousarray(previous + command_delta, dtype=np.float64)


def process_reach_transition(
    state: OptionEpisodeState,
    *,
    trigger_requested: bool,
    option_admissible: bool,
    executed_arm_targets_rad: FloatArray,
) -> OptionEpisodeState:
    """Pure Reach transition for no request, rejection, or one-time acceptance."""

    if state.phase is not OptionPhase.Reach:
        raise ValueError("Reach trigger processing is only valid in Reach")
    if not isinstance(trigger_requested, bool) or not isinstance(option_admissible, bool):
        raise TypeError("trigger_requested and option_admissible must be bool")
    arm = _strict_vector(executed_arm_targets_rad, ARM_POLICY_DIMENSION, "executed_arm_targets_rad")
    events: tuple[OptionEvent, ...] = ()
    if trigger_requested:
        events = (OptionEvent.TriggerRequested,)
        if option_admissible and not state.acceptance_latched:
            events += (OptionEvent.TriggerAccepted,)
            return replace(
                state,
                phase=OptionPhase.ScriptPending,
                phase_step=0,
                trigger_requested=True,
                trigger_accepted=True,
                trigger_rejected=False,
                request_count=state.request_count + 1,
                acceptance_count=1,
                acceptance_latched=True,
                latched_arm_targets_rad=tuple(float(value) for value in arm),
                transition_events=events,
            )
        events += (OptionEvent.TriggerRejected,)
        return replace(
            state,
            trigger_requested=True,
            trigger_accepted=False,
            trigger_rejected=True,
            request_count=state.request_count + 1,
            rejection_count=state.rejection_count + 1,
            transition_events=events,
        )
    return replace(
        state,
        trigger_requested=False,
        trigger_accepted=False,
        trigger_rejected=False,
        transition_events=(),
    )


def _strict_vector(value: object, size: int, name: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _limited_targets(desired: FloatArray, previous: FloatArray, lower: FloatArray, upper: FloatArray) -> FloatArray:
    clipped = np.clip(desired, lower, upper)
    result = clipped.copy()
    arm_delta = np.clip(
        clipped[:ARM_POLICY_DIMENSION] - previous[:ARM_POLICY_DIMENSION],
        -MAXIMUM_ARM_TARGET_STEP_RAD,
        MAXIMUM_ARM_TARGET_STEP_RAD,
    )
    result[:ARM_POLICY_DIMENSION] = previous[:ARM_POLICY_DIMENSION] + arm_delta
    return np.ascontiguousarray(result, dtype=np.float64)


def _next_phase(phase: OptionPhase) -> OptionPhase:
    return {
        OptionPhase.Settle: OptionPhase.Close,
        OptionPhase.Close: OptionPhase.Hold,
        OptionPhase.Hold: OptionPhase.Lift,
        OptionPhase.Lift: OptionPhase.LiftHold,
    }[phase]


@dataclass(slots=True)
class OptionController:
    """One-environment controller enforcing FR-09/10 ordering and safety."""

    config: OptionConfig
    pregrasp_hand_targets_rad: FloatArray
    actuator_lower_limits_rad: FloatArray
    actuator_upper_limits_rad: FloatArray
    timing: OptionTiming | None = None
    state: OptionEpisodeState = OptionEpisodeState()
    _previous_output_rad: FloatArray | None = None
    _step_bilateral: bool = False
    _step_supported: bool = False
    _policy_step_open: bool = False

    def __post_init__(self) -> None:
        self.config.validate()
        if self.config.active_hand_joint_names != ACTIVE_HAND_JOINT_NAMES:
            raise ValueError("Option active-hand order differs from the 13-D action contract")
        self.pregrasp_hand_targets_rad = np.array(
            _strict_vector(self.pregrasp_hand_targets_rad, ACTIVE_HAND_DIMENSION, "pregrasp_hand_targets_rad"),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        self.actuator_lower_limits_rad = np.array(
            _strict_vector(self.actuator_lower_limits_rad, PHYSICAL_ACTION_DIMENSION, "actuator_lower_limits_rad"),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        self.actuator_upper_limits_rad = np.array(
            _strict_vector(self.actuator_upper_limits_rad, PHYSICAL_ACTION_DIMENSION, "actuator_upper_limits_rad"),
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if np.any(self.actuator_lower_limits_rad > self.actuator_upper_limits_rad):
            raise ValueError("actuator lower limits must not exceed upper limits")
        self.timing = OptionTiming.from_config(self.config) if self.timing is None else self.timing
        if self.timing.total_steps != TOTAL_SCRIPT_STEPS:
            raise ValueError("Option timing must total 600 policy steps")

    def reset(self, measured_physical_targets_rad: FloatArray) -> OptionEpisodeState:
        """Clear all episode state and restore Reach/pregrasp output."""

        measured = _strict_vector(
            measured_physical_targets_rad, PHYSICAL_ACTION_DIMENSION, "measured_physical_targets_rad"
        )
        restored = np.array(measured, dtype=np.float64, order="C", copy=True)
        restored[ARM_POLICY_DIMENSION:] = self.pregrasp_hand_targets_rad
        if np.any(restored < self.actuator_lower_limits_rad) or np.any(restored > self.actuator_upper_limits_rad):
            raise ValueError("reset targets lie outside actuator limits")
        self._previous_output_rad = restored
        self._step_bilateral = False
        self._step_supported = False
        self._policy_step_open = False
        self.state = OptionEpisodeState()
        return self.state

    def begin_policy_step(
        self,
        policy_arm_targets_rad: FloatArray,
        *,
        lift_preparation: LiftPreparation | None = None,
    ) -> PhysicalAction:
        """Produce one 13-D target before physics; no trigger can enter it."""

        if self._previous_output_rad is None:
            raise RuntimeError("OptionController must be reset before use")
        if self._policy_step_open:
            raise RuntimeError("finish_policy_step must close the previous policy step")
        policy_arm = _strict_vector(policy_arm_targets_rad, ARM_POLICY_DIMENSION, "policy_arm_targets_rad")
        state = replace(
            self.state,
            trigger_requested=False,
            trigger_accepted=False,
            trigger_rejected=False,
            transition_events=(),
        )
        if state.phase is OptionPhase.ScriptPending:
            state = replace(state, phase=OptionPhase.Settle, phase_step=0)
        self.state = state
        desired = self._desired_targets(state, policy_arm, lift_preparation)
        state = self.state
        output = _limited_targets(
            desired,
            self._previous_output_rad,
            self.actuator_lower_limits_rad,
            self.actuator_upper_limits_rad,
        )
        self._previous_output_rad = output
        self._step_bilateral = False
        self._step_supported = False
        self._policy_step_open = True
        self.state = state
        return PhysicalAction(torch.from_numpy(output.copy()))

    def _desired_targets(
        self,
        state: OptionEpisodeState,
        policy_arm: FloatArray,
        lift_preparation: LiftPreparation | None,
    ) -> FloatArray:
        if state.phase is OptionPhase.Reach:
            return np.ascontiguousarray(np.concatenate((policy_arm, self.pregrasp_hand_targets_rad)), dtype=np.float64)
        if state.phase in {OptionPhase.Success, OptionPhase.Failure}:
            return np.array(self._previous_output_rad, dtype=np.float64, order="C", copy=True)
        if state.latched_arm_targets_rad is None:
            raise RuntimeError("script phase has no latched arm target")
        arm = np.asarray(state.latched_arm_targets_rad, dtype=np.float64).copy()
        hand = self.pregrasp_hand_targets_rad.copy()
        if state.phase in {OptionPhase.Close, OptionPhase.Hold, OptionPhase.Lift, OptionPhase.LiftHold}:
            closed = self.actuator_upper_limits_rad[ARM_POLICY_DIMENSION:].copy()
            closed[0] = self.pregrasp_hand_targets_rad[0]
            if state.phase is OptionPhase.Close:
                if self.timing is None:  # pragma: no cover - established in __post_init__
                    raise RuntimeError("Option timing was not initialized")
                fraction = smoothstep((state.phase_step + 1) / self.timing.close_steps)
                hand = self.pregrasp_hand_targets_rad + fraction * (closed - self.pregrasp_hand_targets_rad)
            else:
                hand = closed
        if state.phase is OptionPhase.Lift:
            state, arm = self._lift_targets(state, lift_preparation)
            self.state = state
        elif state.phase is OptionPhase.LiftHold:
            if state.lift_final_arm_targets_rad is None:
                raise RuntimeError("LiftHold has no final lift target")
            arm = np.asarray(state.lift_final_arm_targets_rad, dtype=np.float64)
        return np.ascontiguousarray(np.concatenate((arm, hand)), dtype=np.float64)

    def _lift_targets(
        self,
        state: OptionEpisodeState,
        preparation: LiftPreparation | None,
    ) -> tuple[OptionEpisodeState, FloatArray]:
        if state.lift_start_arm_targets_rad is None:
            if preparation is None:
                raise ValueError("the first Lift policy step requires LiftPreparation")
            start = np.asarray(self._previous_output_rad[:ARM_POLICY_DIMENSION], dtype=np.float64).copy()
            state = replace(
                state,
                lift_start_arm_targets_rad=tuple(float(value) for value in start),
                lift_start_tcp_position_m=preparation.tcp_world_position_m,
                lift_start_tcp_quaternion_wxyz=preparation.tcp_world_quaternion_wxyz,
            )
        if preparation is None:
            raise ValueError("every Lift policy step requires LiftPreparation")
        if (
            state.lift_start_arm_targets_rad is None
            or state.lift_start_tcp_position_m is None
            or state.lift_start_tcp_quaternion_wxyz is None
        ):
            raise RuntimeError("lift trajectory preparation failed")
        if self.timing is None:  # pragma: no cover - established in __post_init__
            raise RuntimeError("Option timing was not initialized")
        fraction = smoothstep((state.phase_step + 1) / self.timing.lift_steps)
        previous = np.asarray(self._previous_output_rad[:ARM_POLICY_DIMENSION], dtype=np.float64)
        # Use a joint-space lift now that a strict +Z Cartesian path is not
        # required.  The model's negative elbow pitch raises the TCP; a
        # smaller wrist-pitch compensation limits excessive hand rotation.
        if state.lift_start_arm_targets_rad is None:  # pragma: no cover - checked above
            raise RuntimeError("lift start arm target is unavailable")
        target = np.asarray(state.lift_start_arm_targets_rad, dtype=np.float64).copy()
        target[3] += fraction * LIFT_ELBOW_BIAS_RAD
        target[5] += fraction * LIFT_WRIST_PITCH_COMPENSATION_RAD
        command_delta = target - previous
        command_delta = np.clip(
            command_delta,
            -LIFT_MAXIMUM_COMMAND_STEP_RAD,
            LIFT_MAXIMUM_COMMAND_STEP_RAD,
        )
        target = np.ascontiguousarray(previous + command_delta, dtype=np.float64)
        arm_lower = self.actuator_lower_limits_rad[:ARM_POLICY_DIMENSION]
        arm_upper = self.actuator_upper_limits_rad[:ARM_POLICY_DIMENSION]
        if np.any(target < arm_lower) or np.any(target > arm_upper):
            raise RuntimeError("computed lift target lies outside arm joint limits")
        state = replace(
            state,
            lift_final_arm_targets_rad=tuple(float(value) for value in target),
        )
        return state, target

    def observe_substep(self, observation: OptionSubstepObservation) -> OptionEpisodeState:
        """Aggregate contact and immediately fail on any hard-safety event."""

        if not self._policy_step_open:
            raise RuntimeError("begin_policy_step must precede substep observations")
        if not isinstance(observation, OptionSubstepObservation):
            raise TypeError("observation must be OptionSubstepObservation")
        numeric = (
            observation.thumb_side_force_n,
            observation.opposing_side_force_n,
            observation.net_hand_force_n,
            observation.object_lift_height_m,
        )
        finite = observation.state_finite and all(math.isfinite(value) for value in numeric)
        safety_flags = {
            OptionEvent.NonFinite: not finite,
            OptionEvent.Toppled: observation.toppled,
            OptionEvent.Dropped: observation.dropped,
            OptionEvent.IllegalContact: observation.illegal_contact,
        }
        new_safety = tuple(event for event in _HARD_SAFETY_EVENT_ORDER if safety_flags[event])
        if new_safety:
            combined = tuple(dict.fromkeys((*self.state.hard_safety_events, *new_safety)))
            events = tuple(dict.fromkeys((*self.state.transition_events, *new_safety)))
            self.state = replace(
                self.state,
                phase=OptionPhase.Failure,
                phase_step=0,
                hard_safety_events=combined,
                transition_events=events,
            )
            return self.state
        if self.state.script_active:
            bilateral = (
                observation.thumb_side_force_n >= CONTACT_FORCE_THRESHOLD_N
                and observation.opposing_side_force_n >= CONTACT_FORCE_THRESHOLD_N
            )
            self._step_bilateral |= bilateral
            self._step_supported |= (
                observation.net_hand_force_n >= CONTACT_FORCE_THRESHOLD_N
                and observation.object_lift_height_m >= LIFT_EVENT_HEIGHT_M
            )
            if observation.object_lift_height_m >= LIFT_EVENT_HEIGHT_M and not self.state.lift_5cm_event:
                self.state = replace(
                    self.state,
                    lift_5cm_event=True,
                    transition_events=(*self.state.transition_events, OptionEvent.Lift5cm),
                )
        return self.state

    def finish_policy_step(self, *, trigger_requested: bool, option_admissible: bool) -> OptionEpisodeState:
        """Finish one policy transition after all physical substeps were observed."""

        if not self._policy_step_open:
            raise RuntimeError("begin_policy_step must precede finish_policy_step")
        if not isinstance(trigger_requested, bool) or not isinstance(option_admissible, bool):
            raise TypeError("trigger_requested and option_admissible must be bool")
        self._policy_step_open = False
        if self.state.phase is OptionPhase.Reach:
            if self._previous_output_rad is None:  # pragma: no cover - reset invariant
                raise RuntimeError("Option target history is unavailable")
            self.state = process_reach_transition(
                self.state,
                trigger_requested=trigger_requested,
                option_admissible=option_admissible,
                executed_arm_targets_rad=np.ascontiguousarray(
                    self._previous_output_rad[:ARM_POLICY_DIMENSION], dtype=np.float64
                ),
            )
            return self.state
        if self.state.phase is OptionPhase.Failure:
            return self.state
        if self.state.phase in {OptionPhase.Success, OptionPhase.ScriptPending}:
            return self.state

        bilateral_steps = self.state.bilateral_contact_steps + 1 if self._step_bilateral else 0
        support_steps = self.state.supported_lift_steps + 1 if self._step_supported else 0
        maximum_support = max(self.state.maximum_supported_lift_steps, support_steps)
        completed = self.state.script_steps_completed + 1
        phase_step = self.state.phase_step + 1
        state = replace(
            self.state,
            phase_step=phase_step,
            script_steps_completed=completed,
            bilateral_contact_steps=bilateral_steps,
            supported_lift_steps=support_steps,
            maximum_supported_lift_steps=maximum_support,
            bilateral_contact_ever=self.state.bilateral_contact_ever or self._step_bilateral,
            bilateral_persistent_ever=(
                self.state.bilateral_persistent_ever or bilateral_steps >= BILATERAL_PERSISTENCE_STEPS
            ),
            supported_lift_ever=(self.state.supported_lift_ever or support_steps >= SUPPORTED_LIFT_PERSISTENCE_STEPS),
        )
        if self.timing is None:  # pragma: no cover - established in __post_init__
            raise RuntimeError("Option timing was not initialized")
        if phase_step == self.timing.steps_for(state.phase):
            if state.phase is OptionPhase.LiftHold:
                successful = (
                    state.acceptance_latched
                    and state.bilateral_persistent_ever
                    and state.supported_lift_ever
                    and state.lift_5cm_event
                    and not state.hard_safety_events
                )
                event = OptionEvent.CompleteSuccess if successful else OptionEvent.ScriptFailure
                state = replace(
                    state,
                    phase=OptionPhase.Success if successful else OptionPhase.Failure,
                    phase_step=0,
                    transition_events=(*state.transition_events, event),
                )
            else:
                state = replace(state, phase=_next_phase(state.phase), phase_step=0)
        elif phase_step > self.timing.steps_for(state.phase):
            raise RuntimeError("Option phase boundary was advanced more than once")
        self.state = state
        return self.state


__all__ = [
    "BILATERAL_PERSISTENCE_STEPS",
    "CLOSE_STEPS",
    "CONTACT_FORCE_THRESHOLD_N",
    "HOLD_STEPS",
    "LIFT_EVENT_HEIGHT_M",
    "LIFT_HOLD_STEPS",
    "LIFT_STEPS",
    "LIFT_TRAJECTORY_VERSION",
    "POLICY_STEP_S",
    "SETTLE_STEPS",
    "SUPPORTED_LIFT_PERSISTENCE_STEPS",
    "TOTAL_SCRIPT_STEPS",
    "LiftPreparation",
    "OptionController",
    "OptionEpisodeState",
    "OptionEvent",
    "OptionPhase",
    "OptionSubstepObservation",
    "OptionTiming",
    "process_reach_transition",
    "smoothstep",
    "solve_cartesian_lift_target",
]
