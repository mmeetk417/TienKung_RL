"""Pure episode outcome classification and inherited termination signals."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from tienkung_rl.config.models import TerminationConfig


class TerminationReason(StrEnum):
    """Stable failure strings shared by training, evaluation, viewer, and reports."""

    NONFINITE_OR_SIMULATION_ERROR = "nonfinite_or_simulation_error"
    ILLEGAL_CONTACT = "illegal_contact"
    EXCESSIVE_DISPLACEMENT = "excessive_displacement"
    TOPPLE = "topple"
    DROP = "drop"
    SCRIPT_FAILURE = "script_failure"
    TIMEOUT = "timeout"


TERMINATION_REASON_PRIORITY = (
    TerminationReason.NONFINITE_OR_SIMULATION_ERROR,
    TerminationReason.ILLEGAL_CONTACT,
    TerminationReason.EXCESSIVE_DISPLACEMENT,
    TerminationReason.TOPPLE,
    TerminationReason.DROP,
    TerminationReason.SCRIPT_FAILURE,
    TerminationReason.TIMEOUT,
)
TERMINATION_REASON_NAMES = tuple(reason.value for reason in TERMINATION_REASON_PRIORITY)
_PRIORITY_INDEX = {reason: index for index, reason in enumerate(TERMINATION_REASON_PRIORITY)}
_DANGEROUS_REASONS = frozenset(
    {
        TerminationReason.NONFINITE_OR_SIMULATION_ERROR,
        TerminationReason.ILLEGAL_CONTACT,
        TerminationReason.EXCESSIVE_DISPLACEMENT,
        TerminationReason.TOPPLE,
        TerminationReason.DROP,
    }
)


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    return float(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class InheritedTerminationInput:
    """Raw values needed to reproduce the legacy drop/far/contact semantics."""

    object_center_z_m: float
    table_top_z_m: float
    object_position_xy_in_environment_frame_m: tuple[float, float]
    contact_force_by_group_n: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in ("object_center_z_m", "table_top_z_m"):
            _real_scalar(getattr(self, name), name)
        if len(self.object_position_xy_in_environment_frame_m) != 2:
            raise ValueError("object_position_xy_in_environment_frame_m must contain x/y")
        for index, value in enumerate(self.object_position_xy_in_environment_frame_m):
            _real_scalar(value, f"object_position_xy_in_environment_frame_m[{index}]")
        copied: dict[str, float] = {}
        for key, value in self.contact_force_by_group_n.items():
            if not isinstance(key, str) or not key:
                raise TypeError("contact force group names must be non-empty strings")
            copied[key] = _real_scalar(value, f"contact_force_by_group_n[{key!r}]")
        object.__setattr__(self, "contact_force_by_group_n", MappingProxyType(copied))


@dataclass(frozen=True, slots=True, kw_only=True)
class InheritedTerminationSignals:
    dropped: bool
    object_far: bool
    bilateral_contact: bool
    lift_supported: bool


def evaluate_inherited_termination_signals(
    inputs: InheritedTerminationInput,
    config: TerminationConfig,
) -> InheritedTerminationSignals:
    """Evaluate versioned Scratch semantics without importing the legacy runtime."""

    if config.drop_frame_semantics != "object_center_z_below_table_top":
        raise ValueError("unsupported drop frame semantics")
    if config.far_frame_semantics != "environment_relative_xy_norm":
        raise ValueError("unsupported far frame semantics")
    required_groups = (
        config.thumb_contact_group,
        config.opposing_contact_group,
        config.lift_support_contact_group,
    )
    missing = tuple(name for name in required_groups if name not in inputs.contact_force_by_group_n)
    if missing:
        raise KeyError(f"missing configured contact force groups: {missing!r}")
    numeric = (
        inputs.object_center_z_m,
        inputs.table_top_z_m,
        *inputs.object_position_xy_in_environment_frame_m,
        *(inputs.contact_force_by_group_n[name] for name in required_groups),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("inherited termination signal inputs must be finite")
    if any(inputs.contact_force_by_group_n[name] < 0.0 for name in required_groups):
        raise ValueError("contact group forces must be non-negative")
    x_m, y_m = inputs.object_position_xy_in_environment_frame_m
    threshold = config.bilateral_contact_force_n
    return InheritedTerminationSignals(
        dropped=inputs.object_center_z_m < inputs.table_top_z_m - config.drop_below_table_top_m,
        object_far=math.hypot(x_m, y_m) > config.far_xy_distance_m,
        bilateral_contact=(
            inputs.contact_force_by_group_n[config.thumb_contact_group] >= threshold
            and inputs.contact_force_by_group_n[config.opposing_contact_group] >= threshold
        ),
        lift_supported=inputs.contact_force_by_group_n[config.lift_support_contact_group] >= threshold,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class OutcomeInput:
    """All reduced events and grasp measurements for one policy step."""

    step_index: int
    option_accepted: bool = False
    object_lift_height_m: float = 0.0
    completed_lift_hold: bool = False
    thumb_contact_force_n: float = 0.0
    opposing_contact_force_n: float = 0.0
    lift_support_force_n: float = 0.0
    toppled: bool = False
    dropped: bool = False
    object_far: bool = False
    excessive_displacement: bool = False
    illegal_contact: bool = False
    state_finite: bool = True
    simulation_error: bool = False
    script_failed: bool = False
    timed_out: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.step_index, bool) or not isinstance(self.step_index, int):
            raise TypeError("step_index must be an int")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        bool_names = (
            "option_accepted",
            "completed_lift_hold",
            "toppled",
            "dropped",
            "object_far",
            "excessive_displacement",
            "illegal_contact",
            "state_finite",
            "simulation_error",
            "script_failed",
            "timed_out",
        )
        for name in bool_names:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        for name in (
            "object_lift_height_m",
            "thumb_contact_force_n",
            "opposing_contact_force_n",
            "lift_support_force_n",
        ):
            value = _real_scalar(getattr(self, name), name)
            if math.isfinite(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class OutcomeEpisodeState:
    """Immutable classifier memory, including first occurrence of every failure."""

    last_step_index: int = -1
    option_accepted_ever: bool = False
    lift_5cm_ever: bool = False
    bilateral_contact_steps: int = 0
    bilateral_persistent_ever: bool = False
    lift_support_steps: int = 0
    lift_support_persistent_ever: bool = False
    completed_lift_hold_ever: bool = False
    all_states_finite: bool = True
    reason_first_steps: tuple[tuple[TerminationReason, int], ...] = ()

    def __post_init__(self) -> None:
        if self.last_step_index < -1:
            raise ValueError("last_step_index must be at least -1")
        if min(self.bilateral_contact_steps, self.lift_support_steps) < 0:
            raise ValueError("contact dwell counters must be non-negative")
        seen: set[TerminationReason] = set()
        for reason, step in self.reason_first_steps:
            if not isinstance(reason, TerminationReason):
                raise TypeError("reason_first_steps keys must be TerminationReason values")
            if reason in seen:
                raise ValueError("reason_first_steps cannot contain duplicate reasons")
            if step < 0 or step > self.last_step_index:
                raise ValueError("failure occurrence step lies outside classifier history")
            seen.add(reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class OutcomeResult:
    """Current unique classification and the complete ordered reason history."""

    lift_5cm_event: bool
    lift_5cm: bool
    success: bool
    unsafe: bool
    done: bool
    primary_reason: TerminationReason | None
    secondary_reasons: tuple[TerminationReason, ...]
    bilateral_contact: bool
    bilateral_persistent: bool
    lift_supported: bool
    lift_support_persistent: bool
    episode_state: OutcomeEpisodeState

    def __post_init__(self) -> None:
        if self.success and self.unsafe:
            raise ValueError("success and unsafe must be mutually exclusive")
        if self.done != (self.success or self.primary_reason is not None):
            raise ValueError("done must equal success-or-failure")
        if self.primary_reason is not None and self.primary_reason in self.secondary_reasons:
            raise ValueError("primary reason cannot also be secondary")
        if len(set(self.secondary_reasons)) != len(self.secondary_reasons):
            raise ValueError("secondary reasons must be unique")

    @property
    def primary_reason_name(self) -> str | None:
        return None if self.primary_reason is None else self.primary_reason.value

    @property
    def secondary_reason_names(self) -> tuple[str, ...]:
        return tuple(reason.value for reason in self.secondary_reasons)


def _step_failure_reasons(inputs: OutcomeInput, finite: bool) -> tuple[TerminationReason, ...]:
    flags = {
        TerminationReason.NONFINITE_OR_SIMULATION_ERROR: not finite or inputs.simulation_error,
        TerminationReason.ILLEGAL_CONTACT: inputs.illegal_contact,
        # Relative displacement from the reset pose is tracked for telemetry and
        # reward shaping, but it is no longer an unsafe termination condition.
        # Keep the inherited far-from-scene guard separately.
        TerminationReason.EXCESSIVE_DISPLACEMENT: inputs.object_far,
        TerminationReason.TOPPLE: inputs.toppled,
        TerminationReason.DROP: inputs.dropped,
        TerminationReason.SCRIPT_FAILURE: inputs.script_failed,
        TerminationReason.TIMEOUT: inputs.timed_out,
    }
    return tuple(reason for reason in TERMINATION_REASON_PRIORITY if flags[reason])


def _ordered_classification(
    reason_first_steps: tuple[tuple[TerminationReason, int], ...],
) -> tuple[TerminationReason | None, tuple[TerminationReason, ...]]:
    if not reason_first_steps:
        return None, ()
    ordered = sorted(reason_first_steps, key=lambda item: (item[1], _PRIORITY_INDEX[item[0]]))
    primary = ordered[0][0]
    return primary, tuple(reason for reason, _ in ordered[1:])


class OutcomeClassifier:
    """The single stateful wrapper around the pure outcome transition."""

    def __init__(self, config: TerminationConfig) -> None:
        config.validate()
        self._config = config
        self._state = OutcomeEpisodeState()

    @property
    def state(self) -> OutcomeEpisodeState:
        return self._state

    def reset(self) -> OutcomeEpisodeState:
        self._state = OutcomeEpisodeState()
        return self._state

    def step(self, inputs: OutcomeInput) -> OutcomeResult:
        result = classify_outcome(inputs, self._state, self._config)
        self._state = result.episode_state
        return result


def classify_outcome(
    inputs: OutcomeInput,
    episode_state: OutcomeEpisodeState,
    config: TerminationConfig,
) -> OutcomeResult:
    """Advance classification with earliest-step and same-step priority rules."""

    if inputs.step_index <= episode_state.last_step_index:
        raise ValueError("outcome steps must be supplied in strictly increasing order")
    numeric = (
        inputs.object_lift_height_m,
        inputs.thumb_contact_force_n,
        inputs.opposing_contact_force_n,
        inputs.lift_support_force_n,
    )
    finite = inputs.state_finite and all(math.isfinite(value) for value in numeric)
    threshold = config.bilateral_contact_force_n
    bilateral = finite and inputs.thumb_contact_force_n >= threshold and inputs.opposing_contact_force_n >= threshold
    supported = (
        finite
        and inputs.lift_support_force_n >= threshold
        and inputs.object_lift_height_m >= config.lift_event_height_m
    )
    bilateral_steps = episode_state.bilateral_contact_steps + 1 if bilateral else 0
    support_steps = episode_state.lift_support_steps + 1 if supported else 0
    bilateral_persistent = bilateral_steps >= config.bilateral_contact_dwell_steps
    support_persistent = support_steps >= config.lift_support_dwell_steps
    accepted_ever = episode_state.option_accepted_ever or inputs.option_accepted

    first_steps = dict(episode_state.reason_first_steps)
    for reason in _step_failure_reasons(inputs, finite):
        first_steps.setdefault(reason, inputs.step_index)
    stable_first_steps = tuple(sorted(first_steps.items(), key=lambda item: (item[1], _PRIORITY_INDEX[item[0]])))
    primary, secondary = _ordered_classification(stable_first_steps)
    unsafe = any(reason in _DANGEROUS_REASONS for reason in first_steps)

    lift_event = (
        not episode_state.lift_5cm_ever
        and accepted_ever
        and bilateral_persistent
        and inputs.object_lift_height_m >= config.lift_event_height_m
        and not unsafe
        and finite
        and primary is None
    )
    lift_ever = episode_state.lift_5cm_ever or lift_event
    completed_ever = episode_state.completed_lift_hold_ever or inputs.completed_lift_hold
    bilateral_ever = episode_state.bilateral_persistent_ever or bilateral_persistent
    support_ever = episode_state.lift_support_persistent_ever or support_persistent
    all_finite = episode_state.all_states_finite and finite and not inputs.simulation_error
    success = (
        accepted_ever
        and lift_ever
        and completed_ever
        and bilateral_ever
        and support_ever
        and all_finite
        and primary is None
    )
    state = OutcomeEpisodeState(
        last_step_index=inputs.step_index,
        option_accepted_ever=accepted_ever,
        lift_5cm_ever=lift_ever,
        bilateral_contact_steps=bilateral_steps,
        bilateral_persistent_ever=bilateral_ever,
        lift_support_steps=support_steps,
        lift_support_persistent_ever=support_ever,
        completed_lift_hold_ever=completed_ever,
        all_states_finite=all_finite,
        reason_first_steps=stable_first_steps,
    )
    return OutcomeResult(
        lift_5cm_event=lift_event,
        lift_5cm=lift_ever,
        success=success,
        unsafe=unsafe,
        done=success or primary is not None,
        primary_reason=primary,
        secondary_reasons=secondary,
        bilateral_contact=bilateral,
        bilateral_persistent=bilateral_persistent,
        lift_supported=supported,
        lift_support_persistent=support_persistent,
        episode_state=state,
    )


__all__ = [
    "TERMINATION_REASON_NAMES",
    "TERMINATION_REASON_PRIORITY",
    "InheritedTerminationInput",
    "InheritedTerminationSignals",
    "OutcomeClassifier",
    "OutcomeEpisodeState",
    "OutcomeInput",
    "OutcomeResult",
    "TerminationReason",
    "classify_outcome",
    "evaluate_inherited_termination_signals",
]
