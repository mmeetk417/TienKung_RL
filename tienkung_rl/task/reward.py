"""Pure, named reward calculation for reach-then-close transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType

from tienkung_rl.config.models import RewardConfig

REWARD_TERM_NAMES = (
    "position_progress",
    "orientation_progress",
    "step_penalty",
    "joint_velocity_penalty",
    "action_rate_penalty",
    "object_displacement_penalty",
    "unsafe_object_displacement_penalty",
    "accepted_trigger",
    "rejected_trigger",
    "admissible_wait",
    "lift_5cm",
    "complete_success",
    "script_failure",
)


class RewardNonFiniteError(FloatingPointError):
    """A reward input or named result was non-finite.

    ``term_context`` is deliberately retained even when only one term is bad so
    a runner can report the entire calculation instead of a context-free NaN.
    """

    def __init__(self, message: str, *, term_context: Mapping[str, float]) -> None:
        self.term_context = MappingProxyType(dict(term_context))
        rendered = ", ".join(f"{name}={value!r}" for name, value in self.term_context.items())
        super().__init__(f"{message}; reward breakdown: {rendered}")


def _real_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    return float(value)


def _bool(value: object, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _vector7(values: tuple[float, ...], name: str) -> None:
    if len(values) != 7:
        raise ValueError(f"{name} must contain seven values")
    for index, value in enumerate(values):
        _real_scalar(value, f"{name}[{index}]")


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardState:
    """Minimal physics state needed on either side of one transition."""

    position_error_m: float
    orientation_error_quaternion_wxyz: tuple[float, float, float, float]
    arm_joint_velocities_rad_s: tuple[float, ...]
    arm_action: tuple[float, ...]
    object_displacement_m: float

    def __post_init__(self) -> None:
        for name in ("position_error_m", "object_displacement_m"):
            value = _real_scalar(getattr(self, name), name)
            if math.isfinite(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if len(self.orientation_error_quaternion_wxyz) != 4:
            raise ValueError("orientation_error_quaternion_wxyz must contain four values")
        for index, value in enumerate(self.orientation_error_quaternion_wxyz):
            _real_scalar(value, f"orientation_error_quaternion_wxyz[{index}]")
        _vector7(self.arm_joint_velocities_rad_s, "arm_joint_velocities_rad_s")
        _vector7(self.arm_action, "arm_action")


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardEvents:
    """Transition events; strict Ready fields are diagnostics and are ignored."""

    reach_phase: bool = True
    option_admissible: bool = False
    trigger_requested: bool = False
    trigger_accepted: bool = False
    trigger_rejected: bool = False
    lift_5cm_event: bool = False
    complete_success: bool = False
    script_failure: bool = False
    strict_ready: bool = False
    newly_strict_ready: bool = False
    strict_ready_ever: bool = False

    def __post_init__(self) -> None:
        for item in fields(self):
            _bool(getattr(self, item.name), item.name)
        if self.trigger_accepted and self.trigger_rejected:
            raise ValueError("a trigger cannot be accepted and rejected")
        if (self.trigger_accepted or self.trigger_rejected) and not self.trigger_requested:
            raise ValueError("accepted and rejected triggers must be requested")
        if self.complete_success and self.script_failure:
            raise ValueError("complete success and script failure are mutually exclusive")


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardEpisodeState:
    """Episode-local one-shot reward latches."""

    accepted_trigger_rewarded: bool = False
    lift_5cm_rewarded: bool = False
    complete_success_rewarded: bool = False
    script_failure_penalized: bool = False

    def __post_init__(self) -> None:
        for item in fields(self):
            _bool(getattr(self, item.name), item.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardTermSwitches:
    """Independent term switches used by ablations and term-level tests."""

    position_progress: bool = True
    orientation_progress: bool = True
    step_penalty: bool = True
    joint_velocity_penalty: bool = True
    action_rate_penalty: bool = True
    object_displacement_penalty: bool = True
    unsafe_object_displacement_penalty: bool = True
    accepted_trigger: bool = True
    rejected_trigger: bool = True
    admissible_wait: bool = True
    lift_5cm: bool = True
    complete_success: bool = True
    script_failure: bool = True

    def __post_init__(self) -> None:
        if tuple(item.name for item in fields(self)) != REWARD_TERM_NAMES:
            raise RuntimeError("reward switch names differ from the stable reward schema")
        for item in fields(self):
            _bool(getattr(self, item.name), item.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardBreakdown:
    """Signed, uniquely named reward terms and their exact finite sum."""

    position_progress: float
    orientation_progress: float
    step_penalty: float
    joint_velocity_penalty: float
    action_rate_penalty: float
    object_displacement_penalty: float
    unsafe_object_displacement_penalty: float
    accepted_trigger: float
    rejected_trigger: float
    admissible_wait: float
    lift_5cm: float
    complete_success: float
    script_failure: float
    total: float

    def __post_init__(self) -> None:
        field_names = tuple(item.name for item in fields(self))
        if field_names != (*REWARD_TERM_NAMES, "total"):
            raise RuntimeError("RewardBreakdown names differ from the stable reward schema")
        term_values = tuple(float(getattr(self, name)) for name in REWARD_TERM_NAMES)
        context = dict(zip(REWARD_TERM_NAMES, term_values, strict=True))
        context["total"] = float(self.total)
        if not all(math.isfinite(value) for value in context.values()):
            raise RewardNonFiniteError("non-finite named reward", term_context=context)
        expected = math.fsum(term_values)
        if self.total != expected:
            raise ValueError(f"reward total {self.total!r} does not equal named-term sum {expected!r}")

    @property
    def terms(self) -> Mapping[str, float]:
        return MappingProxyType({name: float(getattr(self, name)) for name in REWARD_TERM_NAMES})

    @property
    def all_terms(self) -> Mapping[str, float]:
        return MappingProxyType({**self.terms, "total": self.total})


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardResult:
    breakdown: RewardBreakdown
    episode_state: RewardEpisodeState


def _orientation_error_rad(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    norm = math.sqrt(math.fsum(value * value for value in quaternion_wxyz))
    if not math.isfinite(norm):
        return math.nan
    if norm <= 0.0:
        raise ValueError("orientation error quaternion must have positive norm")
    scalar = min(1.0, abs(quaternion_wxyz[0]) / norm)
    return 2.0 * math.acos(scalar)


def _quality(error: float, scale: float) -> float:
    return math.exp(-((error / scale) ** 2))


def _mean_square(values: tuple[float, ...]) -> float:
    return math.fsum(value * value for value in values) / len(values)


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return value
    return min(1.0, max(0.0, value))


def _checked_configuration(config: RewardConfig) -> None:
    numeric_names = (
        "potential_discount",
        "position_potential_weight",
        "orientation_potential_weight",
        "step_penalty",
        "joint_velocity_penalty_weight",
        "action_rate_penalty_weight",
        "object_displacement_penalty_weight",
        "unsafe_object_displacement_penalty",
        "accepted_trigger_reward",
        "rejected_trigger_penalty",
        "admissible_wait_penalty",
        "lift_5cm_reward",
        "complete_success_reward",
        "script_failure_penalty",
        "ready_reward",
        "position_potential_scale_m",
        "orientation_potential_scale_rad",
        "object_displacement_scale_m",
    )
    context = {name: float(getattr(config, name)) for name in numeric_names}
    if not all(math.isfinite(value) for value in context.values()):
        raise RewardNonFiniteError("non-finite reward configuration", term_context=context)
    if not 0.0 < config.potential_discount <= 1.0:
        raise ValueError("potential_discount must be in (0, 1]")
    if min(context[name] for name in numeric_names[1:]) < 0.0:
        raise ValueError("reward weights and scales must be non-negative")
    for name in (
        "position_potential_scale_m",
        "orientation_potential_scale_rad",
        "object_displacement_scale_m",
    ):
        if context[name] <= 0.0:
            raise ValueError(f"reward.{name} must be positive")
    if config.ready_reward != 0.0:
        raise ValueError("strict Ready reward must remain exactly zero")
    if config.potential_progress_semantics != "gamma_phi_next_minus_phi_current":
        raise ValueError("unsupported potential progress semantics")


def _state_values(state: RewardState, label: str) -> dict[str, float]:
    values = {
        f"{label}.position_error_m": state.position_error_m,
        **{
            f"{label}.orientation_error_quaternion_wxyz[{index}]": value
            for index, value in enumerate(state.orientation_error_quaternion_wxyz)
        },
        **{
            f"{label}.arm_joint_velocities_rad_s[{index}]": value
            for index, value in enumerate(state.arm_joint_velocities_rad_s)
        },
        **{f"{label}.arm_action[{index}]": value for index, value in enumerate(state.arm_action)},
        f"{label}.object_displacement_m": state.object_displacement_m,
    }
    return values


def _require_finite_states(before: RewardState, after: RewardState) -> None:
    state_values = {**_state_values(before, "before"), **_state_values(after, "after")}
    nonfinite = tuple(name for name, value in state_values.items() if not math.isfinite(value))
    if nonfinite:
        context = {name: math.nan for name in (*REWARD_TERM_NAMES, "total")}
        raise RewardNonFiniteError(
            f"non-finite reward state input(s): {', '.join(nonfinite)}",
            term_context=context,
        )


def compute_reward(
    before: RewardState,
    after: RewardState,
    events: RewardEvents,
    episode_state: RewardEpisodeState,
    config: RewardConfig,
    *,
    switches: RewardTermSwitches | None = None,
) -> RewardResult:
    """Compute all signed reward terms and advance one-shot latches."""

    _checked_configuration(config)
    _require_finite_states(before, after)
    enabled = switches if switches is not None else RewardTermSwitches()
    reach = float(events.reach_phase)

    before_position_quality = _quality(before.position_error_m, config.position_potential_scale_m)
    after_position_quality = _quality(after.position_error_m, config.position_potential_scale_m)
    before_orientation_quality = _quality(
        _orientation_error_rad(before.orientation_error_quaternion_wxyz),
        config.orientation_potential_scale_rad,
    )
    after_orientation_quality = _quality(
        _orientation_error_rad(after.orientation_error_quaternion_wxyz),
        config.orientation_potential_scale_rad,
    )

    raw: dict[str, float] = {
        "position_progress": reach
        * config.position_potential_weight
        * (config.potential_discount * after_position_quality - before_position_quality),
        "orientation_progress": reach
        * config.orientation_potential_weight
        * (config.potential_discount * after_orientation_quality - before_orientation_quality),
        "step_penalty": -reach * config.step_penalty,
        "joint_velocity_penalty": (
            -reach * config.joint_velocity_penalty_weight * _mean_square(after.arm_joint_velocities_rad_s)
        ),
        "action_rate_penalty": -reach
        * config.action_rate_penalty_weight
        * _mean_square(
            tuple(current - previous for current, previous in zip(after.arm_action, before.arm_action, strict=True))
        ),
        "object_displacement_penalty": -reach
        * config.object_displacement_penalty_weight
        * (after.object_displacement_m / config.object_displacement_scale_m) ** 2,
        "unsafe_object_displacement_penalty": -reach
        * config.unsafe_object_displacement_penalty
        * float(after.object_displacement_m > config.object_displacement_scale_m),
        "accepted_trigger": config.accepted_trigger_reward
        * float(events.trigger_accepted and not episode_state.accepted_trigger_rewarded),
        "rejected_trigger": -config.rejected_trigger_penalty * float(events.trigger_rejected),
        "admissible_wait": -config.admissible_wait_penalty
        * float(events.reach_phase and events.option_admissible and not events.trigger_requested),
        "lift_5cm": config.lift_5cm_reward * float(events.lift_5cm_event and not episode_state.lift_5cm_rewarded),
        "complete_success": config.complete_success_reward
        * float(events.complete_success and not episode_state.complete_success_rewarded),
        "script_failure": -config.script_failure_penalty
        * float(events.script_failure and not episode_state.script_failure_penalized),
    }
    raw_context = {**raw, "total": math.fsum(raw[name] for name in REWARD_TERM_NAMES)}
    if not all(math.isfinite(value) for value in raw_context.values()):
        raise RewardNonFiniteError("non-finite reward calculation", term_context=raw_context)
    terms = {name: value if getattr(enabled, name) else 0.0 for name, value in raw.items()}
    total = math.fsum(terms[name] for name in REWARD_TERM_NAMES)
    context = {**terms, "total": total}
    if not all(math.isfinite(value) for value in context.values()):
        raise RewardNonFiniteError("non-finite reward calculation", term_context=context)
    breakdown = RewardBreakdown(**terms, total=total)
    next_state = RewardEpisodeState(
        accepted_trigger_rewarded=(episode_state.accepted_trigger_rewarded or events.trigger_accepted),
        lift_5cm_rewarded=(episode_state.lift_5cm_rewarded or events.lift_5cm_event),
        complete_success_rewarded=(episode_state.complete_success_rewarded or events.complete_success),
        script_failure_penalized=(episode_state.script_failure_penalized or events.script_failure),
    )
    return RewardResult(breakdown=breakdown, episode_state=next_state)


__all__ = [
    "REWARD_TERM_NAMES",
    "RewardBreakdown",
    "RewardEpisodeState",
    "RewardEvents",
    "RewardNonFiniteError",
    "RewardResult",
    "RewardState",
    "RewardTermSwitches",
    "compute_reward",
]
