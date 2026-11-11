"""Stable telemetry events, metric schema, and finite aggregators.

The simulator, task, PPO, curriculum, and runner modules emit the typed events
defined here.  This module intentionally has no dependency on MuJoCo, Torch, or
TensorBoard so aggregation can be unit tested without a training runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

Scalar: TypeAlias = str | int | float | bool | None


class MetricLevel(StrEnum):
    STEP = "step"
    EPISODE = "episode"
    WINDOW = "window"
    RUN = "run"


class Reduction(StrEnum):
    MEAN = "mean"
    RATE = "rate"
    RMS = "rms"
    STD = "std"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"


class DiagnosticKind(StrEnum):
    NONFINITE = "nonfinite"
    WORKER = "worker"
    MUJOCO = "mujoco"
    SOLVER = "solver"
    CAPACITY = "capacity"


ADMISSION_FAILURE_REASONS = (
    "finite_state",
    "valid_target",
    "reach_phase",
    "position_error",
    "orientation_error",
    "grasp_center_speed",
    "object_tilt",
    "illegal_contact_impulse",
)

TERMINATION_REASONS = (
    "nonfinite_or_simulation_error",
    "illegal_contact",
    "excessive_displacement",
    "topple",
    "drop",
    "script_failure",
    "timeout",
)

REWARD_TERMS = (
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
    "total",
)

RANDOMIZATION_FIELDS = (
    "cylinder_radius_m",
    "cylinder_height_m",
    "cylinder_mass_kg",
    "hand_sliding_friction",
    "table_sliding_friction",
)

CURRICULUM_STAGES = ("S0", "S1", "S2", "S3")
TELEMETRY_STATE_VERSION = "tienkung_telemetry_aggregator_state_v2"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    level: MetricLevel
    unit: str
    reduction: Reduction
    description: str
    conditional: bool = False


def _spec(
    name: str,
    level: MetricLevel,
    unit: str,
    reduction: Reduction,
    description: str,
    *,
    conditional: bool = False,
) -> MetricSpec:
    return MetricSpec(name, level, unit, reduction, description, conditional)


_BASE_SPECS = (
    _spec("throughput.total_steps", MetricLevel.RUN, "policy_step", Reduction.LAST, "Total vectorized policy steps."),
    _spec("throughput.steps_per_second", MetricLevel.RUN, "policy_step/s", Reduction.LAST, "Policy throughput."),
    _spec("time.sampling_seconds", MetricLevel.RUN, "s", Reduction.LAST, "Cumulative sampling time."),
    _spec("time.learning_seconds", MetricLevel.RUN, "s", Reduction.LAST, "Cumulative learning time."),
    _spec("time.total_seconds", MetricLevel.RUN, "s", Reduction.LAST, "Cumulative wall time."),
    _spec("time.eta_seconds", MetricLevel.RUN, "s", Reduction.LAST, "Estimated remaining time."),
    _spec("ppo.policy_loss", MetricLevel.STEP, "1", Reduction.MEAN, "PPO policy loss."),
    _spec("ppo.value_loss", MetricLevel.STEP, "1", Reduction.MEAN, "PPO value loss."),
    _spec("ppo.entropy", MetricLevel.STEP, "nat", Reduction.MEAN, "Policy entropy."),
    _spec("ppo.kl", MetricLevel.STEP, "nat", Reduction.MEAN, "Policy KL divergence."),
    _spec("ppo.total_loss", MetricLevel.STEP, "1", Reduction.MEAN, "Total optimized loss."),
    _spec("ppo.gradient_norm", MetricLevel.STEP, "1", Reduction.MEAN, "Gradient norm."),
    _spec("ppo.gradient_finite_rate", MetricLevel.STEP, "ratio", Reduction.RATE, "Finite-gradient update rate."),
    _spec("ppo.log_probability", MetricLevel.STEP, "nat", Reduction.MEAN, "Policy log probability."),
    _spec("action.mean", MetricLevel.STEP, "normalized_action", Reduction.MEAN, "Mean arm action."),
    _spec("action.rms", MetricLevel.STEP, "normalized_action", Reduction.RMS, "RMS arm action."),
    _spec(
        "action.std",
        MetricLevel.STEP,
        "normalized_action",
        Reduction.STD,
        "Population standard deviation of arm actions.",
    ),
    _spec("action.clip_rate", MetricLevel.STEP, "ratio", Reduction.RATE, "Arm-action clipping rate."),
    _spec(
        "trigger.request_rate",
        MetricLevel.STEP,
        "ratio",
        Reduction.RATE,
        "Trigger requests per evaluated policy sample.",
    ),
    _spec(
        "trigger.accept_rate",
        MetricLevel.STEP,
        "ratio",
        Reduction.RATE,
        "Accepted requests per request.",
        conditional=True,
    ),
    _spec(
        "trigger.reject_rate",
        MetricLevel.STEP,
        "ratio",
        Reduction.RATE,
        "Rejected requests per request.",
        conditional=True,
    ),
    _spec("admission.admissible_rate", MetricLevel.STEP, "ratio", Reduction.RATE, "Admissible policy samples."),
    _spec(
        "admission.accepted_position_error_m",
        MetricLevel.STEP,
        "m",
        Reduction.MEAN,
        "Position error at acceptance.",
        conditional=True,
    ),
    _spec(
        "admission.accepted_orientation_error_rad",
        MetricLevel.STEP,
        "rad",
        Reduction.MEAN,
        "Orientation error at acceptance.",
        conditional=True,
    ),
    _spec("ready.strict_rate", MetricLevel.STEP, "ratio", Reduction.RATE, "Strict Ready policy-sample rate."),
    _spec(
        "outcome.strict_ready_episode_rate",
        MetricLevel.EPISODE,
        "ratio",
        Reduction.RATE,
        "Episodes that ever reached strict Ready.",
    ),
    _spec("outcome.lift_5cm_rate", MetricLevel.EPISODE, "ratio", Reduction.RATE, "Episodes with the Lift5cm event."),
    _spec(
        "outcome.complete_success_rate", MetricLevel.EPISODE, "ratio", Reduction.RATE, "Complete-success episode rate."
    ),
    _spec("outcome.dangerous_rate", MetricLevel.EPISODE, "ratio", Reduction.RATE, "Dangerous episode rate."),
    _spec("episode.length_steps", MetricLevel.EPISODE, "policy_step", Reduction.MEAN, "Mean completed episode length."),
    _spec("curriculum.stage_index", MetricLevel.WINDOW, "index", Reduction.LAST, "Current S0-S3 stage index."),
    _spec(
        "curriculum.window_progress", MetricLevel.WINDOW, "ratio", Reduction.LAST, "Current promotion-window progress."
    ),
    _spec(
        "curriculum.promotion_rate", MetricLevel.WINDOW, "ratio", Reduction.RATE, "Promotion decisions that promoted."
    ),
    _spec("curriculum.stalled_rate", MetricLevel.WINDOW, "ratio", Reduction.RATE, "Completed windows marked stalled."),
    _spec(
        "curriculum.consecutive_passing_windows",
        MetricLevel.WINDOW,
        "window",
        Reduction.LAST,
        "Consecutive passing windows.",
    ),
    _spec("contact.count", MetricLevel.STEP, "contact", Reduction.MEAN, "Mean contact count."),
    _spec("contact.maximum_force_n", MetricLevel.STEP, "N", Reduction.MAX, "Maximum observed contact force."),
    _spec("contact.maximum_impulse_n_s", MetricLevel.STEP, "N*s", Reduction.MAX, "Maximum observed contact impulse."),
    _spec("object.tilt_rad", MetricLevel.STEP, "rad", Reduction.MEAN, "Mean object tilt."),
    _spec("object.maximum_displacement_m", MetricLevel.STEP, "m", Reduction.MAX, "Maximum displacement from reset."),
    _spec("warning.mujoco.count", MetricLevel.RUN, "warning", Reduction.SUM, "MuJoCo warnings."),
    _spec("warning.solver.count", MetricLevel.RUN, "warning", Reduction.SUM, "Solver anomalies."),
    _spec("warning.capacity.count", MetricLevel.RUN, "warning", Reduction.SUM, "Contact/solver capacity anomalies."),
    _spec("warning.worker.count", MetricLevel.RUN, "warning", Reduction.SUM, "Worker exceptions."),
    _spec("warning.nonfinite.count", MetricLevel.RUN, "warning", Reduction.SUM, "Non-finite inputs or runtime values."),
    _spec("system.memory_rss_bytes", MetricLevel.RUN, "byte", Reduction.LAST, "Resident memory."),
    _spec("system.cpu_percent", MetricLevel.RUN, "percent", Reduction.LAST, "Process CPU utilization."),
    _spec("system.num_envs", MetricLevel.RUN, "environment", Reduction.LAST, "Parallel environment count."),
    _spec("system.worker_threads", MetricLevel.RUN, "thread", Reduction.LAST, "Worker thread count."),
    _spec("system.physics_steps", MetricLevel.RUN, "physics_step", Reduction.LAST, "Cumulative physics steps."),
    _spec("system.policy_steps", MetricLevel.RUN, "policy_step", Reduction.LAST, "Cumulative policy steps."),
    _spec(
        "system.sim_steps_per_second",
        MetricLevel.RUN,
        "physics_step/s",
        Reduction.LAST,
        "Physics simulation throughput.",
    ),
    _spec("system.realtime_factor", MetricLevel.RUN, "ratio", Reduction.LAST, "Simulated time divided by wall time."),
    _spec("randomization.fixed_torsional_friction", MetricLevel.RUN, "1", Reduction.LAST, "Fixed torsional friction."),
    _spec("randomization.fixed_rolling_friction", MetricLevel.RUN, "1", Reduction.LAST, "Fixed rolling friction."),
)

_DYNAMIC_SPECS = (
    tuple(
        _spec(
            f"admission.failure.{reason}_rate",
            MetricLevel.STEP,
            "ratio",
            Reduction.RATE,
            f"Admission evaluations failing {reason}.",
        )
        for reason in ADMISSION_FAILURE_REASONS
    )
    + tuple(
        _spec(
            f"termination.primary.{reason}_rate",
            MetricLevel.EPISODE,
            "ratio",
            Reduction.RATE,
            f"Episodes with primary termination reason {reason}.",
        )
        for reason in TERMINATION_REASONS
    )
    + tuple(
        _spec(
            f"termination.auxiliary.{reason}_rate",
            MetricLevel.EPISODE,
            "ratio",
            Reduction.RATE,
            f"Episodes containing auxiliary termination reason {reason}.",
        )
        for reason in TERMINATION_REASONS
    )
    + tuple(
        _spec(f"reward.{term}", MetricLevel.EPISODE, "reward", Reduction.MEAN, f"Episode total for reward term {term}.")
        for term in REWARD_TERMS
    )
    + tuple(
        _spec(
            f"curriculum.episodes_at_reset.{stage}",
            MetricLevel.EPISODE,
            "episode",
            Reduction.SUM,
            f"Episodes assigned to {stage} at reset.",
        )
        for stage in CURRICULUM_STAGES
    )
    + tuple(
        _spec(
            f"randomization.{name}.{reduction.value}",
            MetricLevel.EPISODE,
            "m" if name.endswith("_m") else "kg" if name.endswith("_kg") else "1",
            reduction,
            f"{reduction.value} sampled {name}.",
        )
        for name in RANDOMIZATION_FIELDS
        for reduction in (Reduction.MEAN, Reduction.STD, Reduction.MIN, Reduction.MAX)
    )
)

METRIC_SPECS = _BASE_SPECS + _DYNAMIC_SPECS
if len({item.name for item in METRIC_SPECS}) != len(METRIC_SPECS):  # pragma: no cover - import-time invariant
    raise RuntimeError("duplicate telemetry metric name")
METRIC_SCHEMA: Mapping[str, MetricSpec] = MappingProxyType({item.name: item for item in METRIC_SPECS})


@dataclass(frozen=True)
class MetricValue:
    """A finite aggregate plus its explicit sample numerator/denominator."""

    value: float
    count: int
    numerator: float
    denominator: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.value) or not math.isfinite(self.numerator):
            raise ValueError("aggregate values must be finite")
        if self.count < 0 or self.denominator < 0:
            raise ValueError("aggregate counts must be non-negative")
        if self.count != self.denominator:
            raise ValueError("metric count and denominator must match")
        if self.count == 0 and (self.value != 0.0 or self.numerator != 0.0):
            raise ValueError("empty aggregates must use value=0 and numerator=0")


class ConditionalAggregator:
    """Numerator/denominator mean that remains finite for an empty condition."""

    def __init__(self) -> None:
        self.numerator = 0.0
        self.denominator = 0

    def add(self, value: float, *, condition: bool = True) -> None:
        if not condition:
            return
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("conditional aggregate input must be finite")
        new_numerator = self.numerator + number
        if not math.isfinite(new_numerator):
            raise OverflowError("conditional aggregate overflow")
        self.numerator = new_numerator
        self.denominator += 1

    def result(self) -> MetricValue:
        if self.denominator == 0:
            return MetricValue(0.0, 0, 0.0, 0)
        return MetricValue(
            self.numerator / self.denominator,
            self.denominator,
            self.numerator,
            self.denominator,
        )


@dataclass(frozen=True)
class TelemetryDiagnostic:
    kind: DiagnosticKind
    message: str
    count: int = 1
    env_id: int | None = None
    policy_step: int | None = None
    config_fingerprint: str | None = None
    context: Mapping[str, Scalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("diagnostic message must not be empty")
        if self.count < 1:
            raise ValueError("diagnostic count must be positive")


class NonFiniteTelemetryError(RuntimeError):
    """Raised after a non-finite diagnostic has been retained for reporting."""


@dataclass(frozen=True)
class StepEvent:
    iteration: int
    env_id: int
    policy_step: int
    physics_steps: int
    actions: Sequence[float]
    action_clipped: Sequence[bool]
    option_admissible: bool
    trigger_requested: bool
    trigger_accepted: bool
    trigger_rejected: bool
    strict_ready: bool
    admission_failures: Sequence[str] = ()
    accepted_position_error_m: float | None = None
    accepted_orientation_error_rad: float | None = None
    contact_count: int = 0
    maximum_contact_force_n: float = 0.0
    maximum_contact_impulse_n_s: float = 0.0
    object_tilt_rad: float = 0.0
    object_displacement_m: float = 0.0


@dataclass(frozen=True)
class PPOUpdateEvent:
    iteration: int
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    total_loss: float
    gradient_norm: float
    gradients_finite: bool
    log_probability: float


@dataclass(frozen=True)
class EpisodeEvent:
    iteration: int
    env_id: int
    episode_length_steps: int
    stage_at_reset: str
    strict_ready_ever: bool
    lift_5cm: bool
    complete_success: bool
    dangerous: bool
    primary_termination_reason: str | None
    auxiliary_termination_reasons: Sequence[str] = ()
    reward_totals: Mapping[str, float] = field(default_factory=dict)
    randomization_samples: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumWindowEvent:
    iteration: int
    stage: str
    completed_episodes: int
    target_episodes: int
    promoted: bool
    stalled: bool
    consecutive_passing_windows: int


@dataclass(frozen=True)
class RunEvent:
    iteration: int
    total_steps: int
    steps_per_second: float
    sampling_seconds: float
    learning_seconds: float
    total_seconds: float
    eta_seconds: float
    memory_rss_bytes: int
    cpu_percent: float
    num_envs: int
    worker_threads: int
    physics_steps: int
    policy_steps: int
    sim_steps_per_second: float
    realtime_factor: float
    fixed_torsional_friction: float
    fixed_rolling_friction: float
    randomization_version: str
    bottleneck: str = ""


@dataclass(frozen=True)
class DiagnosticEvent:
    iteration: int
    diagnostic: TelemetryDiagnostic


TelemetryEvent: TypeAlias = (
    StepEvent | PPOUpdateEvent | EpisodeEvent | CurriculumWindowEvent | RunEvent | DiagnosticEvent
)


@dataclass(frozen=True)
class TelemetrySnapshot:
    iteration: int
    metrics: Mapping[str, MetricValue]
    diagnostics: tuple[TelemetryDiagnostic, ...]
    randomization_version: str | None = None
    bottleneck: str | None = None

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError("snapshot iteration must be non-negative")
        if set(self.metrics) != set(METRIC_SCHEMA):
            raise ValueError("snapshot metrics do not exactly match the stable schema")


class _MetricAccumulator:
    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.last = 0.0

    def add(self, value: float) -> None:
        total = self.total + value
        total_squared = self.total_squared
        if self.spec.reduction in (Reduction.RMS, Reduction.STD):
            total_squared += value * value
        if not math.isfinite(total) or not math.isfinite(total_squared):
            raise OverflowError(f"aggregate overflow for {self.spec.name}")
        self.count += 1
        self.total = total
        self.total_squared = total_squared
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        self.last = value

    def result(self) -> MetricValue:
        if self.count == 0:
            return MetricValue(0.0, 0, 0.0, 0)
        reduction = self.spec.reduction
        if reduction in (Reduction.MEAN, Reduction.RATE):
            value = self.total / self.count
            numerator = self.total
        elif reduction is Reduction.RMS:
            value = math.sqrt(max(0.0, self.total_squared / self.count))
            numerator = self.total_squared
        elif reduction is Reduction.STD:
            mean = self.total / self.count
            value = math.sqrt(max(0.0, self.total_squared / self.count - mean * mean))
            numerator = self.total
        elif reduction is Reduction.SUM:
            value = self.total
            numerator = self.total
        elif reduction is Reduction.MAX:
            value = self.maximum
            numerator = self.maximum
        elif reduction is Reduction.MIN:
            value = self.minimum
            numerator = self.minimum
        else:
            value = self.last
            numerator = self.last
        if not math.isfinite(value):  # defensive check; finite inputs should make this unreachable
            raise RuntimeError(f"aggregator generated non-finite value for {self.spec.name}")
        return MetricValue(value, self.count, numerator, self.count)


class TelemetryAggregator:
    """Aggregate typed production events into the complete stable metric schema."""

    def __init__(self) -> None:
        self._values = {name: _MetricAccumulator(spec) for name, spec in METRIC_SCHEMA.items()}
        self._diagnostics: list[TelemetryDiagnostic] = []
        self._randomization_version: str | None = None
        self._bottleneck: str | None = None

    def _add(self, name: str, value: float | int | bool) -> None:
        try:
            number = float(value)
        except OverflowError:
            self._retain_nonfinite(name, value)
        if not math.isfinite(number):
            self._retain_nonfinite(name, number)
        try:
            self._values[name].add(number)
        except OverflowError:
            self._retain_nonfinite(name, number)

    def _retain_nonfinite(self, name: str, value: object) -> None:
        diagnostic = TelemetryDiagnostic(
            kind=DiagnosticKind.NONFINITE,
            message=f"non-finite telemetry value for {name}",
            context={"metric": name, "value": repr(value)},
        )
        self._diagnostics.append(diagnostic)
        self._values["warning.nonfinite.count"].add(1.0)
        raise NonFiniteTelemetryError(diagnostic.message)

    def _validate_finite_event(self, event: object) -> None:
        def visit(value: object, path: str) -> None:
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, float):
                if not math.isfinite(value):
                    self._retain_nonfinite(path, value)
                return
            if is_dataclass(value) and not isinstance(value, type):
                for item in fields(value):
                    visit(getattr(value, item.name), f"{path}.{item.name}")
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(item, f"{path}.{key}")
                return
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")

        visit(event, type(event).__name__)

    def record_step(self, event: StepEvent) -> None:
        self._validate_finite_event(event)
        if event.iteration < 0 or event.env_id < 0 or event.policy_step < 0 or event.physics_steps < 1:
            raise ValueError("step identifiers must be non-negative and physics_steps must be positive")
        if len(event.actions) != 7 or len(event.action_clipped) != 7:
            raise ValueError("step event requires seven actions and seven clipping flags")
        if event.trigger_accepted and (not event.trigger_requested or not event.option_admissible):
            raise ValueError("an accepted trigger must be requested and admissible")
        if event.trigger_rejected and (not event.trigger_requested or event.trigger_accepted):
            raise ValueError("a rejected trigger must be requested and not accepted")
        if event.trigger_requested != (event.trigger_accepted or event.trigger_rejected):
            raise ValueError("every request must be classified as accepted or rejected")
        unknown_failures = set(event.admission_failures) - set(ADMISSION_FAILURE_REASONS)
        if unknown_failures:
            raise ValueError(f"unknown admission failure reason(s): {sorted(unknown_failures)}")
        if event.option_admissible and event.admission_failures:
            raise ValueError("admissible events cannot contain admission failures")
        if event.trigger_accepted and (
            event.accepted_position_error_m is None or event.accepted_orientation_error_rad is None
        ):
            raise ValueError("accepted events require position and orientation errors")
        if event.contact_count < 0:
            raise ValueError("contact count must be non-negative")
        if (
            min(
                event.maximum_contact_force_n,
                event.maximum_contact_impulse_n_s,
                event.object_tilt_rad,
                event.object_displacement_m,
            )
            < 0
        ):
            raise ValueError("contact/object magnitudes must be non-negative")

        for action in event.actions:
            self._add("action.mean", action)
            self._add("action.rms", action)
            self._add("action.std", action)
        for clipped in event.action_clipped:
            self._add("action.clip_rate", clipped)
        self._add("trigger.request_rate", event.trigger_requested)
        self._add("admission.admissible_rate", event.option_admissible)
        if event.trigger_requested:
            self._add("trigger.accept_rate", event.trigger_accepted)
            self._add("trigger.reject_rate", event.trigger_rejected)
        if event.trigger_accepted:
            self._add("admission.accepted_position_error_m", event.accepted_position_error_m or 0.0)
            self._add("admission.accepted_orientation_error_rad", event.accepted_orientation_error_rad or 0.0)
        failures = set(event.admission_failures)
        for reason in ADMISSION_FAILURE_REASONS:
            self._add(f"admission.failure.{reason}_rate", reason in failures)
        self._add("ready.strict_rate", event.strict_ready)
        self._add("contact.count", event.contact_count)
        self._add("contact.maximum_force_n", event.maximum_contact_force_n)
        self._add("contact.maximum_impulse_n_s", event.maximum_contact_impulse_n_s)
        self._add("object.tilt_rad", event.object_tilt_rad)
        self._add("object.maximum_displacement_m", event.object_displacement_m)

    def record_ppo_update(self, event: PPOUpdateEvent) -> None:
        self._validate_finite_event(event)
        if event.iteration < 0 or event.gradient_norm < 0:
            raise ValueError("PPO iteration and gradient norm must be non-negative")
        if not event.gradients_finite:
            diagnostic = TelemetryDiagnostic(
                DiagnosticKind.NONFINITE,
                "PPO reported non-finite gradients",
                context={"iteration": event.iteration},
            )
            self.record_diagnostic(DiagnosticEvent(event.iteration, diagnostic))
            raise NonFiniteTelemetryError(diagnostic.message)
        mapping = {
            "ppo.policy_loss": event.policy_loss,
            "ppo.value_loss": event.value_loss,
            "ppo.entropy": event.entropy,
            "ppo.kl": event.kl,
            "ppo.total_loss": event.total_loss,
            "ppo.gradient_norm": event.gradient_norm,
            "ppo.gradient_finite_rate": event.gradients_finite,
            "ppo.log_probability": event.log_probability,
        }
        for name, value in mapping.items():
            self._add(name, value)

    def record_episode(self, event: EpisodeEvent) -> None:
        self._validate_finite_event(event)
        if event.iteration < 0 or event.env_id < 0 or event.episode_length_steps < 1:
            raise ValueError("episode identifiers must be non-negative and length must be positive")
        if event.stage_at_reset not in CURRICULUM_STAGES:
            raise ValueError(f"unknown stage_at_reset {event.stage_at_reset!r}")
        if event.complete_success and event.dangerous:
            raise ValueError("complete success and dangerous must be mutually exclusive")
        if event.primary_termination_reason is not None and event.primary_termination_reason not in TERMINATION_REASONS:
            raise ValueError(f"unknown primary termination reason {event.primary_termination_reason!r}")
        unknown_auxiliary = set(event.auxiliary_termination_reasons) - set(TERMINATION_REASONS)
        if unknown_auxiliary:
            raise ValueError(f"unknown auxiliary termination reason(s): {sorted(unknown_auxiliary)}")
        unknown_rewards = set(event.reward_totals) - set(REWARD_TERMS)
        if unknown_rewards:
            raise ValueError(f"unknown reward term(s): {sorted(unknown_rewards)}")
        unknown_randomization = set(event.randomization_samples) - set(RANDOMIZATION_FIELDS)
        if unknown_randomization:
            raise ValueError(f"unknown randomization field(s): {sorted(unknown_randomization)}")

        self._add("episode.length_steps", event.episode_length_steps)
        self._add("outcome.strict_ready_episode_rate", event.strict_ready_ever)
        self._add("outcome.lift_5cm_rate", event.lift_5cm)
        self._add("outcome.complete_success_rate", event.complete_success)
        self._add("outcome.dangerous_rate", event.dangerous)
        auxiliary = set(event.auxiliary_termination_reasons)
        for reason in TERMINATION_REASONS:
            self._add(f"termination.primary.{reason}_rate", event.primary_termination_reason == reason)
            self._add(f"termination.auxiliary.{reason}_rate", reason in auxiliary)
        for term, value in event.reward_totals.items():
            self._add(f"reward.{term}", value)
        for stage in CURRICULUM_STAGES:
            self._add(f"curriculum.episodes_at_reset.{stage}", stage == event.stage_at_reset)
        for name, value in event.randomization_samples.items():
            for reduction in (Reduction.MEAN, Reduction.STD, Reduction.MIN, Reduction.MAX):
                self._add(f"randomization.{name}.{reduction.value}", value)

    def record_window(self, event: CurriculumWindowEvent) -> None:
        self._validate_finite_event(event)
        if event.iteration < 0 or event.stage not in CURRICULUM_STAGES:
            raise ValueError("window iteration/stage is invalid")
        if event.target_episodes < 1 or not 0 <= event.completed_episodes <= event.target_episodes:
            raise ValueError("window episode progress is invalid")
        if event.consecutive_passing_windows < 0 or (event.promoted and event.stalled):
            raise ValueError("window promotion/stall state is invalid")
        self._add("curriculum.stage_index", CURRICULUM_STAGES.index(event.stage))
        self._add("curriculum.window_progress", event.completed_episodes / event.target_episodes)
        self._add("curriculum.promotion_rate", event.promoted)
        self._add("curriculum.stalled_rate", event.stalled)
        self._add("curriculum.consecutive_passing_windows", event.consecutive_passing_windows)

    def record_run(self, event: RunEvent) -> None:
        self._validate_finite_event(event)
        integer_values = (
            event.iteration,
            event.total_steps,
            event.memory_rss_bytes,
            event.num_envs,
            event.worker_threads,
            event.physics_steps,
            event.policy_steps,
        )
        float_values = (
            event.steps_per_second,
            event.sampling_seconds,
            event.learning_seconds,
            event.total_seconds,
            event.eta_seconds,
            event.cpu_percent,
            event.sim_steps_per_second,
            event.realtime_factor,
            event.fixed_torsional_friction,
            event.fixed_rolling_friction,
        )
        if min(integer_values) < 0 or min(float_values) < 0 or event.num_envs < 1 or event.worker_threads < 1:
            raise ValueError("run counts, durations, rates, and magnitudes must be non-negative")
        if event.worker_threads > event.num_envs:
            raise ValueError("worker_threads cannot exceed num_envs")
        if not event.randomization_version.strip():
            raise ValueError("randomization_version must not be empty")
        mapping = {
            "throughput.total_steps": event.total_steps,
            "throughput.steps_per_second": event.steps_per_second,
            "time.sampling_seconds": event.sampling_seconds,
            "time.learning_seconds": event.learning_seconds,
            "time.total_seconds": event.total_seconds,
            "time.eta_seconds": event.eta_seconds,
            "system.memory_rss_bytes": event.memory_rss_bytes,
            "system.cpu_percent": event.cpu_percent,
            "system.num_envs": event.num_envs,
            "system.worker_threads": event.worker_threads,
            "system.physics_steps": event.physics_steps,
            "system.policy_steps": event.policy_steps,
            "system.sim_steps_per_second": event.sim_steps_per_second,
            "system.realtime_factor": event.realtime_factor,
            "randomization.fixed_torsional_friction": event.fixed_torsional_friction,
            "randomization.fixed_rolling_friction": event.fixed_rolling_friction,
        }
        for name, value in mapping.items():
            self._add(name, value)
        self._randomization_version = event.randomization_version
        self._bottleneck = event.bottleneck.strip() or None

    def record_diagnostic(self, event: DiagnosticEvent) -> None:
        if event.iteration < 0:
            raise ValueError("diagnostic iteration must be non-negative")
        diagnostic = event.diagnostic
        self._diagnostics.append(diagnostic)
        self._add(f"warning.{diagnostic.kind.value}.count", diagnostic.count)

    def ingest(self, event: TelemetryEvent) -> None:
        """Single typed integration point for later production modules."""

        if isinstance(event, StepEvent):
            self.record_step(event)
        elif isinstance(event, PPOUpdateEvent):
            self.record_ppo_update(event)
        elif isinstance(event, EpisodeEvent):
            self.record_episode(event)
        elif isinstance(event, CurriculumWindowEvent):
            self.record_window(event)
        elif isinstance(event, RunEvent):
            self.record_run(event)
        elif isinstance(event, DiagnosticEvent):
            self.record_diagnostic(event)
        else:  # pragma: no cover - TypeAlias protects typed callers
            raise TypeError(f"unsupported telemetry event {type(event).__name__}")

    def snapshot(self, iteration: int) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            iteration=iteration,
            metrics=MappingProxyType({name: accumulator.result() for name, accumulator in self._values.items()}),
            diagnostics=tuple(self._diagnostics),
            randomization_version=self._randomization_version,
            bottleneck=self._bottleneck,
        )

    def state_dict(self) -> dict[str, object]:
        """Return the complete finite accumulator state for checkpoint resume."""

        values: dict[str, object] = {}
        for name, accumulator in self._values.items():
            values[name] = {
                "count": accumulator.count,
                "total": accumulator.total,
                "total_squared": accumulator.total_squared,
                "minimum": None if accumulator.count == 0 else accumulator.minimum,
                "maximum": None if accumulator.count == 0 else accumulator.maximum,
                "last": accumulator.last,
            }
        diagnostics = [
            {
                "kind": item.kind.value,
                "message": item.message,
                "count": item.count,
                "env_id": item.env_id,
                "policy_step": item.policy_step,
                "config_fingerprint": item.config_fingerprint,
                "context": dict(item.context),
            }
            for item in self._diagnostics
        ]
        return {
            "version": TELEMETRY_STATE_VERSION,
            "values": values,
            "diagnostics": diagnostics,
            "randomization_version": self._randomization_version,
            "bottleneck": self._bottleneck,
        }

    def load_state_dict(self, raw: Mapping[str, object]) -> None:
        """Validate fully, then atomically replace checkpointed aggregates."""

        required = {"version", "values", "diagnostics", "randomization_version", "bottleneck"}
        if set(raw) != required or raw["version"] != TELEMETRY_STATE_VERSION:
            raise ValueError("telemetry checkpoint fields/version are incomplete or unknown")
        values = raw["values"]
        diagnostics = raw["diagnostics"]
        if not isinstance(values, Mapping) or set(values) != set(METRIC_SCHEMA):
            raise ValueError("telemetry checkpoint metric schema mismatch")
        if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
            raise TypeError("telemetry checkpoint diagnostics must be a sequence")
        restored_values: dict[str, _MetricAccumulator] = {}
        metric_fields = {"count", "total", "total_squared", "minimum", "maximum", "last"}
        for name, spec in METRIC_SCHEMA.items():
            item = values[name]
            if not isinstance(item, Mapping) or set(item) != metric_fields:
                raise ValueError(f"telemetry checkpoint metric {name} fields mismatch")
            count = item["count"]
            if type(count) is not int or count < 0:
                raise ValueError(f"telemetry checkpoint metric {name} count is invalid")
            numbers: dict[str, float] = {}
            for field_name in ("total", "total_squared", "last"):
                value = item[field_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"telemetry checkpoint metric {name}.{field_name} is non-finite")
                numbers[field_name] = float(value)
            minimum = item["minimum"]
            maximum = item["maximum"]
            if count == 0:
                if minimum is not None or maximum is not None or any(numbers.values()):
                    raise ValueError(f"empty telemetry checkpoint metric {name} is not zeroed")
                minimum_value, maximum_value = math.inf, -math.inf
            else:
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    for value in (minimum, maximum)
                ):
                    raise ValueError(f"telemetry checkpoint metric {name} extrema are invalid")
                minimum_value, maximum_value = float(minimum), float(maximum)
                if minimum_value > maximum_value:
                    raise ValueError(f"telemetry checkpoint metric {name} extrema are reversed")
            accumulator = _MetricAccumulator(spec)
            accumulator.count = count
            accumulator.total = numbers["total"]
            accumulator.total_squared = numbers["total_squared"]
            accumulator.minimum = minimum_value
            accumulator.maximum = maximum_value
            accumulator.last = numbers["last"]
            # Materialize once to validate the restored reduction result.
            accumulator.result()
            restored_values[name] = accumulator
        restored_diagnostics: list[TelemetryDiagnostic] = []
        diagnostic_fields = {
            "kind",
            "message",
            "count",
            "env_id",
            "policy_step",
            "config_fingerprint",
            "context",
        }
        for item in diagnostics:
            if not isinstance(item, Mapping) or set(item) != diagnostic_fields:
                raise ValueError("telemetry checkpoint diagnostic fields mismatch")
            context = item["context"]
            if not isinstance(context, Mapping):
                raise TypeError("telemetry checkpoint diagnostic context must be a mapping")
            restored_diagnostics.append(
                TelemetryDiagnostic(
                    kind=DiagnosticKind(str(item["kind"])),
                    message=str(item["message"]),
                    count=int(item["count"]),
                    env_id=None if item["env_id"] is None else int(item["env_id"]),
                    policy_step=None if item["policy_step"] is None else int(item["policy_step"]),
                    config_fingerprint=(
                        None if item["config_fingerprint"] is None else str(item["config_fingerprint"])
                    ),
                    context=dict(context),
                )
            )
        randomization_version = raw["randomization_version"]
        bottleneck = raw["bottleneck"]
        if randomization_version is not None and not isinstance(randomization_version, str):
            raise TypeError("telemetry randomization version must be a string or null")
        if bottleneck is not None and not isinstance(bottleneck, str):
            raise TypeError("telemetry bottleneck must be a string or null")
        self._values = restored_values
        self._diagnostics = restored_diagnostics
        self._randomization_version = randomization_version
        self._bottleneck = bottleneck


__all__ = [
    "ADMISSION_FAILURE_REASONS",
    "CURRICULUM_STAGES",
    "METRIC_SCHEMA",
    "METRIC_SPECS",
    "RANDOMIZATION_FIELDS",
    "REWARD_TERMS",
    "TELEMETRY_STATE_VERSION",
    "TERMINATION_REASONS",
    "ConditionalAggregator",
    "CurriculumWindowEvent",
    "DiagnosticEvent",
    "DiagnosticKind",
    "EpisodeEvent",
    "MetricLevel",
    "MetricSpec",
    "MetricValue",
    "NonFiniteTelemetryError",
    "PPOUpdateEvent",
    "Reduction",
    "RunEvent",
    "StepEvent",
    "TelemetryAggregator",
    "TelemetryDiagnostic",
    "TelemetryEvent",
    "TelemetrySnapshot",
]
