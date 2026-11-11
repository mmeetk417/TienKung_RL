"""Configuration-driven S0--S3 target sampling and monotonic curriculum state.

This module deliberately contains no simulator calls.  Episode ownership is
captured by ``stage_at_reset`` and only completed episode summaries can mutate
the promotion window.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol, cast

import numpy as np

from tienkung_rl.config.models import CurriculumConfig, CurriculumStageConfig, PromotionGateConfig, ResolvedRunConfig
from tienkung_rl.config.serialization import sha256_fingerprint

from .randomization import CurriculumPlacement


class CurriculumError(ValueError):
    """A sample, episode result, or restored state violates the contract."""


class CurriculumStage(StrEnum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


_STAGES = tuple(CurriculumStage)


class ForcedTargetAuthority(StrEnum):
    """Workflows permitted to override a normally sampled target."""

    VIEWER = "viewer"
    ASSET_DIAGNOSTIC = "asset_diagnostic"
    SCRIPT_DIAGNOSTIC = "script_diagnostic"
    FROZEN_MANIFEST = "frozen_manifest"


class UnitRandomSource(Protocol):
    def random(self) -> float: ...


def _unit(source: UnitRandomSource) -> float:
    value = float(source.random())
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise CurriculumError(f"unit random source returned {value!r} outside [0, 1]")
    return value


def _uniform(source: UnitRandomSource, bounds: tuple[float, float]) -> float:
    value = bounds[0] + _unit(source) * (bounds[1] - bounds[0])
    return min(bounds[1], max(bounds[0], value))


def _clamp_closed_range(
    value: float,
    bounds: tuple[float, float],
    *,
    world_value: float,
    origin_value: float,
) -> float:
    """Accept closed-range endpoints despite subtraction-sized roundoff."""

    tolerance = 8.0 * max(
        math.ulp(world_value),
        math.ulp(origin_value),
        math.ulp(bounds[0]),
        math.ulp(bounds[1]),
    )
    if value < bounds[0] - tolerance or value > bounds[1] + tolerance:
        raise CurriculumError("forced XY lies outside the allowed range")
    return min(bounds[1], max(bounds[0], value))


def _stage(stage: CurriculumStage | str) -> CurriculumStage:
    try:
        return CurriculumStage(stage)
    except ValueError as error:
        raise CurriculumError(f"unknown curriculum stage {stage!r}") from error


def _stage_config(config: CurriculumConfig, stage: CurriculumStage) -> CurriculumStageConfig:
    for candidate in config.stages:
        if candidate.name == stage.value:
            return candidate
    raise CurriculumError(f"resolved configuration is missing stage {stage.value}")


def _sample_range(
    source: UnitRandomSource,
    config: CurriculumConfig,
    stage: CurriculumStage,
) -> tuple[float, float]:
    stage_config = _stage_config(config, stage)
    return (
        _uniform(source, stage_config.x_offset_range_m),
        _uniform(source, stage_config.y_offset_range_m),
    )


def _sample_boundary(
    source: UnitRandomSource,
    config: CurriculumConfig,
    stage_config: CurriculumStageConfig,
) -> tuple[tuple[float, float], str]:
    choice = min(
        len(config.boundary_corners_xy_offset_m) - 1,
        int(_unit(source) * len(config.boundary_corners_xy_offset_m)),
    )
    corner = config.boundary_corners_xy_offset_m[choice]
    jitter_x = (2.0 * _unit(source) - 1.0) * stage_config.corner_jitter_half_width_m[0]
    jitter_y = (2.0 * _unit(source) - 1.0) * stage_config.corner_jitter_half_width_m[1]
    x = min(stage_config.x_offset_range_m[1], max(stage_config.x_offset_range_m[0], corner[0] + jitter_x))
    y = min(stage_config.y_offset_range_m[1], max(stage_config.y_offset_range_m[0], corner[1] + jitter_y))
    return (x, y), f"boundary_corner_{choice}"


def sample_curriculum_target(
    config: CurriculumConfig,
    stage: CurriculumStage | str,
    source: UnitRandomSource,
    *,
    forced_xy_m: tuple[float, float] | None = None,
    authority: ForcedTargetAuthority | None = None,
) -> CurriculumPlacement:
    """Sample one target, then optionally apply an authorized finite override."""

    config.validate()
    selected_stage = _stage(stage)
    stage_config = _stage_config(config, selected_stage)
    component_draw = _unit(source)
    if stage_config.current_fraction == 1.0 or component_draw < stage_config.current_fraction:
        offset = _sample_range(source, config, selected_stage)
        component = "current_range"
    elif stage_config.boundary_fraction == 0.0 or component_draw < (
        stage_config.current_fraction + stage_config.replay_fraction
    ):
        replay_stage = _stage(stage_config.replay_source)
        offset = _sample_range(source, config, replay_stage)
        component = f"replay_{replay_stage.value.lower()}_range"
    else:
        if stage_config.boundary_fraction <= 0.0:
            raise CurriculumError(f"stage {selected_stage.value} selected an unconfigured boundary component")
        offset, component = _sample_boundary(source, config, stage_config)

    if forced_xy_m is not None:
        if not isinstance(authority, ForcedTargetAuthority):
            raise CurriculumError("forced XY requires viewer, diagnostic, or frozen-manifest authority")
        if len(forced_xy_m) != 2 or not all(math.isfinite(value) for value in forced_xy_m):
            raise CurriculumError("forced XY must contain two finite values")
        origin = config.cylinder_origin_xy_m
        raw_offset = (forced_xy_m[0] - origin[0], forced_xy_m[1] - origin[1])
        try:
            forced_offset = (
                _clamp_closed_range(
                    raw_offset[0],
                    stage_config.x_offset_range_m,
                    world_value=forced_xy_m[0],
                    origin_value=origin[0],
                ),
                _clamp_closed_range(
                    raw_offset[1],
                    stage_config.y_offset_range_m,
                    world_value=forced_xy_m[1],
                    origin_value=origin[1],
                ),
            )
        except CurriculumError as error:
            raise CurriculumError(f"forced XY lies outside the allowed {selected_stage.value} range") from error
        offset = forced_offset
        component = f"forced_{authority.value}_after_{component}"
    elif authority is not None:
        raise CurriculumError("forced target authority is invalid without forced XY")

    return CurriculumPlacement(
        stage_at_reset=selected_stage.value,
        target_xy_m=(config.cylinder_origin_xy_m[0] + offset[0], config.cylinder_origin_xy_m[1] + offset[1]),
        component_label=component,
    )


@dataclass(frozen=True, slots=True)
class EpisodeCurriculumResult:
    """Promotion-relevant facts from exactly one completed episode."""

    stage_at_reset: CurriculumStage | str
    option_accepted: bool
    lifted_5cm: bool
    complete_success: bool
    dangerous: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_at_reset", _stage(self.stage_at_reset))
        for name in ("option_accepted", "lifted_5cm", "complete_success", "dangerous"):
            if type(getattr(self, name)) is not bool:
                raise CurriculumError(f"episode {name} must be bool")


@dataclass(frozen=True, slots=True)
class WindowCounts:
    completed_episodes: int = 0
    option_accepted: int = 0
    lifted_5cm: int = 0
    complete_success: int = 0
    dangerous: int = 0

    def validate(self) -> None:
        values = asdict(self)
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise CurriculumError("curriculum window counts must be non-negative integers")
        if any(value > self.completed_episodes for name, value in values.items() if name != "completed_episodes"):
            raise CurriculumError("curriculum event counts cannot exceed completed episodes")

    def add(self, result: EpisodeCurriculumResult) -> WindowCounts:
        return WindowCounts(
            completed_episodes=self.completed_episodes + 1,
            option_accepted=self.option_accepted + int(result.option_accepted),
            lifted_5cm=self.lifted_5cm + int(result.lifted_5cm),
            complete_success=self.complete_success + int(result.complete_success),
            dangerous=self.dangerous + int(result.dangerous),
        )


@dataclass(frozen=True, slots=True)
class WindowRates:
    option_accepted: float
    lifted_5cm: float
    complete_success: float
    dangerous: float


@dataclass(frozen=True, slots=True)
class PromotionDiagnostics:
    stage: CurriculumStage
    completed_episodes: int
    target_episodes: int
    progress: float
    rates: WindowRates
    threshold_gaps: WindowRates
    complete_window: bool
    passed: bool
    stalled: bool
    counted: bool
    ignored_old_stage: bool
    promoted: bool
    consecutive_passing_windows: int


@dataclass(frozen=True, slots=True)
class CurriculumCheckpointState:
    curriculum_version: str
    sampler_version: str
    combined_fingerprint: str
    stage: CurriculumStage
    window: WindowCounts
    consecutive_passing_windows: int
    completed_windows: int
    strict_ready_events: int
    samples_drawn: int
    rng_state_json: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", _stage(self.stage))
        if not isinstance(self.window, WindowCounts):
            raise CurriculumError("curriculum checkpoint window must be WindowCounts")
        self.window.validate()
        for name in (
            "consecutive_passing_windows",
            "completed_windows",
            "strict_ready_events",
            "samples_drawn",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise CurriculumError(f"curriculum state {name} must be a non-negative integer")
        for name in ("curriculum_version", "sampler_version", "combined_fingerprint", "rng_state_json"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise CurriculumError(f"curriculum state {name} must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "curriculum_version": self.curriculum_version,
            "sampler_version": self.sampler_version,
            "combined_fingerprint": self.combined_fingerprint,
            "stage": self.stage.value,
            "window": asdict(self.window),
            "consecutive_passing_windows": self.consecutive_passing_windows,
            "completed_windows": self.completed_windows,
            "strict_ready_events": self.strict_ready_events,
            "samples_drawn": self.samples_drawn,
            "rng_state": json.loads(self.rng_state_json),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CurriculumCheckpointState:
        required = {
            "curriculum_version",
            "sampler_version",
            "combined_fingerprint",
            "stage",
            "window",
            "consecutive_passing_windows",
            "completed_windows",
            "strict_ready_events",
            "samples_drawn",
            "rng_state",
        }
        if set(raw) != required:
            raise CurriculumError("curriculum state fields are incomplete or unknown")
        window_raw = raw["window"]
        if not isinstance(window_raw, Mapping):
            raise CurriculumError("curriculum window state must be a mapping")
        window_fields = {"completed_episodes", "option_accepted", "lifted_5cm", "complete_success", "dangerous"}
        if set(window_raw) != window_fields:
            raise CurriculumError("curriculum window fields are incomplete or unknown")

        def integer(name: str, value: object) -> int:
            if type(value) is not int or value < 0:
                raise CurriculumError(f"curriculum state {name} must be a non-negative integer")
            return value

        window = WindowCounts(**{name: integer(name, window_raw[name]) for name in window_fields})
        window.validate()
        strings = ("curriculum_version", "sampler_version", "combined_fingerprint")
        if any(type(raw[name]) is not str or not cast(str, raw[name]) for name in strings):
            raise CurriculumError("curriculum state versions and fingerprint must be non-empty strings")
        rng_state_json = json.dumps(raw["rng_state"], sort_keys=True, separators=(",", ":"))
        return cls(
            curriculum_version=cast(str, raw["curriculum_version"]),
            sampler_version=cast(str, raw["sampler_version"]),
            combined_fingerprint=cast(str, raw["combined_fingerprint"]),
            stage=_stage(cast(str, raw["stage"])),
            window=window,
            consecutive_passing_windows=integer("consecutive_passing_windows", raw["consecutive_passing_windows"]),
            completed_windows=integer("completed_windows", raw["completed_windows"]),
            strict_ready_events=integer("strict_ready_events", raw["strict_ready_events"]),
            samples_drawn=integer("samples_drawn", raw["samples_drawn"]),
            rng_state_json=rng_state_json,
        )


class CurriculumManager:
    """Own target RNG and promotion windows for one training run."""

    def __init__(self, config: ResolvedRunConfig, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise CurriculumError("curriculum seed must be a non-negative integer")
        config.validate()
        self._config = config.curriculum
        self._fingerprint = sha256_fingerprint(config)
        self._stage = _stage(config.curriculum.initial_stage)
        self._window = WindowCounts()
        self._consecutive = 0
        self._completed_windows = 0
        self._strict_ready_events = 0
        self._samples_drawn = 0
        self._rng = np.random.Generator(np.random.PCG64(seed))

    @property
    def stage(self) -> CurriculumStage:
        return self._stage

    @property
    def window(self) -> WindowCounts:
        return self._window

    @property
    def consecutive_passing_windows(self) -> int:
        return self._consecutive

    @property
    def strict_ready_events(self) -> int:
        return self._strict_ready_events

    def sample_target(
        self,
        *,
        forced_xy_m: tuple[float, float] | None = None,
        authority: ForcedTargetAuthority | None = None,
    ) -> CurriculumPlacement:
        placement = sample_curriculum_target(
            self._config,
            self._stage,
            self._rng,
            forced_xy_m=forced_xy_m,
            authority=authority,
        )
        self._samples_drawn += 1
        return placement

    def record_strict_ready(self) -> None:
        """Record a diagnostic event without touching any promotion statistic."""

        self._strict_ready_events += 1

    def _gate(self) -> PromotionGateConfig | None:
        for gate in self._config.promotion_gates:
            if gate.from_stage == self._stage.value:
                return gate
        return None

    @staticmethod
    def _rates(window: WindowCounts) -> WindowRates:
        count = window.completed_episodes
        if count == 0:
            return WindowRates(0.0, 0.0, 0.0, 0.0)
        return WindowRates(
            option_accepted=window.option_accepted / count,
            lifted_5cm=window.lifted_5cm / count,
            complete_success=window.complete_success / count,
            dangerous=window.dangerous / count,
        )

    def _diagnostics(
        self,
        *,
        window: WindowCounts,
        counted: bool,
        ignored_old_stage: bool,
        complete: bool,
        passed: bool,
        stalled: bool,
        promoted: bool,
        consecutive: int,
    ) -> PromotionDiagnostics:
        gate = self._gate()
        target = gate.window_episodes if gate is not None else 0
        rates = self._rates(window)
        gaps = WindowRates(
            option_accepted=0.0 if gate is None else max(0.0, gate.option_accepted_rate_min - rates.option_accepted),
            lifted_5cm=0.0 if gate is None else max(0.0, gate.lift_5cm_rate_min - rates.lifted_5cm),
            complete_success=0.0 if gate is None else max(0.0, gate.complete_success_rate_min - rates.complete_success),
            dangerous=0.0 if gate is None else max(0.0, rates.dangerous - gate.dangerous_rate_max),
        )
        return PromotionDiagnostics(
            stage=self._stage,
            completed_episodes=window.completed_episodes,
            target_episodes=target,
            progress=0.0 if target == 0 else min(1.0, window.completed_episodes / target),
            rates=rates,
            threshold_gaps=gaps,
            complete_window=complete,
            passed=passed,
            stalled=stalled,
            counted=counted,
            ignored_old_stage=ignored_old_stage,
            promoted=promoted,
            consecutive_passing_windows=consecutive,
        )

    def record_completed_episode(self, result: EpisodeCurriculumResult) -> PromotionDiagnostics:
        result_stage = cast(CurriculumStage, result.stage_at_reset)
        if _STAGES.index(result_stage) > _STAGES.index(self._stage):
            raise CurriculumError("episode stage_at_reset is ahead of the global curriculum stage")
        if result_stage != self._stage:
            return self._diagnostics(
                window=self._window,
                counted=False,
                ignored_old_stage=True,
                complete=False,
                passed=False,
                stalled=False,
                promoted=False,
                consecutive=self._consecutive,
            )
        gate = self._gate()
        if gate is None:
            return self._diagnostics(
                window=self._window,
                counted=False,
                ignored_old_stage=False,
                complete=False,
                passed=False,
                stalled=False,
                promoted=False,
                consecutive=0,
            )

        completed = self._window.add(result)
        if completed.completed_episodes > gate.window_episodes:
            raise CurriculumError("curriculum window exceeded its configured size")
        if completed.completed_episodes < gate.window_episodes:
            self._window = completed
            return self._diagnostics(
                window=completed,
                counted=True,
                ignored_old_stage=False,
                complete=False,
                passed=False,
                stalled=False,
                promoted=False,
                consecutive=self._consecutive,
            )

        rates = self._rates(completed)
        passed = (
            rates.option_accepted >= gate.option_accepted_rate_min
            and rates.lifted_5cm >= gate.lift_5cm_rate_min
            and rates.complete_success >= gate.complete_success_rate_min
            and rates.dangerous <= gate.dangerous_rate_max
        )
        self._completed_windows += 1
        self._consecutive = self._consecutive + 1 if passed else 0
        promoted = self._consecutive >= self._config.consecutive_passing_windows
        completed_consecutive = self._consecutive
        diagnostics = self._diagnostics(
            window=completed,
            counted=True,
            ignored_old_stage=False,
            complete=True,
            passed=passed,
            stalled=not passed,
            promoted=promoted,
            consecutive=completed_consecutive,
        )
        self._window = WindowCounts()
        if promoted:
            self._stage = _STAGES[_STAGES.index(self._stage) + 1]
            self._consecutive = 0
        return diagnostics

    def diagnostics(self) -> PromotionDiagnostics:
        return self._diagnostics(
            window=self._window,
            counted=False,
            ignored_old_stage=False,
            complete=False,
            passed=False,
            stalled=False,
            promoted=False,
            consecutive=self._consecutive,
        )

    def state(self) -> CurriculumCheckpointState:
        rng_state = json.dumps(self._rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
        return CurriculumCheckpointState(
            curriculum_version=self._config.version,
            sampler_version=self._config.sampler_version,
            combined_fingerprint=self._fingerprint,
            stage=self._stage,
            window=self._window,
            consecutive_passing_windows=self._consecutive,
            completed_windows=self._completed_windows,
            strict_ready_events=self._strict_ready_events,
            samples_drawn=self._samples_drawn,
            rng_state_json=rng_state,
        )

    def restore(self, state: CurriculumCheckpointState | Mapping[str, object]) -> None:
        restored = state if isinstance(state, CurriculumCheckpointState) else CurriculumCheckpointState.from_dict(state)
        if restored.curriculum_version != self._config.version:
            raise CurriculumError("curriculum version mismatch while restoring state")
        if restored.sampler_version != self._config.sampler_version:
            raise CurriculumError("curriculum sampler version mismatch while restoring state")
        if _STAGES.index(restored.stage) < _STAGES.index(self._stage):
            raise CurriculumError("curriculum restore cannot downgrade the current stage")
        restored.window.validate()
        gate = next(
            (candidate for candidate in self._config.promotion_gates if candidate.from_stage == restored.stage.value),
            None,
        )
        if gate is None:
            if restored.window != WindowCounts() or restored.consecutive_passing_windows != 0:
                raise CurriculumError("terminal S3 state cannot contain a promotion window")
        else:
            if restored.window.completed_episodes >= gate.window_episodes:
                raise CurriculumError("restored curriculum window must be incomplete")
            if restored.consecutive_passing_windows >= self._config.consecutive_passing_windows:
                raise CurriculumError("restored passing-window count should already have promoted")
        generator = np.random.Generator(np.random.PCG64())
        try:
            raw_rng_state = json.loads(restored.rng_state_json)
            if not isinstance(raw_rng_state, Mapping):
                raise TypeError
            generator.bit_generator.state = cast(Mapping[str, object], raw_rng_state)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise CurriculumError("curriculum RNG state is incompatible with PCG64") from error
        self._stage = restored.stage
        self._window = restored.window
        self._consecutive = restored.consecutive_passing_windows
        self._completed_windows = restored.completed_windows
        self._strict_ready_events = restored.strict_ready_events
        self._samples_drawn = restored.samples_drawn
        self._rng = generator


__all__ = [
    "CurriculumCheckpointState",
    "CurriculumError",
    "CurriculumManager",
    "CurriculumStage",
    "EpisodeCurriculumResult",
    "ForcedTargetAuthority",
    "PromotionDiagnostics",
    "WindowCounts",
    "WindowRates",
    "sample_curriculum_target",
]
