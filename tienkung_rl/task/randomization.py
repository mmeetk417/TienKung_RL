"""Per-episode cylinder randomization and ordered reset orchestration."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar, cast

import numpy as np

from tienkung_rl.config.models import RandomizationConfig, ResolvedRunConfig, UniformDistributionConfig
from tienkung_rl.sim.backend import EnvironmentBackend
from tienkung_rl.sim.state import CylinderParameters, PhysicsState, ResetSpec


class RandomizationError(ValueError):
    """A sample, forced override, RNG snapshot, or reset violates its contract."""


class ForcedParameterAuthority(StrEnum):
    """The only workflows authorized to replace sampled cylinder values."""

    VIEWER = "viewer"
    ASSET_DIAGNOSTIC = "asset_diagnostic"
    SCRIPT_DIAGNOSTIC = "script_diagnostic"
    FROZEN_MANIFEST = "frozen_manifest"


class UnitRandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True, slots=True)
class ForcedCylinderParameters:
    """Optional post-sampling values; omitted fields retain sampled values."""

    radius_m: float | None = None
    height_m: float | None = None
    mass_kg: float | None = None
    hand_sliding_friction: float | None = None
    table_sliding_friction: float | None = None


@dataclass(frozen=True, slots=True)
class CylinderRandomizationSample:
    """The sole episode-level representation consumed by downstream modules."""

    radius_m: float
    height_m: float
    mass_kg: float
    hand_sliding_friction: float
    table_sliding_friction: float
    torsional_friction: float
    rolling_friction: float
    distribution_version: str
    rng_position: int

    def __post_init__(self) -> None:
        values = (
            self.radius_m,
            self.height_m,
            self.mass_kg,
            self.hand_sliding_friction,
            self.table_sliding_friction,
            self.torsional_friction,
            self.rolling_friction,
        )
        if not all(math.isfinite(value) for value in values):
            raise RandomizationError("cylinder randomization sample must be finite")
        if min(self.radius_m, self.height_m, self.mass_kg) <= 0.0:
            raise RandomizationError("cylinder geometry and mass must be positive")
        if (
            min(
                self.hand_sliding_friction,
                self.table_sliding_friction,
                self.torsional_friction,
                self.rolling_friction,
            )
            < 0.0
        ):
            raise RandomizationError("cylinder friction must be non-negative")
        if not self.distribution_version:
            raise RandomizationError("distribution_version must not be empty")
        if self.rng_position < 0:
            raise RandomizationError("rng_position must be non-negative")

    @property
    def solid_inertia_kg_m2(self) -> tuple[float, float, float]:
        half_side_sq = 0.5 * self.radius_m**2
        transverse = self.mass_kg * (4.0 * half_side_sq + self.height_m**2) / 12.0
        axial = (2.0 / 3.0) * self.mass_kg * half_side_sq
        return (transverse, transverse, axial)

    def to_cylinder_parameters(self) -> CylinderParameters:
        return CylinderParameters(
            radius_m=self.radius_m,
            height_m=self.height_m,
            mass_kg=self.mass_kg,
            hand_sliding_friction=self.hand_sliding_friction,
            table_sliding_friction=self.table_sliding_friction,
            torsional_friction=self.torsional_friction,
            rolling_friction=self.rolling_friction,
        )

    def to_dict(self) -> dict[str, float | int | str | list[float]]:
        result: dict[str, float | int | str | list[float]] = dict(asdict(self))
        result["solid_inertia_kg_m2"] = list(self.solid_inertia_kg_m2)
        return result


_STREAM_NAMES = (
    "radius",
    "height",
    "mass",
    "hand_sliding_friction",
    "table_sliding_friction",
)


@dataclass(frozen=True, slots=True)
class RandomizationRngState:
    distribution_version: str
    samples_drawn: int
    stream_states_json: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.distribution_version:
            raise RandomizationError("RNG state distribution_version must not be empty")
        if self.samples_drawn < 0:
            raise RandomizationError("RNG samples_drawn must be non-negative")
        if len(self.stream_states_json) != len(_STREAM_NAMES):
            raise RandomizationError("RNG state must contain all five independent streams")
        for state in self.stream_states_json:
            try:
                value = json.loads(state)
            except json.JSONDecodeError as error:
                raise RandomizationError("RNG stream state is not valid JSON") from error
            if not isinstance(value, dict):
                raise RandomizationError("RNG stream state must decode to an object")

    def to_dict(self) -> dict[str, object]:
        return {
            "distribution_version": self.distribution_version,
            "samples_drawn": self.samples_drawn,
            "streams": {
                name: json.loads(state) for name, state in zip(_STREAM_NAMES, self.stream_states_json, strict=True)
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RandomizationRngState:
        """Strictly reconstruct one environment's five independent streams."""

        required = {"distribution_version", "samples_drawn", "streams"}
        if set(raw) != required:
            raise RandomizationError("RNG state fields are incomplete or unknown")
        version = raw["distribution_version"]
        samples = raw["samples_drawn"]
        streams = raw["streams"]
        if not isinstance(version, str) or not version:
            raise RandomizationError("RNG state distribution_version must not be empty")
        if type(samples) is not int or samples < 0:
            raise RandomizationError("RNG state samples_drawn must be a non-negative integer")
        if not isinstance(streams, Mapping) or set(streams) != set(_STREAM_NAMES):
            raise RandomizationError("RNG state must contain exactly the five named streams")
        encoded = tuple(json.dumps(streams[name], sort_keys=True, separators=(",", ":")) for name in _STREAM_NAMES)
        return cls(version, samples, encoded)


def sample_closed_uniform(source: UnitRandomSource, distribution: UniformDistributionConfig) -> float:
    """Map a unit sample to both closed endpoints without changing the law."""

    unit = float(source.random())
    if not math.isfinite(unit) or not 0.0 <= unit <= 1.0:
        raise RandomizationError(f"unit random source returned {unit!r} outside [0, 1]")
    value = distribution.minimum + unit * (distribution.maximum - distribution.minimum)
    return min(distribution.maximum, max(distribution.minimum, value))


def _validate_value(name: str, value: float, distribution: UniformDistributionConfig) -> None:
    if not math.isfinite(value) or not distribution.minimum <= value <= distribution.maximum:
        raise RandomizationError(
            f"forced {name}={value!r} lies outside [{distribution.minimum}, {distribution.maximum}]"
        )


class CylinderRandomizer:
    """Five independently seeded PCG64 streams with exact save/restore."""

    def __init__(self, config: RandomizationConfig, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise RandomizationError("randomization seed must be a non-negative integer")
        config.validate()
        self._config = config
        children = np.random.SeedSequence(seed).spawn(len(_STREAM_NAMES))
        self._streams = tuple(np.random.Generator(np.random.PCG64(child)) for child in children)
        self._samples_drawn = 0

    @property
    def samples_drawn(self) -> int:
        return self._samples_drawn

    def state(self) -> RandomizationRngState:
        states = tuple(
            json.dumps(stream.bit_generator.state, sort_keys=True, separators=(",", ":")) for stream in self._streams
        )
        return RandomizationRngState(self._config.version, self._samples_drawn, states)

    def restore(self, state: RandomizationRngState) -> None:
        if state.distribution_version != self._config.version:
            raise RandomizationError("RNG state distribution version differs from the resolved randomization contract")
        restored: list[np.random.Generator] = []
        for encoded in state.stream_states_json:
            generator = np.random.Generator(np.random.PCG64())
            raw_state = json.loads(encoded)
            try:
                generator.bit_generator.state = cast(Mapping[str, object], raw_state)
            except (TypeError, ValueError) as error:
                raise RandomizationError("RNG stream state is incompatible with PCG64") from error
            restored.append(generator)
        self._streams = tuple(restored)
        self._samples_drawn = state.samples_drawn

    def sample(
        self,
        *,
        forced: ForcedCylinderParameters | None = None,
        authority: ForcedParameterAuthority | None = None,
    ) -> CylinderRandomizationSample:
        if forced is not None and not isinstance(authority, ForcedParameterAuthority):
            raise RandomizationError(
                "forced cylinder parameters require viewer, diagnostic, or frozen-manifest authority"
            )
        cfg = self._config
        replacements: dict[str, float] = {}
        if forced is not None:
            for field_name, distribution in (
                ("radius_m", cfg.radius),
                ("height_m", cfg.height),
                ("mass_kg", cfg.mass),
                ("hand_sliding_friction", cfg.hand_sliding_friction),
                ("table_sliding_friction", cfg.table_sliding_friction),
            ):
                value = getattr(forced, field_name)
                if value is not None:
                    _validate_value(field_name, value, distribution)
                    replacements[field_name] = value
        position = self._samples_drawn
        values = tuple(
            sample_closed_uniform(stream, distribution)
            for stream, distribution in zip(
                self._streams,
                (
                    cfg.radius,
                    cfg.height,
                    cfg.mass,
                    cfg.hand_sliding_friction,
                    cfg.table_sliding_friction,
                ),
                strict=True,
            )
        )
        sample = CylinderRandomizationSample(
            radius_m=values[0],
            height_m=values[1],
            mass_kg=values[2],
            hand_sliding_friction=values[3],
            table_sliding_friction=values[4],
            torsional_friction=cfg.fixed_torsional_friction,
            rolling_friction=cfg.fixed_rolling_friction,
            distribution_version=cfg.version,
            rng_position=position,
        )
        self._samples_drawn += 1
        if forced is None:
            return sample
        return CylinderRandomizationSample(
            radius_m=replacements.get("radius_m", sample.radius_m),
            height_m=replacements.get("height_m", sample.height_m),
            mass_kg=replacements.get("mass_kg", sample.mass_kg),
            hand_sliding_friction=replacements.get("hand_sliding_friction", sample.hand_sliding_friction),
            table_sliding_friction=replacements.get("table_sliding_friction", sample.table_sliding_friction),
            torsional_friction=sample.torsional_friction,
            rolling_friction=sample.rolling_friction,
            distribution_version=sample.distribution_version,
            rng_position=sample.rng_position,
        )


@dataclass(frozen=True, slots=True)
class CurriculumPlacement:
    stage_at_reset: str
    target_xy_m: tuple[float, float]
    component_label: str

    def __post_init__(self) -> None:
        if not self.stage_at_reset or not self.component_label:
            raise RandomizationError("curriculum placement labels must not be empty")
        if len(self.target_xy_m) != 2 or not all(math.isfinite(value) for value in self.target_xy_m):
            raise RandomizationError("curriculum target XY must contain two finite values")


@dataclass(frozen=True, slots=True)
class EpisodeRandomizationParameters:
    """Canonical structure for checkpoint, metrics, manifest and reports."""

    stage_at_reset: str
    curriculum_component: str
    cylinder_position_m: tuple[float, float, float]
    grasp_target_position_m: tuple[float, float, float]
    cylinder: CylinderRandomizationSample
    rng_state_after: RandomizationRngState
    forced_authority: str | None
    reset_sequence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_at_reset": self.stage_at_reset,
            "curriculum_component": self.curriculum_component,
            "cylinder_position_m": list(self.cylinder_position_m),
            "grasp_target_position_m": list(self.grasp_target_position_m),
            "cylinder": self.cylinder.to_dict(),
            "rng_state_after": self.rng_state_after.to_dict(),
            "forced_authority": self.forced_authority,
            "reset_sequence": list(self.reset_sequence),
        }


ObservationT = TypeVar("ObservationT")


@dataclass(frozen=True, slots=True)
class EpisodeResetResult(Generic[ObservationT]):
    state: PhysicsState
    observation: ObservationT
    parameters: EpisodeRandomizationParameters


_RESET_SEQUENCE = (
    "curriculum_xy",
    "five_parameter_sample",
    "inertia_and_friction_application",
    "physics_reset",
    "transient_clear",
    "first_observation",
)


class EpisodeResetOrchestrator:
    """Perform the M06 reset transaction in its one permitted order."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        backend: EnvironmentBackend,
        randomizer: CylinderRandomizer,
    ) -> None:
        config.validate()
        self._config = config
        self._backend = backend
        self._randomizer = randomizer

    def reset(
        self,
        curriculum_sampler: Callable[[], CurriculumPlacement],
        clear_transients: Callable[[], None],
        build_first_observation: Callable[[PhysicsState, EpisodeRandomizationParameters], ObservationT],
        *,
        forced: ForcedCylinderParameters | None = None,
        authority: ForcedParameterAuthority | None = None,
    ) -> EpisodeResetResult[ObservationT]:
        placement = curriculum_sampler()
        if not isinstance(placement, CurriculumPlacement):
            raise TypeError("curriculum_sampler must return CurriculumPlacement")
        sample = self._randomizer.sample(forced=forced, authority=authority)
        table = self._config.task.scene
        cylinder_center_z = (
            table.table_center_m[2]
            + 0.5 * table.table_size_m[2]
            + table.cylinder_table_clearance_m
            + 0.5 * sample.height_m
        )
        cylinder_position = (
            placement.target_xy_m[0],
            placement.target_xy_m[1],
            cylinder_center_z,
        )
        state = self._backend.reset(
            ResetSpec(
                cylinder_position_m=cylinder_position,
                cylinder=sample.to_cylinder_parameters(),
            )
        )
        if state.cylinder_parameters != sample.to_cylinder_parameters() or not np.allclose(
            state.cylinder_position_m,
            np.asarray(cylinder_position, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RandomizationError("backend reset did not apply the sampled cylinder transaction")
        clear_transients()
        parameters = EpisodeRandomizationParameters(
            stage_at_reset=placement.stage_at_reset,
            curriculum_component=placement.component_label,
            cylinder_position_m=cylinder_position,
            grasp_target_position_m=cylinder_position,
            cylinder=sample,
            rng_state_after=self._randomizer.state(),
            forced_authority=None if authority is None else authority.value,
            reset_sequence=_RESET_SEQUENCE,
        )
        observation = build_first_observation(state, parameters)
        return EpisodeResetResult(state=state, observation=observation, parameters=parameters)


__all__ = [
    "CurriculumPlacement",
    "CylinderRandomizationSample",
    "CylinderRandomizer",
    "EpisodeRandomizationParameters",
    "EpisodeResetOrchestrator",
    "EpisodeResetResult",
    "ForcedCylinderParameters",
    "ForcedParameterAuthority",
    "RandomizationError",
    "RandomizationRngState",
    "sample_closed_uniform",
]
