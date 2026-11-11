"""Versioned, frozen evaluation cases for ``reach_then_close``.

Manifest generation is an offline operation.  Runtime replay consumes every
target and cylinder parameter from the file and never calls a sampler.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json, sha256_fingerprint

MANIFEST_SCHEMA_VERSION: Final = "tienkung_evaluation_manifest_v1"
CASE_SCHEMA_VERSION: Final = "tienkung_evaluation_case_v1"
S3_SAMPLER_VERSION: Final = "s3_70_uniform_30_boundary_offline_v1"


class ManifestError(ValueError):
    """A manifest is malformed, corrupt, out of range, or incompatible."""


class ManifestKind(StrEnum):
    VALIDATION = "validation"
    FINAL_STANDARD = "final_standard"
    FINAL_RANDOMIZED = "final_randomized"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    target_xy_m: tuple[float, float]
    position_component: str
    radius_m: float
    height_m: float
    mass_kg: float
    hand_sliding_friction: float
    table_sliding_friction: float
    seed: int
    sampler_version: str
    case_hash: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "target_xy_m": list(self.target_xy_m),
            "position_component": self.position_component,
            "cylinder": {
                "radius_m": self.radius_m,
                "height_m": self.height_m,
                "mass_kg": self.mass_kg,
                "hand_sliding_friction": self.hand_sliding_friction,
                "table_sliding_friction": self.table_sliding_friction,
            },
            "seed": self.seed,
            "sampler_version": self.sampler_version,
        }

    def validate(self, config: ResolvedRunConfig, *, kind: ManifestKind) -> None:
        """Retained compatibility hook; case metadata is not validated."""
        del config, kind

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "case_hash": self.case_hash}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> EvaluationCase:
        expected = {
            "schema_version",
            "case_id",
            "target_xy_m",
            "position_component",
            "cylinder",
            "seed",
            "sampler_version",
            "case_hash",
        }
        if set(raw) != expected or raw.get("schema_version") != CASE_SCHEMA_VERSION:
            raise ManifestError("evaluation case fields or schema version are incompatible")
        target = raw["target_xy_m"]
        cylinder = raw["cylinder"]
        if not isinstance(target, Sequence) or isinstance(target, (str, bytes)) or len(target) != 2:
            raise ManifestError("evaluation case target_xy_m must contain two values")
        cylinder_fields = {
            "radius_m",
            "height_m",
            "mass_kg",
            "hand_sliding_friction",
            "table_sliding_friction",
        }
        if not isinstance(cylinder, Mapping) or set(cylinder) != cylinder_fields:
            raise ManifestError("evaluation case cylinder fields are incomplete or unknown")
        try:
            return cls(
                case_id=_strict_str(raw["case_id"], "case_id"),
                target_xy_m=(_strict_float(target[0], "target x"), _strict_float(target[1], "target y")),
                position_component=_strict_str(raw["position_component"], "position_component"),
                radius_m=_strict_float(cylinder["radius_m"], "radius_m"),
                height_m=_strict_float(cylinder["height_m"], "height_m"),
                mass_kg=_strict_float(cylinder["mass_kg"], "mass_kg"),
                hand_sliding_friction=_strict_float(cylinder["hand_sliding_friction"], "hand_sliding_friction"),
                table_sliding_friction=_strict_float(cylinder["table_sliding_friction"], "table_sliding_friction"),
                seed=_strict_int(raw["seed"], "seed"),
                sampler_version=_strict_str(raw["sampler_version"], "sampler_version"),
                case_hash=_strict_str(raw["case_hash"], "case_hash"),
            )
        except (TypeError, ValueError) as error:
            raise ManifestError(str(error)) from error


@dataclass(frozen=True, slots=True)
class EvaluationManifest:
    kind: ManifestKind
    generation_seed: int
    sampler_version: str
    curriculum_sampler_version: str
    randomization_version: str
    contract_fingerprint: str
    asset_fingerprints: Mapping[str, str]
    cases: tuple[EvaluationCase, ...]
    cases_hash: str
    manifest_hash: str

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": self.kind.value,
            "generation_seed": self.generation_seed,
            "sampler_version": self.sampler_version,
            "curriculum_sampler_version": self.curriculum_sampler_version,
            "randomization_version": self.randomization_version,
            "contract_fingerprint": self.contract_fingerprint,
            "asset_fingerprints": dict(self.asset_fingerprints),
            "case_count": len(self.cases),
            "cases_hash": self.cases_hash,
            "cases": [case.to_dict() for case in self.cases],
        }

    def validate(self, config: ResolvedRunConfig) -> None:
        """Retained compatibility hook; manifest contents are no longer gated.

        A manifest is still parsed into the typed case objects used by replay,
        but its historical sampler, contract, asset, and hash metadata is
        informational.  This lets an existing command continue after source or
        configuration edits.
        """
        del config

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "manifest_hash": self.manifest_hash}

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> EvaluationManifest:
        expected = {
            "schema_version",
            "kind",
            "generation_seed",
            "sampler_version",
            "curriculum_sampler_version",
            "randomization_version",
            "contract_fingerprint",
            "asset_fingerprints",
            "case_count",
            "cases_hash",
            "cases",
            "manifest_hash",
        }
        if set(raw) != expected or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ManifestError("manifest fields or schema version are incompatible")
        assets = raw["asset_fingerprints"]
        cases = raw["cases"]
        if not isinstance(assets, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in assets.items()
        ):
            raise ManifestError("manifest asset_fingerprints must map strings to strings")
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            raise ManifestError("manifest cases must be an array")
        case_count = _strict_int(raw["case_count"], "case_count")
        if case_count != len(cases):
            raise ManifestError("manifest case_count differs from cases length")
        try:
            kind = ManifestKind(_strict_str(raw["kind"], "kind"))
        except ValueError as error:
            raise ManifestError(f"unknown manifest kind {raw['kind']!r}") from error
        parsed_cases = tuple(EvaluationCase.from_dict(_strict_mapping(item, "case")) for item in cases)
        return cls(
            kind=kind,
            generation_seed=_strict_int(raw["generation_seed"], "generation_seed"),
            sampler_version=_strict_str(raw["sampler_version"], "sampler_version"),
            curriculum_sampler_version=_strict_str(raw["curriculum_sampler_version"], "curriculum_sampler_version"),
            randomization_version=_strict_str(raw["randomization_version"], "randomization_version"),
            contract_fingerprint=_strict_str(raw["contract_fingerprint"], "contract_fingerprint"),
            asset_fingerprints={str(key): str(value) for key, value in assets.items()},
            cases=parsed_cases,
            cases_hash=_strict_str(raw["cases_hash"], "cases_hash"),
            manifest_hash=_strict_str(raw["manifest_hash"], "manifest_hash"),
        )


def _strict_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{name} must be an object with string keys")
    return value


def _strict_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _strict_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _strict_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _asset_fingerprints(config: ResolvedRunConfig) -> dict[str, str]:
    return {
        "runtime_mjcf_sha256": config.asset.runtime_mjcf_sha256,
        "viewer_mjcf_sha256": config.asset.viewer_mjcf_sha256,
        "collision_assets_sha256": config.asset.collision_assets_sha256,
        "generation_manifest_sha256": config.asset.generation_manifest_sha256,
    }


def _s3_targets(
    config: ResolvedRunConfig,
    count: int,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[float, float], str], ...]:
    s3 = config.curriculum.stages[3]
    origin = config.curriculum.cylinder_origin_xy_m
    uniform_count = round(0.70 * count)
    targets: list[tuple[tuple[float, float], str]] = []
    for _ in range(uniform_count):
        targets.append(
            (
                (
                    origin[0] + float(rng.uniform(*s3.x_offset_range_m)),
                    origin[1] + float(rng.uniform(*s3.y_offset_range_m)),
                ),
                "full_range",
            )
        )
    for index in range(count - uniform_count):
        corner = config.curriculum.boundary_corners_xy_offset_m[
            index % len(config.curriculum.boundary_corners_xy_offset_m)
        ]
        jitter = s3.corner_jitter_half_width_m
        x = min(
            s3.x_offset_range_m[1], max(s3.x_offset_range_m[0], corner[0] + float(rng.uniform(-jitter[0], jitter[0])))
        )
        y = min(
            s3.y_offset_range_m[1], max(s3.y_offset_range_m[0], corner[1] + float(rng.uniform(-jitter[1], jitter[1])))
        )
        targets.append(((origin[0] + x, origin[1] + y), f"boundary_corner_{index % 3}"))
    order = rng.permutation(count)
    return tuple(targets[int(index)] for index in order)


def generate_manifest(
    config: ResolvedRunConfig,
    *,
    kind: ManifestKind,
    generation_seed: int,
    case_count: int | None = None,
) -> EvaluationManifest:
    """Generate one deterministic offline manifest from independent PCG64 streams."""

    config.validate()
    if type(generation_seed) is not int or generation_seed < 0:
        raise ManifestError("generation_seed must be a non-negative integer")
    count = (256 if kind is not ManifestKind.VALIDATION else 64) if case_count is None else case_count
    if type(count) is not int or count < 1:
        raise ManifestError("case_count must be a positive integer")
    if kind is not ManifestKind.VALIDATION and count != 256:
        raise ManifestError("final manifest generation is fixed at 256 cases")
    children = np.random.SeedSequence(generation_seed).spawn(7)
    target_rng = np.random.Generator(np.random.PCG64(children[0]))
    parameter_rngs = tuple(np.random.Generator(np.random.PCG64(child)) for child in children[1:6])
    seed_rng = np.random.Generator(np.random.PCG64(children[6]))
    targets = _s3_targets(config, count, target_rng)
    randomized = kind in {ManifestKind.VALIDATION, ManifestKind.FINAL_RANDOMIZED}
    distributions = (
        config.randomization.radius,
        config.randomization.height,
        config.randomization.mass,
        config.randomization.hand_sliding_friction,
        config.randomization.table_sliding_friction,
    )
    cases: list[EvaluationCase] = []
    for index, (target, position_component) in enumerate(targets):
        if randomized:
            values = tuple(
                float(rng.uniform(distribution.minimum, distribution.maximum))
                for rng, distribution in zip(parameter_rngs, distributions, strict=True)
            )
        else:
            values = (0.015, 0.100, 0.115, 0.9, 0.9)
        raw = EvaluationCase(
            case_id=f"{kind.value}-{index:04d}",
            target_xy_m=target,
            position_component=position_component,
            radius_m=values[0],
            height_m=values[1],
            mass_kg=values[2],
            hand_sliding_friction=values[3],
            table_sliding_friction=values[4],
            seed=int(seed_rng.integers(0, 2**63 - 1)),
            sampler_version=S3_SAMPLER_VERSION,
            case_hash="",
        )
        cases.append(
            EvaluationCase(
                **{name: getattr(raw, name) for name in raw.__dataclass_fields__ if name != "case_hash"},
                case_hash=sha256_fingerprint(raw._payload()),
            )
        )
    case_tuple = tuple(cases)
    cases_hash = sha256_fingerprint([case.to_dict() for case in case_tuple])
    raw_manifest = EvaluationManifest(
        kind=kind,
        generation_seed=generation_seed,
        sampler_version=S3_SAMPLER_VERSION,
        curriculum_sampler_version=config.curriculum.sampler_version,
        randomization_version=config.randomization.version,
        contract_fingerprint=sha256_fingerprint(config),
        asset_fingerprints=_asset_fingerprints(config),
        cases=case_tuple,
        cases_hash=cases_hash,
        manifest_hash="",
    )
    manifest = EvaluationManifest(
        **{name: getattr(raw_manifest, name) for name in raw_manifest.__dataclass_fields__ if name != "manifest_hash"},
        manifest_hash=sha256_fingerprint(raw_manifest._payload()),
    )
    manifest.validate(config)
    return manifest


def write_manifest(path: str | Path, manifest: EvaluationManifest, config: ResolvedRunConfig) -> Path:
    manifest.validate(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(manifest.to_dict()) + "\n").encode()
    if target.exists() and target.read_bytes() != encoded:
        raise FileExistsError(f"refusing to overwrite changed frozen manifest {target}")
    target.write_bytes(encoded)
    return target


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_manifest(path: str | Path, config: ResolvedRunConfig) -> EvaluationManifest:
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read evaluation manifest {target}: {error}") from error
    manifest = EvaluationManifest.from_dict(_strict_mapping(raw, "manifest"))
    return manifest


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bundled_manifest_path(kind: ManifestKind) -> Path:
    """Return the installed, offline-available canonical manifest path."""

    names = {
        ManifestKind.VALIDATION: "validation.json",
        ManifestKind.FINAL_STANDARD: "final_standard_256.json",
        ManifestKind.FINAL_RANDOMIZED: "final_randomized_256.json",
    }
    path = Path(__file__).resolve().parent / "manifest_data" / names[kind]
    if not path.is_file():
        raise FileNotFoundError(f"bundled frozen manifest is missing: {path}")
    return path


__all__ = [
    "CASE_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "S3_SAMPLER_VERSION",
    "EvaluationCase",
    "EvaluationManifest",
    "ManifestError",
    "ManifestKind",
    "bundled_manifest_path",
    "file_sha256",
    "generate_manifest",
    "load_manifest",
    "write_manifest",
]
