"""Typed scene declaration and strict binding to the generated runtime asset.

This module deliberately uses XML/YAML and the simulator's already-resolved
name cache instead of importing MuJoCo.  It is therefore safe for the task
layer while still failing closed against the exact M02 runtime product.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import yaml  # type: ignore[import-untyped]

from tienkung_rl.config.models import MaterialConfig, ResolvedRunConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSET_PATCH_ROOT = REPOSITORY_ROOT / "tienkung_rl/assets/tienkung2_pro/patches"
CONTACT_RULES_PATH = ASSET_PATCH_ROOT / "contact_groups.yaml"
SITE_DECLARATION_PATH = ASSET_PATCH_ROOT / "sites.yaml"


class SceneContractError(RuntimeError):
    """The configured scene cannot be bound unambiguously to the runtime asset."""


class RuntimeNameCache(Protocol):
    """Narrow structural view exported by ``MujocoEnvironmentBackend``."""

    object_ids: Mapping[str, Mapping[str, int]]
    home_key_id: int


@runtime_checkable
class SceneBackend(Protocol):
    """Backend capabilities required only while binding a scene."""

    @property
    def name_cache(self) -> RuntimeNameCache: ...

    @property
    def runtime_path(self) -> Path: ...


@dataclass(frozen=True, slots=True)
class RobotInitialPose:
    keyframe_name: str
    fixed_base_body_name: str
    environment_origin_m: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TableDescription:
    body_name: str
    top_geom_name: str
    leg_geom_names: tuple[str, ...]
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]

    @property
    def top_surface_height_m(self) -> float:
        return self.center_m[2] + 0.5 * self.size_m[2]


@dataclass(frozen=True, slots=True)
class CylinderDescription:
    body_name: str
    free_joint_name: str
    collision_geom_name: str
    axis_site_name: str
    origin_xy_m: tuple[float, float]


@dataclass(frozen=True, slots=True)
class NamedTaskFrame:
    name: str
    body_name: str
    runtime_site_name: str | None
    runtime_site_position_m: tuple[float, float, float] | None
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ContactPhaseRule:
    allowed_group_pairs: tuple[tuple[str, str], ...]
    illegal_robot_groups: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContactRules:
    version: str
    groups: Mapping[str, tuple[str, tuple[str, ...]]]
    reach: ContactPhaseRule
    script: ContactPhaseRule


@dataclass(frozen=True, slots=True)
class SceneDescription:
    version: str
    robot: RobotInitialPose
    table: TableDescription
    cylinder: CylinderDescription
    task_frames: tuple[NamedTaskFrame, ...]
    contact_rules: ContactRules


@dataclass(frozen=True, slots=True)
class BoundScene:
    """Scene declaration proven to match one runtime model and PhysicsConfig."""

    description: SceneDescription
    runtime_mjcf_path: Path
    bound_object_ids: Mapping[str, Mapping[str, int]]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SceneContractError(f"{context} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _strings(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise SceneContractError(f"{context} must contain non-empty names")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise SceneContractError(f"{context} contains duplicate names")
    return result


def _float_tuple(value: object, length: int, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SceneContractError(f"{context} must contain {length} numbers")
    result: list[float] = []
    for item in value:
        if type(item) not in (int, float) or not math.isfinite(float(item)):
            raise SceneContractError(f"{context} must contain only finite numbers")
        result.append(float(item))
    return tuple(result)


def _load_yaml(path: Path, context: str) -> Mapping[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SceneContractError(f"cannot read {context} {path}: {error}") from error
    return _mapping(raw, context)


def _load_frames(path: Path) -> tuple[NamedTaskFrame, ...]:
    declaration = _load_yaml(path, "site declaration")
    if declaration.get("schema_version") != "tienkung2_pro_sites_v1":
        raise SceneContractError("unsupported task-frame declaration version")
    frames_value = declaration.get("frames")
    if not isinstance(frames_value, list):
        raise SceneContractError("site declaration frames must be a list")
    runtime_sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
    sites_value = declaration.get("sites")
    if not isinstance(sites_value, list):
        raise SceneContractError("site declaration sites must be a list")
    for index, raw_site in enumerate(sites_value):
        site = _mapping(raw_site, f"sites[{index}]")
        frame = site.get("frame")
        name = site.get("name")
        if frame is not None:
            if not isinstance(frame, str) or not isinstance(name, str):
                raise SceneContractError(f"sites[{index}] has an invalid frame binding")
            if frame in runtime_sites:
                raise SceneContractError(f"frame {frame!r} has multiple runtime sites")
            runtime_sites[frame] = (
                name,
                cast(
                    tuple[float, float, float],
                    _float_tuple(site.get("pos_m"), 3, f"sites[{index}].pos_m"),
                ),
            )
    frames: list[NamedTaskFrame] = []
    for index, raw_frame in enumerate(frames_value):
        frame = _mapping(raw_frame, f"frames[{index}]")
        name = frame.get("name")
        body = frame.get("body")
        if not isinstance(name, str) or not name or not isinstance(body, str) or not body:
            raise SceneContractError(f"frames[{index}] has invalid names")
        runtime_site = runtime_sites.get(name)
        frames.append(
            NamedTaskFrame(
                name=name,
                body_name=body,
                runtime_site_name=None if runtime_site is None else runtime_site[0],
                runtime_site_position_m=None if runtime_site is None else runtime_site[1],
                position_m=cast(
                    tuple[float, float, float],
                    _float_tuple(frame.get("pos_m"), 3, f"frames[{index}].pos_m"),
                ),
                quaternion_wxyz=cast(
                    tuple[float, float, float, float],
                    _float_tuple(frame.get("quaternion_wxyz"), 4, f"frames[{index}].quaternion_wxyz"),
                ),
            )
        )
    if len({item.name for item in frames}) != len(frames):
        raise SceneContractError("task-frame names must be unique")
    return tuple(frames)


def _phase_rule(value: object, groups: Mapping[str, tuple[str, tuple[str, ...]]], context: str) -> ContactPhaseRule:
    raw = _mapping(value, context)
    if set(raw) != {"allowed_pairs", "illegal_robot_groups"}:
        raise SceneContractError(f"{context} has missing or unknown fields")
    pairs_value = raw["allowed_pairs"]
    if not isinstance(pairs_value, list):
        raise SceneContractError(f"{context}.allowed_pairs must be a list")
    pairs: list[tuple[str, str]] = []
    for index, pair in enumerate(pairs_value):
        names = _strings(pair, f"{context}.allowed_pairs[{index}]")
        if len(names) != 2:
            raise SceneContractError(f"{context}.allowed_pairs[{index}] must contain two groups")
        if any(name not in groups for name in names):
            raise SceneContractError(f"{context}.allowed_pairs[{index}] names an unknown group")
        pairs.append((names[0], names[1]))
    illegal = _strings(raw["illegal_robot_groups"], f"{context}.illegal_robot_groups")
    if any(name not in groups or groups[name][0] != "geom" for name in illegal):
        raise SceneContractError(f"{context} names an unknown or non-geom illegal group")
    return ContactPhaseRule(tuple(pairs), illegal)


def _load_contact_rules(path: Path) -> ContactRules:
    declaration = _load_yaml(path, "contact declaration")
    version = declaration.get("schema_version")
    if version != "tienkung2_pro_contact_groups_v1":
        raise SceneContractError("unsupported contact declaration version")
    raw_groups = _mapping(declaration.get("groups"), "contact groups")
    groups: dict[str, tuple[str, tuple[str, ...]]] = {}
    for name, raw_group in raw_groups.items():
        group = _mapping(raw_group, f"contact group {name}")
        if set(group) != {"kind", "members"} or group["kind"] not in {"geom", "body", "collision_mask"}:
            raise SceneContractError(f"contact group {name!r} has an invalid schema")
        groups[name] = (group["kind"], _strings(group["members"], f"contact group {name}.members"))
    phase_rules = _mapping(declaration.get("phase_rules"), "phase_rules")
    if set(phase_rules) != {"reach", "script"}:
        raise SceneContractError("contact declaration must contain exactly reach and script rules")
    return ContactRules(
        version=version,
        groups=groups,
        reach=_phase_rule(phase_rules["reach"], groups, "phase_rules.reach"),
        script=_phase_rule(phase_rules["script"], groups, "phase_rules.script"),
    )


def create_scene_description(
    config: ResolvedRunConfig,
    *,
    contact_rules_path: Path = CONTACT_RULES_PATH,
    site_declaration_path: Path = SITE_DECLARATION_PATH,
) -> SceneDescription:
    """Create the complete typed scene solely from frozen configuration/declarations."""

    scene = config.task.scene
    return SceneDescription(
        version="reach_then_close_scene_v1",
        robot=RobotInitialPose(
            keyframe_name="robot_and_scene_home",
            fixed_base_body_name="pelvis",
            environment_origin_m=(0.0, 0.0, 0.0),
        ),
        table=TableDescription(
            body_name="table",
            top_geom_name="table_top_collision",
            leg_geom_names=tuple(f"table_leg_collision_{index}" for index in range(4)),
            center_m=scene.table_center_m,
            size_m=scene.table_size_m,
        ),
        cylinder=CylinderDescription(
            body_name="cylinder",
            free_joint_name="cylinder_free_joint",
            collision_geom_name="cylinder_collision",
            axis_site_name="cylinder_axis_site",
            origin_xy_m=scene.cylinder_origin_xy_m,
        ),
        task_frames=_load_frames(site_declaration_path),
        contact_rules=_load_contact_rules(contact_rules_path),
    )


def _xml_numbers(element: ET.Element, attribute: str, length: int, context: str) -> tuple[float, ...]:
    raw = element.get(attribute)
    if raw is None:
        raise SceneContractError(f"{context} is missing {attribute}")
    try:
        values = tuple(float(item) for item in raw.split())
    except ValueError as error:
        raise SceneContractError(f"{context}.{attribute} contains a non-number") from error
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise SceneContractError(f"{context}.{attribute} must contain {length} finite numbers")
    return values


def _close_tuple(actual: tuple[float, ...], expected: tuple[float, ...], context: str) -> None:
    if len(actual) != len(expected) or any(
        not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12) for left, right in zip(actual, expected, strict=True)
    ):
        raise SceneContractError(f"{context} differs from resolved configuration: {actual} != {expected}")


def _unique_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    matches = [item for item in root.iter(tag) if item.get("name") == name]
    if len(matches) != 1:
        raise SceneContractError(f"runtime must contain exactly one {tag} named {name!r}, got {len(matches)}")
    return matches[0]


def _validate_material_contact(
    geom: ET.Element,
    material: MaterialConfig,
    config: ResolvedRunConfig,
) -> None:
    context = f"geom {geom.get('name')!r}"
    if geom.get("material") != material.name:
        raise SceneContractError(f"{context} material differs from PhysicsConfig")
    _close_tuple(
        _xml_numbers(geom, "friction", 3, context),
        (material.sliding_friction, material.torsional_friction, material.rolling_friction),
        f"{context}.friction",
    )
    contact = (
        (config.physics.cylinder_contact or config.physics.contact)
        if material.name == config.physics.cylinder_material.name
        else config.physics.contact
    )
    # Mesh contacts use their convex surface directly, without a primitive margin.
    margin = 0.0 if geom.get("type") == "mesh" else contact.margin_m
    _close_tuple(_xml_numbers(geom, "margin", 1, context), (margin,), f"{context}.margin")
    _close_tuple(_xml_numbers(geom, "gap", 1, context), (contact.gap_m,), f"{context}.gap")
    _close_tuple(_xml_numbers(geom, "solimp", 3, context), contact.impedance, f"{context}.solimp")
    _close_tuple(
        _xml_numbers(geom, "solref", 2, context),
        (contact.reference_time_s, contact.reference_damping),
        f"{context}.solref",
    )


def _validate_xml(description: SceneDescription, config: ResolvedRunConfig, runtime_path: Path) -> None:
    try:
        root = ET.parse(runtime_path).getroot()
    except (OSError, ET.ParseError) as error:
        raise SceneContractError(f"cannot read runtime MJCF {runtime_path}: {error}") from error
    option = root.find("option")
    size = root.find("size")
    if option is None or size is None:
        raise SceneContractError("runtime MJCF is missing option or size")
    physics = config.physics
    expected_option = {
        "integrator": physics.solver.integrator,
        "solver": physics.solver.solver,
    }
    for attribute, expected in expected_option.items():
        actual = option.get(attribute)
        if actual is None or actual.casefold() != expected.casefold():
            raise SceneContractError(f"runtime option {attribute} differs from PhysicsConfig")
    for numeric_attribute, numeric_expected in (
        ("timestep", physics.physics_dt_s),
        ("iterations", float(physics.solver.position_iterations)),
        ("ls_iterations", float(physics.solver.line_search_iterations)),
        ("noslip_iterations", float(physics.solver.no_slip_iterations)),
        ("tolerance", physics.solver.tolerance),
    ):
        _close_tuple(
            _xml_numbers(option, numeric_attribute, 1, "option"),
            (numeric_expected,),
            f"option.{numeric_attribute}",
        )
    _close_tuple(_xml_numbers(option, "gravity", 3, "option"), physics.gravity_m_s2, "option.gravity")
    if size.get("nconmax") != str(physics.contact.maximum_contacts):
        raise SceneContractError("runtime contact capacity differs from PhysicsConfig")
    for material in (physics.robot_material, physics.table_material, physics.cylinder_material):
        restitution = _unique_named(root, "numeric", f"material.restitution.{material.name}")
        _close_tuple(
            _xml_numbers(restitution, "data", 1, f"material.restitution.{material.name}"),
            (material.restitution,),
            f"material.restitution.{material.name}",
        )

    table_top = _unique_named(root, "geom", description.table.top_geom_name)
    _close_tuple(_xml_numbers(table_top, "pos", 3, "table top"), description.table.center_m, "table center")
    _close_tuple(
        _xml_numbers(table_top, "size", 3, "table top"),
        tuple(0.5 * item for item in description.table.size_m),
        "table size",
    )
    cylinder_body = _unique_named(root, "body", description.cylinder.body_name)
    nominal_height = 0.5 * (config.randomization.height.minimum + config.randomization.height.maximum)
    expected_cylinder_center = (
        *description.cylinder.origin_xy_m,
        description.table.top_surface_height_m + config.task.scene.cylinder_table_clearance_m + 0.5 * nominal_height,
    )
    _close_tuple(
        _xml_numbers(cylinder_body, "pos", 3, "cylinder body"),
        expected_cylinder_center,
        "cylinder nominal center",
    )

    table_geoms = (description.table.top_geom_name, *description.table.leg_geom_names)
    for name in table_geoms:
        _validate_material_contact(_unique_named(root, "geom", name), physics.table_material, config)
    cylinder_geom = _unique_named(root, "geom", description.cylinder.collision_geom_name)
    _validate_material_contact(cylinder_geom, physics.cylinder_material, config)
    scene_geom_names = {*table_geoms, description.cylinder.collision_geom_name}
    for geom in root.iter("geom"):
        geom_name = geom.get("name")
        if geom_name is not None and geom_name not in scene_geom_names and geom.get("group") == "3":
            _validate_material_contact(geom, physics.robot_material, config)

    parent_by_child = {child: parent for parent in root.iter() for child in parent}
    for frame in description.task_frames:
        if frame.runtime_site_name is None or frame.runtime_site_position_m is None:
            continue
        site = _unique_named(root, "site", frame.runtime_site_name)
        _close_tuple(
            _xml_numbers(site, "pos", 3, frame.runtime_site_name),
            frame.runtime_site_position_m,
            f"{frame.runtime_site_name} position",
        )
        parent = parent_by_child.get(site)
        if parent is None or parent.tag != "frame" or parent.get("name") != frame.name:
            raise SceneContractError(
                f"runtime site {frame.runtime_site_name!r} is not bound through frame {frame.name!r}"
            )
        _close_tuple(_xml_numbers(parent, "pos", 3, frame.name), frame.position_m, f"frame {frame.name} position")
        _close_tuple(
            _xml_numbers(parent, "quat", 4, frame.name),
            frame.quaternion_wxyz,
            f"frame {frame.name} quaternion",
        )
    _unique_named(root, "key", description.robot.keyframe_name)


def _validate_names(description: SceneDescription, cache: RuntimeNameCache) -> None:
    required: dict[str, tuple[str, ...]] = {
        "bodies": (
            description.robot.fixed_base_body_name,
            description.table.body_name,
            description.cylinder.body_name,
            *(frame.body_name for frame in description.task_frames),
        ),
        "joints": (description.cylinder.free_joint_name,),
        "geoms": (
            description.table.top_geom_name,
            *description.table.leg_geom_names,
            description.cylinder.collision_geom_name,
        ),
        "sites": (
            description.cylinder.axis_site_name,
            *(frame.runtime_site_name for frame in description.task_frames if frame.runtime_site_name is not None),
        ),
        "keys": (description.robot.keyframe_name,),
    }
    for group_kind, members in description.contact_rules.groups.values():
        namespace = {"geom": "geoms", "body": "bodies"}.get(group_kind)
        if namespace is not None:
            required[namespace] = (*required.get(namespace, ()), *members)
    for namespace, names in required.items():
        available = cache.object_ids.get(namespace)
        if available is None:
            raise SceneContractError(f"runtime name cache has no {namespace} namespace")
        missing = tuple(name for name in dict.fromkeys(names) if name not in available)
        if missing:
            raise SceneContractError(f"runtime is missing declared {namespace}: {missing}")


def bind_scene(
    description: SceneDescription,
    config: ResolvedRunConfig,
    backend: object,
    *,
    runtime_mjcf_path: Path | None = None,
) -> BoundScene:
    """Strictly bind names, coordinates, contacts and physics to a live backend."""

    if not isinstance(backend, SceneBackend):
        raise SceneContractError("backend does not expose a runtime path and resolved name cache")
    path = backend.runtime_path if runtime_mjcf_path is None else runtime_mjcf_path
    _validate_names(description, backend.name_cache)
    _validate_xml(description, config, path)
    return BoundScene(description, path.resolve(), backend.name_cache.object_ids)


__all__ = [
    "ASSET_PATCH_ROOT",
    "BoundScene",
    "ContactPhaseRule",
    "ContactRules",
    "CylinderDescription",
    "NamedTaskFrame",
    "RobotInitialPose",
    "SceneBackend",
    "SceneContractError",
    "SceneDescription",
    "TableDescription",
    "bind_scene",
    "create_scene_description",
]
