"""Deterministic TienKung 2 Pro URDF-to-MJCF asset pipeline.

The source URDF is imported by the pinned MuJoCo compiler first.  The saved
base MJCF is then normalized and patched with explicitly versioned MuJoCo
features.  Runtime code consumes committed outputs and never converts assets
on installation or startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml  # type: ignore[import-untyped]

from tienkung_rl.config.load import DEFAULT_CONFIG_PATH, load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.paths import ASSET_ROOT
from tienkung_rl.paths import PROJECT_ROOT as REPOSITORY_ROOT

SOURCE_DIR = ASSET_ROOT / "source"
PATCH_DIR = ASSET_ROOT / "patches"
MJCF_DIR = ASSET_ROOT / "mjcf"
SOURCE_URDF = SOURCE_DIR / "tiangong2.0_pro_with_hands.urdf"
VISUAL_MESH_DIR = SOURCE_DIR / "meshes"
BASE_MJCF_NAME = "tienkung2_pro.base.xml"
RUNTIME_MJCF_NAME = "reach_then_close.xml"
VIEWER_MJCF_NAME = "reach_then_close.viewer.xml"
SOURCE_MANIFEST_NAME = "source_manifest.json"
NAME_INDEX_NAME = "name_index.json"
JOINT_HIERARCHY_NAME = "joint_hierarchy.json"
GENERATION_MANIFEST_NAME = "manifest.json"
MOVABLE_URDF_TYPES = frozenset({"revolute", "continuous", "prismatic"})
EXPECTED_SOURCE_MOVABLE_JOINTS = 54
EXPECTED_TOTAL_URDF_JOINTS = 62
EXPECTED_MESHES = 63
PATCH_SCHEMA_VERSION = "tienkung2_pro_patch_schema_v2"
GENERATOR_VERSION = "tienkung2_pro_asset_pipeline_v3"
FLOAT_FORMAT = ".12g"
URDF_IMPORT_OPTIONS = {
    "discardvisual": "false",
    "fusestatic": "false",
    "strippath": "false",
}
GENERATED_RELATIVE_PATHS = (
    Path("source") / SOURCE_MANIFEST_NAME,
    Path("mjcf") / BASE_MJCF_NAME,
    Path("mjcf") / RUNTIME_MJCF_NAME,
    Path("mjcf") / VIEWER_MJCF_NAME,
    Path("mjcf") / NAME_INDEX_NAME,
    Path("mjcf") / JOINT_HIERARCHY_NAME,
    Path(GENERATION_MANIFEST_NAME),
)


class AssetPipelineError(RuntimeError):
    """Base class for deterministic asset generation failures."""


class SourceAssetError(AssetPipelineError):
    """The immutable URDF/STL source inventory is incomplete or inconsistent."""


class PatchSchemaError(AssetPipelineError):
    """A versioned patch declaration violates its strict schema."""


class PatchConflictError(AssetPipelineError):
    """Two patch operations claim the same semantic target."""


class UnknownPatchTargetError(AssetPipelineError):
    """A patch names a body, joint, site, actuator, or geom that does not exist."""


@dataclass(frozen=True)
class SourceJoint:
    """URDF joint metadata retained independently of MuJoCo compilation."""

    index: int
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float]
    lower: float
    upper: float
    effort: float
    velocity: float
    mimic_joint: str | None
    mimic_multiplier: float | None
    mimic_offset: float | None


@dataclass(frozen=True)
class SourceInventory:
    """Validated source inputs in deterministic order."""

    urdf_path: Path
    mesh_dir: Path
    joints: tuple[SourceJoint, ...]
    links: tuple[str, ...]
    mesh_references: tuple[str, ...]
    mesh_paths: tuple[Path, ...]

    @property
    def joint_by_name(self) -> dict[str, SourceJoint]:
        return {joint.name: joint for joint in self.joints}

    @property
    def mimic(self) -> dict[str, tuple[str, float, float]]:
        return {
            joint.name: (
                _not_none(joint.mimic_joint),
                _not_none(joint.mimic_multiplier),
                _not_none(joint.mimic_offset),
            )
            for joint in self.joints
            if joint.mimic_joint is not None
        }


@dataclass(frozen=True)
class PatchBundle:
    """Strictly loaded declarations forming one atomic patch version."""

    main: Mapping[str, Any]
    collisions: Mapping[str, Any]
    contacts: Mapping[str, Any]
    sites: Mapping[str, Any]
    scene: Mapping[str, Any]
    file_paths: tuple[Path, ...]


@dataclass(frozen=True)
class GenerationResult:
    """Paths and digests produced by one generation run."""

    output_root: Path
    hashes: Mapping[str, str]
    model_summary: Mapping[str, int]


class PatchTracker:
    """Detect duplicate names and attribute overwrites across ordered patches."""

    def __init__(self) -> None:
        self._claims: dict[tuple[str, str], tuple[str, Any]] = {}

    def claim(self, namespace: str, target: str, source: str, value: Any) -> None:
        key = (namespace, target)
        previous = self._claims.get(key)
        if previous is not None:
            previous_source, previous_value = previous
            raise PatchConflictError(
                f"patch conflict for {namespace} {target!r}: {previous_source}={previous_value!r}, {source}={value!r}"
            )
        self._claims[key] = (source, value)

    def set_attribute(
        self,
        element: ET.Element,
        attribute: str,
        value: str,
        *,
        source: str,
        target: str,
    ) -> None:
        self.claim("attribute", f"{target}@{attribute}", source, value)
        element.set(attribute, value)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise PatchSchemaError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _not_none(value: Any) -> Any:
    if value is None:
        raise AssertionError("validated value unexpectedly absent")
    return value


def _require(condition: bool, message: str, error: type[Exception] = AssetPipelineError) -> None:
    if not condition:
        raise error(message)


def _require_keys(
    value: Any,
    *,
    required: Iterable[str],
    allowed: Iterable[str],
    context: str,
) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{context} must be a mapping", PatchSchemaError)
    keys = set(value)
    required_set = set(required)
    allowed_set = set(allowed)
    missing = sorted(required_set - keys)
    unknown = sorted(keys - allowed_set)
    _require(not missing, f"{context} missing keys: {missing}", PatchSchemaError)
    _require(not unknown, f"{context} has unknown keys: {unknown}", PatchSchemaError)
    return value


def _tuple_of_finite_numbers(value: Any, length: int, context: str) -> tuple[float, ...]:
    _require(
        isinstance(value, list) and len(value) == length,
        f"{context} must contain {length} numbers",
        PatchSchemaError,
    )
    result = tuple(float(item) for item in value)
    _require(all(math.isfinite(item) for item in result), f"{context} must be finite", PatchSchemaError)
    return result


def _format_number(value: float) -> str:
    number = float(value)
    if number == 0.0:
        number = 0.0
    return format(number, FLOAT_FORMAT)


def _format_numbers(values: Iterable[float]) -> str:
    return " ".join(_format_number(value) for value in values)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError, PatchSchemaError) as exc:
        if isinstance(exc, PatchSchemaError):
            raise
        raise PatchSchemaError(f"cannot load {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a mapping", PatchSchemaError)
    return value


def _float_attribute(element: ET.Element, name: str, context: str) -> float:
    raw = element.get(name)
    _require(raw is not None, f"{context} is missing {name}", SourceAssetError)
    try:
        value = float(_not_none(raw))
    except ValueError as exc:
        raise SourceAssetError(f"{context} has invalid {name}={raw!r}") from exc
    _require(math.isfinite(value), f"{context} has non-finite {name}", SourceAssetError)
    return value


def _parse_axis(joint: ET.Element) -> tuple[float, float, float]:
    axis = joint.find("axis")
    raw = "1 0 0" if axis is None else axis.get("xyz", "1 0 0")
    try:
        values = tuple(float(item) for item in raw.split())
    except ValueError as exc:
        raise SourceAssetError(f"joint {joint.get('name')!r} has invalid axis {raw!r}") from exc
    _require(len(values) == 3, f"joint {joint.get('name')!r} axis must have 3 values", SourceAssetError)
    _require(all(math.isfinite(item) for item in values), "joint axis must be finite", SourceAssetError)
    length = math.sqrt(sum(item * item for item in values))
    _require(length > 0.0, f"joint {joint.get('name')!r} axis is zero", SourceAssetError)
    return values  # type: ignore[return-value]


def resolve_urdf_mesh_reference(reference: str, *, urdf_path: Path, mesh_dir: Path) -> Path:
    """Resolve the archived ``../meshes`` convention with exact-case checks."""

    posix = PurePosixPath(reference)
    _require(not posix.is_absolute(), f"absolute mesh reference is forbidden: {reference}", SourceAssetError)
    _require(".." not in posix.parts[1:], f"unsafe mesh reference: {reference}", SourceAssetError)
    _require(
        len(posix.parts) == 3 and posix.parts[:2] == ("..", "meshes"),
        f"unexpected mesh reference {reference!r} in {urdf_path.name}; expected ../meshes/<file>",
        SourceAssetError,
    )
    exact_names = {entry.name for entry in mesh_dir.iterdir() if entry.is_file()}
    filename = posix.name
    _require(
        filename in exact_names,
        f"missing or case-mismatched mesh reference {reference!r} in {urdf_path.name}",
        SourceAssetError,
    )
    candidate = mesh_dir / filename
    _require(candidate.is_file(), f"mesh is not a regular file: {candidate}", SourceAssetError)
    return candidate


def inspect_source_assets(urdf_path: Path = SOURCE_URDF, mesh_dir: Path = VISUAL_MESH_DIR) -> SourceInventory:
    """Validate source structure, exact mesh references, and 54-joint contract."""

    _require(urdf_path.is_file(), f"source URDF not found: {urdf_path}", SourceAssetError)
    _require(mesh_dir.is_dir(), f"visual mesh directory not found: {mesh_dir}", SourceAssetError)
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise SourceAssetError(f"invalid URDF XML {urdf_path}: {exc}") from exc
    _require(root.tag == "robot", f"{urdf_path} is not a URDF robot", SourceAssetError)

    link_elements = root.findall("link")
    links = tuple(_not_none(link.get("name")) for link in link_elements)
    _require(len(links) == len(set(links)), "URDF contains duplicate link names", SourceAssetError)
    link_set = set(links)
    all_joints = root.findall("joint")
    _require(
        len(all_joints) == EXPECTED_TOTAL_URDF_JOINTS,
        f"expected {EXPECTED_TOTAL_URDF_JOINTS} total URDF joints, found {len(all_joints)}",
        SourceAssetError,
    )
    movable = [joint for joint in all_joints if joint.get("type") in MOVABLE_URDF_TYPES]
    _require(
        len(movable) == EXPECTED_SOURCE_MOVABLE_JOINTS,
        f"expected {EXPECTED_SOURCE_MOVABLE_JOINTS} movable URDF joints, found {len(movable)}",
        SourceAssetError,
    )
    names = [_not_none(joint.get("name")) for joint in movable]
    _require(len(names) == len(set(names)), "URDF contains duplicate movable joint names", SourceAssetError)

    joints: list[SourceJoint] = []
    for index, joint in enumerate(movable):
        name = _not_none(joint.get("name"))
        parent_element = joint.find("parent")
        child_element = joint.find("child")
        limit = joint.find("limit")
        _require(parent_element is not None, f"joint {name} has no parent", SourceAssetError)
        _require(child_element is not None, f"joint {name} has no child", SourceAssetError)
        _require(limit is not None, f"joint {name} has no limits", SourceAssetError)
        parent_element = _not_none(parent_element)
        child_element = _not_none(child_element)
        limit = _not_none(limit)
        parent = _not_none(parent_element.get("link"))
        child = _not_none(child_element.get("link"))
        _require(parent in link_set and child in link_set, f"joint {name} targets unknown link", SourceAssetError)
        lower = _float_attribute(limit, "lower", f"joint {name} limit")
        upper = _float_attribute(limit, "upper", f"joint {name} limit")
        effort = _float_attribute(limit, "effort", f"joint {name} limit")
        velocity = _float_attribute(limit, "velocity", f"joint {name} limit")
        _require(lower <= upper, f"joint {name} lower limit exceeds upper", SourceAssetError)
        _require(effort > 0.0 and velocity > 0.0, f"joint {name} effort/velocity must be positive", SourceAssetError)
        mimic = joint.find("mimic")
        mimic_joint: str | None = None
        multiplier: float | None = None
        offset: float | None = None
        if mimic is not None:
            mimic_joint = _not_none(mimic.get("joint"))
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            _require(
                math.isfinite(multiplier) and math.isfinite(offset),
                f"joint {name} mimic is non-finite",
                SourceAssetError,
            )
        joints.append(
            SourceJoint(
                index=index,
                name=name,
                joint_type=_not_none(joint.get("type")),
                parent_link=parent,
                child_link=child,
                axis_xyz=_parse_axis(joint),
                lower=lower,
                upper=upper,
                effort=effort,
                velocity=velocity,
                mimic_joint=mimic_joint,
                mimic_multiplier=multiplier,
                mimic_offset=offset,
            )
        )
    joint_names = {joint.name for joint in joints}
    for follower, (leader, _, _) in SourceInventory(urdf_path, mesh_dir, tuple(joints), links, (), ()).mimic.items():
        _require(leader in joint_names, f"mimic {follower} targets unknown joint {leader}", SourceAssetError)

    mesh_references = tuple(_not_none(mesh.get("filename")) for mesh in root.findall(".//mesh"))
    mesh_paths_by_name: dict[str, Path] = {}
    for reference in mesh_references:
        path = resolve_urdf_mesh_reference(reference, urdf_path=urdf_path, mesh_dir=mesh_dir)
        mesh_paths_by_name[path.name] = path
    _require(
        len(mesh_paths_by_name) == EXPECTED_MESHES,
        f"expected {EXPECTED_MESHES} distinct STL references, found {len(mesh_paths_by_name)}",
        SourceAssetError,
    )
    actual_stl = tuple(sorted(path for path in mesh_dir.iterdir() if path.suffix == ".STL"))
    _require(
        len(actual_stl) == EXPECTED_MESHES,
        f"expected {EXPECTED_MESHES} exact-case .STL files, found {len(actual_stl)}",
        SourceAssetError,
    )
    _require(
        {path.name for path in actual_stl} == set(mesh_paths_by_name),
        "source mesh directory contains unreferenced or missing STL files",
        SourceAssetError,
    )
    _require("right_tcp_link" in link_set, "URDF is missing right_tcp_link", SourceAssetError)
    return SourceInventory(
        urdf_path=urdf_path,
        mesh_dir=mesh_dir,
        joints=tuple(joints),
        links=links,
        mesh_references=mesh_references,
        mesh_paths=actual_stl,
    )


def build_source_manifest(inventory: SourceInventory) -> Mapping[str, Any]:
    reference_by_name: dict[str, list[str]] = {}
    for reference in inventory.mesh_references:
        reference_by_name.setdefault(PurePosixPath(reference).name, []).append(reference)
    return {
        "schema_version": "tienkung2_pro_source_manifest_v1",
        "source_origin": "legacy/isaaclab/source/legged_lab/assets/tienkung2_pro",
        "mapping_rule": "URDF ../meshes/<name> -> source/meshes/<exact-name>",
        "urdf": {
            "path": f"source/{inventory.urdf_path.name}",
            "size_bytes": inventory.urdf_path.stat().st_size,
            "sha256": sha256_file(inventory.urdf_path),
        },
        "meshes": [
            {
                "path": f"source/meshes/{path.name}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "urdf_references": sorted(set(reference_by_name[path.name])),
            }
            for path in inventory.mesh_paths
        ],
        "notices": [
            {
                "path": f"source/{name}",
                "size_bytes": (SOURCE_DIR / name).stat().st_size,
                "sha256": sha256_file(SOURCE_DIR / name),
            }
            for name in ("LICENSE", "NOTICE", "README.md")
        ],
        "counts": {
            "links": len(inventory.links),
            "total_joints": EXPECTED_TOTAL_URDF_JOINTS,
            "movable_joints": len(inventory.joints),
            "distinct_meshes": len(inventory.mesh_paths),
            "mesh_references": len(inventory.mesh_references),
            "mimic_relations": len(inventory.mimic),
        },
    }


def _validate_main_patch(main: Mapping[str, Any]) -> None:
    expected_order = [
        "fixed_base",
        "joint_posture",
        "gravity_compensation",
        "visual_only",
        "collisions",
        "contact_groups",
        "sites",
        "actuators",
        "mimic",
        "scene",
        "metadata",
    ]
    required = {
        "schema_version",
        "patch_version",
        "asset_contract_version",
        "application_order",
        "fixed_base",
        "joint_posture",
        "gravity_compensation",
        "actuators",
        "mimic",
        "sites",
        "collisions",
        "contact_groups",
        "scene",
    }
    _require_keys(main, required=required, allowed=required, context="asset_patch.yaml")
    _require(main["schema_version"] == PATCH_SCHEMA_VERSION, "unsupported patch schema", PatchSchemaError)
    _require(main["application_order"] == expected_order, "patch application_order is not canonical", PatchSchemaError)
    _require(
        isinstance(main["patch_version"], str) and bool(main["patch_version"]),
        "empty patch_version",
        PatchSchemaError,
    )


def load_patch_bundle(patch_dir: Path = PATCH_DIR) -> PatchBundle:
    paths = tuple(
        patch_dir / name
        for name in (
            "asset_patch.yaml",
            "collisions.yaml",
            "contact_groups.yaml",
            "sites.yaml",
            "scene.yaml",
            "schema.json",
        )
    )
    for path in paths:
        _require(path.is_file(), f"patch file not found: {path}", PatchSchemaError)
    main = _load_yaml(paths[0])
    collisions = _load_yaml(paths[1])
    contacts = _load_yaml(paths[2])
    sites = _load_yaml(paths[3])
    scene = _load_yaml(paths[4])
    _validate_main_patch(main)
    _require(
        collisions.get("schema_version") == "tienkung2_pro_collisions_v1", "bad collision schema", PatchSchemaError
    )
    collision_calibration = _require_keys(
        collisions.get("calibration"),
        required={"version", "status", "method"},
        allowed={"version", "status", "method"},
        context="collisions.calibration",
    )
    _require(
        collision_calibration["status"] == "starting_point_unvalidated",
        "collision calibration must remain explicitly unvalidated until formal acceptance",
        PatchSchemaError,
    )
    _require(
        contacts.get("schema_version") == "tienkung2_pro_contact_groups_v1", "bad contact schema", PatchSchemaError
    )
    _require(sites.get("schema_version") == "tienkung2_pro_sites_v1", "bad site schema", PatchSchemaError)
    _require(scene.get("schema_version") == "tienkung2_pro_scene_v1", "bad scene schema", PatchSchemaError)
    _require(isinstance(collisions.get("geometries"), list), "collisions.geometries must be a list", PatchSchemaError)
    _require(isinstance(contacts.get("groups"), dict), "contact groups must be a mapping", PatchSchemaError)
    _require(isinstance(contacts.get("phase_rules"), dict), "phase rules must be a mapping", PatchSchemaError)
    _require(isinstance(sites.get("frames"), list), "sites.frames must be a list", PatchSchemaError)
    _require(isinstance(sites.get("sites"), list), "sites.sites must be a list", PatchSchemaError)
    return PatchBundle(main, collisions, contacts, sites, scene, paths)


def _import_urdf_with_mujoco(inventory: SourceInventory, expected_version: str) -> ET.Element:
    try:
        import mujoco  # type: ignore[import-untyped]
    except ImportError as exc:
        raise AssetPipelineError("MuJoCo is required to regenerate assets") from exc
    _require(
        mujoco.__version__ == expected_version == "3.11.0",
        f"asset compiler requires MuJoCo 3.11.0, found {mujoco.__version__}",
    )
    source_root = ET.parse(inventory.urdf_path).getroot()
    _require(
        source_root.find("mujoco") is None, "source URDF already contains MuJoCo compiler directives", SourceAssetError
    )
    mujoco_directive = ET.SubElement(source_root, "mujoco")
    ET.SubElement(mujoco_directive, "compiler", dict(URDF_IMPORT_OPTIONS))
    source_xml = ET.tostring(source_root, encoding="unicode")
    assets = {path.name: path.read_bytes() for path in inventory.mesh_paths}
    try:
        model = mujoco.MjModel.from_xml_string(source_xml, assets)
    except ValueError as exc:
        raise AssetPipelineError(f"MuJoCo URDF import failed: {exc}") from exc
    _require(model.njnt == EXPECTED_SOURCE_MOVABLE_JOINTS, "MuJoCo import lost source joints")
    with tempfile.TemporaryDirectory(prefix="tienkung-mjcf-import-") as temporary:
        raw_path = Path(temporary) / "raw.xml"
        mujoco.mj_saveLastXML(str(raw_path), model)
        raw_root = ET.parse(raw_path).getroot()
    return _normalize_base_mjcf(raw_root, inventory)


def _body_map(root: ET.Element) -> dict[str, ET.Element]:
    worldbody = root.find("worldbody")
    _require(worldbody is not None, "MJCF has no worldbody")
    worldbody = _not_none(worldbody)
    result = {_not_none(body.get("name")): body for body in worldbody.iter("body") if body.get("name") is not None}
    _require(len(result) == len(list(worldbody.iter("body"))), "MJCF contains unnamed/duplicate bodies")
    return result


def _joint_map(root: ET.Element) -> dict[str, ET.Element]:
    worldbody = root.find("worldbody")
    _require(worldbody is not None, "MJCF has no worldbody")
    worldbody = _not_none(worldbody)
    result = {_not_none(joint.get("name")): joint for joint in worldbody.iter("joint") if joint.get("name") is not None}
    return result


def _normalize_base_mjcf(root: ET.Element, inventory: SourceInventory) -> ET.Element:
    root.set("model", "tienkung2_pro_base")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.attrib.clear()
    compiler.attrib.update({"angle": "radian", "autolimits": "true", "meshdir": "../source/meshes"})
    asset = root.find("asset")
    _require(asset is not None, "MuJoCo import produced no asset section")
    asset = _not_none(asset)
    meshes = list(asset.findall("mesh"))
    _require(len(meshes) == EXPECTED_MESHES, "MuJoCo import lost visual meshes")
    for mesh in meshes:
        filename = PurePosixPath(_not_none(mesh.get("file"))).name
        _require((inventory.mesh_dir / filename).is_file(), f"imported mesh is unknown: {filename}")
        mesh.set("file", filename)
        mesh.attrib.pop("content_type", None)

    body_by_element = {element: name for name, element in _body_map(root).items()}
    worldbody = _not_none(root.find("worldbody"))
    for parent in [worldbody, *list(worldbody.iter("body"))]:
        body_name = "pelvis" if parent is worldbody else body_by_element[parent]
        counters = {"visual": 0, "collision": 0}
        for geom in parent.findall("geom"):
            visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            role = "visual" if visual else "urdf_collision"
            number = counters["visual" if visual else "collision"]
            counters["visual" if visual else "collision"] += 1
            suffix = "" if number == 0 else f"_{number}"
            geom.set("name", f"{body_name}_{role}{suffix}")
            if visual:
                geom.set("group", "1")
                geom.set("density", "0")
    _require(len(_joint_map(root)) == EXPECTED_SOURCE_MOVABLE_JOINTS, "normalized base lost joints")
    _require(set(_joint_map(root)) == {joint.name for joint in inventory.joints}, "joint names differ after import")
    bodies = _body_map(root)
    _require("right_tcp_link" in bodies and "R_base_link" in bodies, "fusestatic=false was not honored")
    return root


def _find_body(root: ET.Element, name: str, source: str) -> ET.Element:
    if name == "world":
        worldbody = root.find("worldbody")
        return _not_none(worldbody)
    body = _body_map(root).get(name)
    if body is None:
        raise UnknownPatchTargetError(f"{source} targets unknown body {name!r}")
    return body


def _ensure_section(root: ET.Element, tag: str) -> ET.Element:
    existing = root.find(tag)
    if existing is not None:
        return existing
    element = ET.Element(tag)
    root.append(element)
    return element


def _custom_text(root: ET.Element, tracker: PatchTracker, name: str, data: str, source: str) -> None:
    tracker.claim("custom", name, source, data)
    custom = _ensure_section(root, "custom")
    ET.SubElement(custom, "text", {"name": name, "data": data})


def _custom_numeric(root: ET.Element, tracker: PatchTracker, name: str, values: Iterable[float], source: str) -> None:
    data = _format_numbers(values)
    tracker.claim("custom", name, source, data)
    custom = _ensure_section(root, "custom")
    ET.SubElement(custom, "numeric", {"name": name, "data": data})


def _apply_fixed_base(root: ET.Element, bundle: PatchBundle, tracker: PatchTracker) -> None:
    spec = _require_keys(
        bundle.main["fixed_base"],
        required={"root_body", "position_m", "quaternion_wxyz", "require_no_free_joint"},
        allowed={"root_body", "position_m", "quaternion_wxyz", "require_no_free_joint"},
        context="fixed_base",
    )
    body_name = str(spec["root_body"])
    body = _find_body(root, body_name, "fixed_base")
    if bool(spec["require_no_free_joint"]):
        _require(
            not any(child.tag == "freejoint" or child.get("type") == "free" for child in body),
            f"fixed base body {body_name} has a free joint",
            PatchSchemaError,
        )
    tracker.set_attribute(
        body,
        "pos",
        _format_numbers(_tuple_of_finite_numbers(spec["position_m"], 3, "fixed_base.position_m")),
        source="fixed_base",
        target=f"body:{body_name}",
    )
    tracker.set_attribute(
        body,
        "quat",
        _format_numbers(_tuple_of_finite_numbers(spec["quaternion_wxyz"], 4, "fixed_base.quaternion_wxyz")),
        source="fixed_base",
        target=f"body:{body_name}",
    )
    _custom_text(root, tracker, "asset.fixed_base", body_name, "fixed_base")


def _joint_roles(config: ResolvedRunConfig, inventory: SourceInventory) -> dict[str, str]:
    arm = set(config.physics.actuator_calibration.arm.joint_names)
    active_hand = set(config.physics.actuator_calibration.hand.joint_names)
    right_mimic = {name for name in inventory.mimic if name.startswith("right_")}
    _require(len(arm) == 7, "M01 arm calibration must name 7 joints", PatchSchemaError)
    _require(len(active_hand) == 6, "M01 hand calibration must name 6 joints", PatchSchemaError)
    _require(len(right_mimic) == 6, "source URDF must contain 6 right-hand mimic followers", SourceAssetError)
    _require(not (arm & active_hand or arm & right_mimic or active_hand & right_mimic), "active joint roles overlap")
    source_names = {joint.name for joint in inventory.joints}
    _require(arm | active_hand | right_mimic <= source_names, "M01 actuator joint absent from URDF")
    return {
        joint.name: (
            "active_arm"
            if joint.name in arm
            else "active_hand"
            if joint.name in active_hand
            else "mimic_follower"
            if joint.name in right_mimic
            else "fixed_non_task"
        )
        for joint in inventory.joints
    }


def _apply_joint_posture(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    inventory: SourceInventory,
) -> tuple[dict[str, str], dict[str, float]]:
    spec = _require_keys(
        bundle.main["joint_posture"],
        required={
            "default_fixed_position_rad",
            "fixed_position_overrides_rad",
            "home_keyframe",
            "minimum_robot_table_clearance_m",
        },
        allowed={
            "default_fixed_position_rad",
            "fixed_position_overrides_rad",
            "home_keyframe",
            "minimum_robot_table_clearance_m",
        },
        context="joint_posture",
    )
    minimum_clearance = float(spec["minimum_robot_table_clearance_m"])
    _require(
        math.isfinite(minimum_clearance) and minimum_clearance > 0.0,
        "joint_posture.minimum_robot_table_clearance_m must be finite and positive",
        PatchSchemaError,
    )
    default = float(spec["default_fixed_position_rad"])
    overrides = spec["fixed_position_overrides_rad"]
    _require(isinstance(overrides, dict), "joint posture overrides must be a mapping", PatchSchemaError)
    joint_map = _joint_map(root)
    unknown = sorted(set(overrides) - set(joint_map))
    if unknown:
        raise UnknownPatchTargetError(f"joint posture targets unknown joints: {unknown}")
    roles = _joint_roles(config, inventory)
    fixed_positions: dict[str, float] = {}
    equality = _ensure_section(root, "equality")
    for joint in inventory.joints:
        if roles[joint.name] != "fixed_non_task":
            continue
        position = float(overrides.get(joint.name, default))
        _require(math.isfinite(position), f"fixed posture for {joint.name} is non-finite", PatchSchemaError)
        _require(
            joint.lower <= position <= joint.upper,
            f"fixed posture {position} outside {joint.name} source range [{joint.lower}, {joint.upper}]",
            PatchSchemaError,
        )
        name = f"fixed__{joint.name}"
        tracker.claim("equality", name, "joint_posture", position)
        ET.SubElement(
            equality,
            "joint",
            {
                "name": name,
                "joint1": joint.name,
                "polycoef": _format_numbers((position, 0.0, 0.0, 0.0, 0.0)),
                "solimp": _format_numbers(config.physics.contact.impedance),
                "solref": _format_numbers(
                    (
                        config.physics.physics_dt_s,
                        config.physics.contact.reference_damping,
                    )
                ),
            },
        )
        fixed_positions[joint.name] = position
    _require(len(fixed_positions) == 35, f"expected 35 fixed non-task joints, found {len(fixed_positions)}")
    _custom_text(
        root,
        tracker,
        "asset.fixed_non_task_joints",
        ",".join(fixed_positions),
        "joint_posture",
    )
    return roles, fixed_positions


def _apply_gravity_compensation(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    inventory: SourceInventory,
) -> tuple[str, ...]:
    """Apply exact MuJoCo gravity compensation to the active right-arm subtree."""

    spec = _require_keys(
        bundle.main["gravity_compensation"],
        required={"root_body", "factor", "include_descendants"},
        allowed={"root_body", "factor", "include_descendants"},
        context="gravity_compensation",
    )
    root_body_name = str(spec["root_body"])
    factor = float(spec["factor"])
    _require(
        math.isfinite(factor) and 0.0 <= factor <= 1.0,
        "gravity_compensation.factor must lie in [0, 1]",
        PatchSchemaError,
    )
    _require(
        spec["include_descendants"] is True,
        "gravity_compensation must include the complete distal subtree",
        PatchSchemaError,
    )
    root_body = _find_body(root, root_body_name, "gravity_compensation")
    compensated = tuple(root_body.iter("body"))
    names = tuple(_not_none(body.get("name")) for body in compensated)
    _require(len(names) == len(set(names)), "gravity-compensated subtree contains duplicate body names")

    controlled_joints = {
        *config.physics.actuator_calibration.arm.joint_names,
        *config.physics.actuator_calibration.hand.joint_names,
    }
    distal_links = {
        joint.child_link
        for joint in inventory.joints
        if joint.name in controlled_joints or (joint.mimic_joint is not None and joint.name.startswith("right_"))
    }
    _require(
        distal_links <= set(names),
        "gravity-compensated subtree does not cover every active-arm, active-hand and mimic body",
        PatchSchemaError,
    )
    for body, name in zip(compensated, names, strict=True):
        tracker.set_attribute(
            body,
            "gravcomp",
            _format_number(factor),
            source="gravity_compensation",
            target=f"body:{name}",
        )
    _custom_text(root, tracker, "asset.gravity_compensation.root_body", root_body_name, "gravity_compensation")
    _custom_numeric(root, tracker, "asset.gravity_compensation.factor", (factor,), "gravity_compensation")
    _custom_numeric(
        root,
        tracker,
        "asset.gravity_compensation.body_count",
        (float(len(names)),),
        "gravity_compensation",
    )
    return names


def _mesh_collision_bodies(bundle: PatchBundle) -> set[str]:
    """Return bodies whose imported URDF mesh collision is retained."""

    result: set[str] = set()
    for index, raw in enumerate(bundle.collisions["geometries"]):
        _require(isinstance(raw, dict), f"collision {index} must be a mapping", PatchSchemaError)
        if str(raw.get("type", "")) == "mesh":
            body = raw.get("body")
            _require(isinstance(body, str) and body, f"mesh collision {index} needs a body", PatchSchemaError)
            result.add(body)
    return result


def _physical_mesh_collision_count(bundle: PatchBundle) -> int:
    return sum(
        1
        for raw in bundle.collisions["geometries"]
        if isinstance(raw, dict)
        and str(raw.get("type", "")) == "mesh"
        and str(raw.get("mode", "physical")) == "physical"
    )


def _apply_visual_only(root: ET.Element, preserve_mesh_bodies: set[str]) -> None:
    worldbody = _not_none(root.find("worldbody"))
    removed = 0
    body_names = {element: name for name, element in _body_map(root).items()}
    for parent in [worldbody, *list(worldbody.iter("body"))]:
        body_name = body_names.get(parent)
        for geom in list(parent.findall("geom")):
            if geom.get("type") == "mesh" and not (geom.get("contype") == "0" and geom.get("conaffinity") == "0"):
                if body_name in preserve_mesh_bodies:
                    continue
                parent.remove(geom)
                removed += 1
    expected_removed = EXPECTED_MESHES - len(preserve_mesh_bodies)
    _require(
        removed == expected_removed, f"expected to remove {expected_removed} STL collision geoms, removed {removed}"
    )
    for geom in worldbody.iter("geom"):
        if geom.get("type") == "mesh":
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            parent = next(
                (candidate for candidate in [worldbody, *worldbody.iter("body")] if geom in list(candidate)), None
            )
            _require(
                parent is not None and body_names.get(parent) in preserve_mesh_bodies,
                f"unexpected collision-enabled STL mesh {geom.get('name')!r}",
                PatchSchemaError,
            )


def _contact_attributes(config: ResolvedRunConfig, *, material_name: str, mask: Mapping[str, Any]) -> dict[str, str]:
    contact = (
        (config.physics.cylinder_contact or config.physics.contact)
        if material_name == config.physics.cylinder_material.name
        else config.physics.contact
    )
    material = next(
        candidate
        for candidate in (
            config.physics.robot_material,
            config.physics.table_material,
            config.physics.cylinder_material,
        )
        if candidate.name == material_name
    )
    return {
        "material": material.name,
        "friction": _format_numbers(
            (material.sliding_friction, material.torsional_friction, material.rolling_friction)
        ),
        "solimp": _format_numbers(contact.impedance),
        "solref": _format_numbers((contact.reference_time_s, contact.reference_damping)),
        "margin": _format_number(contact.margin_m),
        "gap": _format_number(contact.gap_m),
        "contype": str(int(mask["contype"])),
        "conaffinity": str(int(mask["conaffinity"])),
        "group": "3",
        "density": "0",
    }


def _apply_collisions(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
) -> set[str]:
    masks = _require_keys(
        bundle.scene["collision_masks"],
        required={"robot", "table", "cylinder"},
        allowed={"robot", "table", "cylinder"},
        context="scene.collision_masks",
    )
    _require(
        isinstance(masks, dict) and isinstance(masks.get("robot"), dict),
        "missing robot collision mask",
        PatchSchemaError,
    )
    names: set[str] = set()
    valid_types = {"sphere", "capsule", "box", "cylinder", "mesh"}
    for index, raw in enumerate(bundle.collisions["geometries"]):
        _require(isinstance(raw, dict), f"collision {index} must be a mapping", PatchSchemaError)
        geom_type = str(raw.get("type", ""))
        required = {"name", "body", "type", "contact_groups"}
        required.add("mesh" if geom_type == "mesh" else "size_m")
        allowed = required | {"pos_m", "fromto_m", "quat_wxyz", "mode"}
        spec = _require_keys(raw, required=required, allowed=allowed, context=f"collision[{index}]")
        name = str(spec["name"])
        _require(geom_type in valid_types, f"unsupported collision type {geom_type}", PatchSchemaError)
        tracker.claim("geom", name, "collisions.yaml", spec)
        body = _find_body(root, str(spec["body"]), "collisions.yaml")
        attributes = {
            "name": name,
            "type": geom_type,
            **_contact_attributes(
                config,
                material_name=config.physics.robot_material.name,
                mask=masks["robot"],
            ),
        }
        if geom_type == "mesh":
            mesh_name = str(spec["mesh"])
            asset = _not_none(root.find("asset"))
            _require(
                any(item.tag == "mesh" and item.get("name") == mesh_name for item in asset),
                f"collision {name} references unknown mesh {mesh_name!r}",
                PatchSchemaError,
            )
            candidates = [
                geom
                for geom in body.findall("geom")
                if geom.get("type") == "mesh" and geom.get("mesh") == mesh_name and geom.get("contype") != "0"
            ]
            _require(
                len(candidates) == 1,
                f"mesh collision {name} needs exactly one imported URDF collision geom",
                PatchSchemaError,
            )
            geom = candidates[0]
            mode = str(spec.get("mode", "physical"))
            _require(
                mode in {"physical", "diagnostic_only"}, f"unsupported mesh collision mode {mode!r}", PatchSchemaError
            )
            # STL contact surfaces are already conservative (MuJoCo uses a
            # convex collision representation), so do not add the primitive
            # 2 mm early-contact margin used by the calibrated arm shapes.
            attributes["margin"] = "0"
            if mode == "diagnostic_only":
                attributes.update({"contype": "0", "conaffinity": "0", "group": "1", "density": "0"})
            geom.attrib.update(attributes)
            # Fingers may be below the tabletop in the calibrated home pose.
            # Keep their STL contacts with the task object, but exclude the
            # robot/table pair so reset does not inject support forces into
            # the arm controller.
            if mode == "physical":
                contact = _ensure_section(root, "contact")
                table_body = str(bundle.scene["table"]["body"])
                if not any(
                    item.tag == "exclude" and {item.get("body1"), item.get("body2")} == {str(spec["body"]), table_body}
                    for item in contact
                ):
                    ET.SubElement(contact, "exclude", {"body1": str(spec["body"]), "body2": table_body})
            names.add(name)
            continue
        size_value = spec["size_m"]
        _require(isinstance(size_value, list) and bool(size_value), f"collision {name} has no size", PatchSchemaError)
        sizes = tuple(float(value) for value in size_value)
        _require(
            all(math.isfinite(value) and value > 0.0 for value in sizes),
            f"collision {name} size must be positive",
            PatchSchemaError,
        )
        expected_sizes = 3 if geom_type == "box" else 2 if geom_type == "cylinder" else 1
        _require(len(sizes) == expected_sizes, f"collision {name} has wrong size arity", PatchSchemaError)
        attributes["size"] = _format_numbers(sizes)
        if "pos_m" in spec:
            attributes["pos"] = _format_numbers(_tuple_of_finite_numbers(spec["pos_m"], 3, f"{name}.pos_m"))
        if "fromto_m" in spec:
            _require(geom_type == "capsule", f"{name}.fromto_m requires capsule", PatchSchemaError)
            attributes["fromto"] = _format_numbers(_tuple_of_finite_numbers(spec["fromto_m"], 6, f"{name}.fromto_m"))
        if "quat_wxyz" in spec:
            attributes["quat"] = _format_numbers(_tuple_of_finite_numbers(spec["quat_wxyz"], 4, f"{name}.quat_wxyz"))
        ET.SubElement(body, "geom", attributes)
        groups = spec["contact_groups"]
        _require(
            isinstance(groups, list) and bool(groups),
            f"collision {name} needs groups",
            PatchSchemaError,
        )
        names.add(name)
    return names


def _apply_contact_groups(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    declared_collision_names: set[str],
) -> dict[str, Mapping[str, Any]]:
    groups = bundle.contacts["groups"]
    _require(isinstance(groups, dict), "groups must be a mapping", PatchSchemaError)
    body_names = set(_body_map(root))
    planned_scene_geoms = {
        str(bundle.scene["table"]["top_geom"]),
        *(f"{bundle.scene['table']['leg_geom_prefix']}_{index}" for index in range(4)),
        str(bundle.scene["cylinder"]["geom"]),
    }
    known_geoms = declared_collision_names | planned_scene_geoms
    result: dict[str, Mapping[str, Any]] = {}
    for name, raw in groups.items():
        spec = _require_keys(
            raw,
            required={"kind", "members"},
            allowed={"kind", "members"},
            context=f"contact group {name}",
        )
        kind = str(spec["kind"])
        members = spec["members"]
        _require(
            isinstance(members, list) and bool(members),
            f"contact group {name} is empty",
            PatchSchemaError,
        )
        _require(len(members) == len(set(members)), f"contact group {name} contains duplicates", PatchSchemaError)
        if kind == "geom":
            unknown = sorted(set(members) - known_geoms)
        elif kind == "body":
            unknown = sorted(set(members) - body_names)
        elif kind == "collision_mask":
            unknown = [] if members == ["robot"] else list(members)
        else:
            raise PatchSchemaError(f"contact group {name} has unsupported kind {kind!r}")
        if unknown:
            raise UnknownPatchTargetError(f"contact group {name} has unknown {kind} members: {unknown}")
        _custom_text(root, tracker, f"contact_group.{name}", ",".join(map(str, members)), "contact_groups.yaml")
        result[str(name)] = {"kind": kind, "members": list(map(str, members))}
    observation = result.get("right_hand_observation_links")
    _require(observation is not None and len(observation["members"]) == 12, "exactly 12 observation links are required")
    for phase_name, raw in bundle.contacts["phase_rules"].items():
        spec = _require_keys(
            raw,
            required={"allowed_pairs", "illegal_robot_groups"},
            allowed={"allowed_pairs", "illegal_robot_groups"},
            context=f"phase rule {phase_name}",
        )
        referenced = set(spec["illegal_robot_groups"])
        for pair in spec["allowed_pairs"]:
            _require(
                isinstance(pair, list) and len(pair) == 2, "allowed contact pair must have two groups", PatchSchemaError
            )
            referenced.update(pair)
        unknown_groups = sorted(referenced - set(result))
        if unknown_groups:
            raise UnknownPatchTargetError(f"phase {phase_name} names unknown contact groups: {unknown_groups}")
        _custom_text(
            root,
            tracker,
            f"contact_phase.{phase_name}",
            json.dumps(spec, sort_keys=True, separators=(",", ":")),
            "contact_groups.yaml",
        )
    return result


def _site_attributes(spec: Mapping[str, Any], context: str) -> dict[str, str]:
    valid_types = {"sphere", "capsule", "box", "cylinder"}
    site_type = str(spec["type"])
    _require(site_type in valid_types, f"{context} has unsupported type {site_type}", PatchSchemaError)
    size = spec["size_m"]
    _require(
        isinstance(size, list) and bool(size),
        f"{context}.size_m must be a list",
        PatchSchemaError,
    )
    attributes = {
        "name": str(spec["name"]),
        "type": site_type,
        "size": _format_numbers(float(item) for item in size),
        "rgba": _format_numbers(_tuple_of_finite_numbers(spec["rgba"], 4, f"{context}.rgba")),
        "group": "4",
    }
    if "pos_m" in spec:
        attributes["pos"] = _format_numbers(_tuple_of_finite_numbers(spec["pos_m"], 3, f"{context}.pos_m"))
    if "fromto_m" in spec:
        attributes["fromto"] = _format_numbers(_tuple_of_finite_numbers(spec["fromto_m"], 6, f"{context}.fromto_m"))
    return attributes


def _apply_sites(root: ET.Element, bundle: PatchBundle, tracker: PatchTracker) -> list[Mapping[str, Any]]:
    frames: dict[str, tuple[ET.Element, str]] = {}
    declarations: list[Mapping[str, Any]] = []
    for index, raw in enumerate(bundle.sites["frames"]):
        spec = _require_keys(
            raw,
            required={"name", "body", "pos_m", "quaternion_wxyz"},
            allowed={"name", "body", "pos_m", "quaternion_wxyz"},
            context=f"frame[{index}]",
        )
        name = str(spec["name"])
        tracker.claim("frame", name, "sites.yaml", spec)
        body_name = str(spec["body"])
        parent = _find_body(root, body_name, "sites.yaml")
        frame = ET.SubElement(
            parent,
            "frame",
            {
                "name": name,
                "pos": _format_numbers(_tuple_of_finite_numbers(spec["pos_m"], 3, f"frame {name}.pos_m")),
                "quat": _format_numbers(
                    _tuple_of_finite_numbers(spec["quaternion_wxyz"], 4, f"frame {name}.quaternion_wxyz")
                ),
            },
        )
        frames[name] = (frame, body_name)
        declarations.append({"kind": "frame", **dict(spec)})
    for index, raw in enumerate(bundle.sites["sites"]):
        required = {"name", "body", "type", "size_m", "rgba"}
        allowed = required | {"pos_m", "fromto_m", "frame"}
        spec = _require_keys(raw, required=required, allowed=allowed, context=f"site[{index}]")
        name = str(spec["name"])
        tracker.claim("site", name, "sites.yaml", spec)
        body_name = str(spec["body"])
        parent = _find_body(root, body_name, "sites.yaml")
        frame_name = spec.get("frame")
        if frame_name is not None:
            if frame_name not in frames:
                raise UnknownPatchTargetError(f"site {name} targets unknown frame {frame_name!r}")
            frame, frame_body = frames[str(frame_name)]
            _require(frame_body == body_name, f"site {name} frame/body mismatch", PatchSchemaError)
            parent = frame
        ET.SubElement(parent, "site", _site_attributes(spec, f"site {name}"))
        declarations.append({"kind": "site", **dict(spec)})
    return declarations


def _safe_ctrl_range(joint: SourceJoint, inventory: SourceInventory) -> tuple[float, float]:
    lower, upper = joint.lower, joint.upper
    for follower, (leader, multiplier, offset) in inventory.mimic.items():
        if leader != joint.name:
            continue
        follower_joint = inventory.joint_by_name[follower]
        if multiplier > 0.0:
            lower = max(lower, (follower_joint.lower - offset) / multiplier)
            upper = min(upper, (follower_joint.upper - offset) / multiplier)
        elif multiplier < 0.0:
            lower = max(lower, (follower_joint.upper - offset) / multiplier)
            upper = min(upper, (follower_joint.lower - offset) / multiplier)
    _require(lower < upper, f"mimic constraints empty ctrl range for {joint.name}")
    return lower, upper


def _apply_actuators(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    inventory: SourceInventory,
) -> list[str]:
    spec = bundle.main["actuators"]
    _require(spec["source"] == "m01.physics.actuator_calibration", "actuators must come from M01", PatchSchemaError)
    calibration = config.physics.actuator_calibration
    _require(calibration.control_mode == spec["expected_control_mode"] == "position", "position control required")
    size = _ensure_section(root, "size")
    tracker.set_attribute(size, "nuser_actuator", "4", source="actuators", target="size")
    actuator_section = _ensure_section(root, "actuator")
    names: list[str] = []
    for role, group in (("arm", calibration.arm), ("hand", calibration.hand)):
        expected_count = int(spec[f"{role}_count" if role == "arm" else "hand_active_count"])
        _require(len(group.joint_names) == expected_count, f"{role} actuator count mismatch")
        for values in zip(
            group.joint_names,
            group.stiffness_n_m_per_rad,
            group.damping_n_m_s_per_rad,
            group.effort_limit_n_m,
            group.velocity_limit_rad_s,
            strict=True,
        ):
            joint_name, stiffness, damping, effort, velocity = values
            joint = inventory.joint_by_name.get(joint_name)
            if joint is None:
                raise UnknownPatchTargetError(f"actuator targets unknown joint {joint_name!r}")
            _require(effort <= joint.effort, f"{joint_name} configured effort exceeds URDF")
            _require(velocity <= joint.velocity, f"{joint_name} configured velocity exceeds URDF")
            lower, upper = _safe_ctrl_range(joint, inventory)
            name = f"{joint_name}_position"
            tracker.claim("actuator", name, "M01 actuator calibration", values)
            ET.SubElement(
                actuator_section,
                "position",
                {
                    "name": name,
                    "joint": joint_name,
                    "kp": _format_number(stiffness),
                    "kv": _format_number(damping),
                    "ctrllimited": "true",
                    "ctrlrange": _format_numbers((lower, upper)),
                    "forcelimited": "true",
                    "forcerange": _format_numbers((-effort, effort)),
                    "user": _format_numbers((velocity, effort, stiffness, damping)),
                },
            )
            _custom_numeric(
                root,
                tracker,
                f"actuator.velocity_limit_rad_s.{name}",
                (velocity,),
                "M01 actuator calibration",
            )
            names.append(name)
    _require(len(names) == 13, "runtime MJCF must expose 7 arm + 6 hand actuators")
    _custom_text(root, tracker, "actuator.calibration_version", calibration.version, "M01 actuator calibration")
    _custom_text(root, tracker, "actuator.calibration_status", calibration.status, "M01 actuator calibration")
    return names


def _apply_mimic(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    inventory: SourceInventory,
) -> dict[str, tuple[str, float, float]]:
    spec = _require_keys(
        bundle.main["mimic"],
        required={
            "source",
            "side",
            "equality_type",
            "constraint_calibration_version",
            "constraint_calibration_status",
            "solref_direct",
            "solimp",
            "follower_damping_source",
            "validation_motion",
        },
        allowed={
            "source",
            "side",
            "equality_type",
            "constraint_calibration_version",
            "constraint_calibration_status",
            "solref_direct",
            "solimp",
            "follower_damping_source",
            "validation_motion",
        },
        context="mimic",
    )
    _require(spec["source"] == "urdf.joint.mimic", "mimic must come from URDF", PatchSchemaError)
    _require(spec["equality_type"] == "joint", "mimic must use MuJoCo joint equality", PatchSchemaError)
    _require(
        spec["follower_damping_source"] == "m01.physics.actuator_calibration.hand",
        "mimic follower damping must come from M01",
        PatchSchemaError,
    )
    solref = _tuple_of_finite_numbers(spec["solref_direct"], 2, "mimic.solref_direct")
    solimp = _tuple_of_finite_numbers(spec["solimp"], 3, "mimic.solimp")
    _require(all(value < 0.0 for value in solref), "mimic solref must use direct negative stiffness/damping")
    _require(all(0.0 < value < 1.0 for value in solimp), "mimic solimp values must lie in (0, 1)")
    side = f"{spec['side']}_"
    relationships = {follower: values for follower, values in inventory.mimic.items() if follower.startswith(side)}
    _require(len(relationships) == 6, "expected six right-hand mimic followers")
    equality = _ensure_section(root, "equality")
    hand_damping = dict(
        zip(
            config.physics.actuator_calibration.hand.joint_names,
            config.physics.actuator_calibration.hand.damping_n_m_s_per_rad,
            strict=True,
        )
    )
    joint_elements = _joint_map(root)
    for follower, (leader, multiplier, offset) in relationships.items():
        active_leader = leader
        while active_leader in relationships:
            active_leader = relationships[active_leader][0]
        _require(active_leader in hand_damping, f"mimic chain for {follower} has no active M01 leader")
        tracker.set_attribute(
            joint_elements[follower],
            "damping",
            _format_number(hand_damping[active_leader]),
            source="M01 hand damping for mimic follower",
            target=f"joint:{follower}",
        )
        name = f"mimic__{follower}"
        tracker.claim("equality", name, "URDF mimic", (follower, leader, multiplier, offset))
        ET.SubElement(
            equality,
            "joint",
            {
                "name": name,
                "joint1": follower,
                "joint2": leader,
                "polycoef": _format_numbers((offset, multiplier, 0.0, 0.0, 0.0)),
                "solref": _format_numbers(solref),
                "solimp": _format_numbers(solimp),
            },
        )
    _custom_text(root, tracker, "asset.right_hand_mimic_source", "source/tiangong2.0_pro_with_hands.urdf", "URDF mimic")
    _custom_text(
        root,
        tracker,
        "asset.right_hand_mimic_constraint_calibration",
        str(spec["constraint_calibration_version"]),
        "URDF mimic",
    )
    return relationships


def _add_materials(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
) -> None:
    asset = _ensure_section(root, "asset")
    rgba_by_role = {
        "robot": (0.75, 0.75, 0.75, 1.0),
        "table": tuple(float(item) for item in bundle.scene["table"]["rgba"]),
        "cylinder": tuple(float(item) for item in bundle.scene["cylinder"]["rgba"]),
    }
    for role, material in (
        ("robot", config.physics.robot_material),
        ("table", config.physics.table_material),
        ("cylinder", config.physics.cylinder_material),
    ):
        tracker.claim("material", material.name, "M01 physics materials", asdict(material))
        ET.SubElement(asset, "material", {"name": material.name, "rgba": _format_numbers(rgba_by_role[role])})
        _custom_numeric(
            root,
            tracker,
            f"material.restitution.{material.name}",
            (material.restitution,),
            "M01 physics materials",
        )


def _scene_geom_attributes(config: ResolvedRunConfig, material_name: str, mask: Mapping[str, Any]) -> dict[str, str]:
    attributes = _contact_attributes(config, material_name=material_name, mask=mask)
    attributes.pop("density")
    return attributes


def _apply_floor(root: ET.Element, bundle: PatchBundle, tracker: PatchTracker) -> Mapping[str, Any]:
    """Add rmm-sim's checkerboard floor as a display-only MuJoCo plane.

    The source scene uses MuJoCo's builtin checker texture, so there is no
    external image to copy.  Keeping the plane non-colliding preserves this
    project's existing robot/table contact contract while matching the
    source viewer appearance.
    """

    raw = _require_keys(
        bundle.scene.get("floor"),
        required={"geom", "texture", "material", "type", "size_m", "pos_m", "visual_only", "checker"},
        allowed={"geom", "texture", "material", "type", "size_m", "pos_m", "visual_only", "checker"},
        context="scene.floor",
    )
    checker = _require_keys(
        raw["checker"],
        required={
            "builtin",
            "mark",
            "rgb1",
            "rgb2",
            "markrgb",
            "width",
            "height",
            "texuniform",
            "texrepeat",
            "reflectance",
        },
        allowed={
            "builtin",
            "mark",
            "rgb1",
            "rgb2",
            "markrgb",
            "width",
            "height",
            "texuniform",
            "texrepeat",
            "reflectance",
        },
        context="scene.floor.checker",
    )
    geom_name = str(raw["geom"])
    texture_name = str(raw["texture"])
    material_name = str(raw["material"])
    _require(raw["visual_only"] is True, "scene.floor must be visual_only", PatchSchemaError)
    _require(str(raw["type"]) == "plane", "scene.floor.type must be plane", PatchSchemaError)
    size = _tuple_of_finite_numbers(raw["size_m"], 3, "scene.floor.size_m")
    position = _tuple_of_finite_numbers(raw["pos_m"], 3, "scene.floor.pos_m")
    colors = {
        key: _format_numbers(_tuple_of_finite_numbers(checker[key], 3, f"scene.floor.checker.{key}"))
        for key in ("rgb1", "rgb2", "markrgb")
    }
    width = int(checker["width"])
    height = int(checker["height"])
    _require(width > 0 and height > 0, "scene.floor checker dimensions must be positive", PatchSchemaError)
    repeat = _tuple_of_finite_numbers(checker["texrepeat"], 2, "scene.floor.checker.texrepeat")
    _require(all(value > 0.0 for value in repeat), "scene.floor checker repeat must be positive", PatchSchemaError)
    reflectance = float(checker["reflectance"])
    _require(
        math.isfinite(reflectance) and 0.0 <= reflectance <= 1.0, "scene.floor reflectance is invalid", PatchSchemaError
    )

    asset = _ensure_section(root, "asset")
    tracker.claim("texture", texture_name, "scene.yaml", checker)
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": texture_name,
            "builtin": str(checker["builtin"]),
            "mark": str(checker["mark"]),
            "rgb1": colors["rgb1"],
            "rgb2": colors["rgb2"],
            "markrgb": colors["markrgb"],
            "width": str(width),
            "height": str(height),
        },
    )
    tracker.claim("material", material_name, "scene.yaml", raw)
    ET.SubElement(
        asset,
        "material",
        {
            "name": material_name,
            "texture": texture_name,
            "texuniform": "true" if bool(checker["texuniform"]) else "false",
            "texrepeat": _format_numbers(repeat),
            "reflectance": _format_number(reflectance),
        },
    )
    worldbody = _not_none(root.find("worldbody"))
    tracker.claim("geom", geom_name, "scene.yaml", raw)
    geom = ET.Element(
        "geom",
        {
            "name": geom_name,
            "type": "plane",
            "size": _format_numbers(size),
            "pos": _format_numbers(position),
            "material": material_name,
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "1",
        },
    )
    worldbody.insert(0, geom)
    return {
        "geom": geom_name,
        "texture": texture_name,
        "material": material_name,
        "source": "rmm-sim/mj_assets/stanford_tidybot2/scene.xml",
        "visual_only": True,
        "checker": {
            "builtin": str(checker["builtin"]),
            "mark": str(checker["mark"]),
            "rgb1": tuple(
                float(value) for value in _tuple_of_finite_numbers(checker["rgb1"], 3, "scene.floor.checker.rgb1")
            ),
            "rgb2": tuple(
                float(value) for value in _tuple_of_finite_numbers(checker["rgb2"], 3, "scene.floor.checker.rgb2")
            ),
            "markrgb": tuple(
                float(value) for value in _tuple_of_finite_numbers(checker["markrgb"], 3, "scene.floor.checker.markrgb")
            ),
            "width": width,
            "height": height,
            "texuniform": bool(checker["texuniform"]),
            "texrepeat": repeat,
            "reflectance": reflectance,
        },
    }


def _apply_scene(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
) -> Mapping[str, Any]:
    _add_materials(root, bundle, tracker, config)
    floor_summary = _apply_floor(root, bundle, tracker)
    physics = config.physics
    option = _ensure_section(root, "option")
    solver_names = {"newton": "Newton", "cg": "CG", "pgs": "PGS"}
    integrator_names = {
        "implicitfast": "implicitfast",
        "implicit": "implicit",
        "rk4": "RK4",
        "euler": "Euler",
    }
    option_values = {
        "timestep": _format_number(physics.physics_dt_s),
        "gravity": _format_numbers(physics.gravity_m_s2),
        "integrator": integrator_names[physics.solver.integrator],
        "solver": solver_names[physics.solver.solver],
        "iterations": str(physics.solver.position_iterations),
        "noslip_iterations": str(physics.solver.no_slip_iterations),
        "ls_iterations": str(physics.solver.line_search_iterations),
        "tolerance": _format_number(physics.solver.tolerance),
    }
    for attribute, value in option_values.items():
        tracker.set_attribute(option, attribute, value, source="M01 physics", target="option")
    size = _ensure_section(root, "size")
    tracker.set_attribute(
        size,
        "nconmax",
        str(physics.contact.maximum_contacts),
        source="M01 contact",
        target="size",
    )
    _custom_text(root, tracker, "physics.version", physics.version, "M01 physics")
    _custom_text(root, tracker, "physics.calibration_status", physics.calibration_status, "M01 physics")

    worldbody = _not_none(root.find("worldbody"))
    masks = bundle.scene["collision_masks"]
    table_spec = bundle.scene["table"]
    scene = config.task.scene
    table_body_name = str(table_spec["body"])
    tracker.claim("body", table_body_name, "scene.yaml", table_spec)
    table_body = ET.SubElement(worldbody, "body", {"name": table_body_name})
    table_size = tuple(scene.table_size_m)
    table_center = tuple(scene.table_center_m)
    top_name = str(table_spec["top_geom"])
    tracker.claim("geom", top_name, "scene.yaml", table_spec)
    ET.SubElement(
        table_body,
        "geom",
        {
            "name": top_name,
            "type": "box",
            "pos": _format_numbers(table_center),
            "size": _format_numbers(value / 2.0 for value in table_size),
            **_scene_geom_attributes(config, physics.table_material.name, masks["table"]),
        },
    )
    leg_width = float(table_spec["leg_width_m"])
    underside_z = table_center[2] - table_size[2] / 2.0
    _require(underside_z > 0.0 and leg_width > 0.0, "invalid table leg geometry", PatchSchemaError)
    x_offset = table_size[0] / 2.0 - leg_width / 2.0
    y_offset = table_size[1] / 2.0 - leg_width / 2.0
    leg_names: list[str] = []
    for index, (dx, dy) in enumerate(
        ((-x_offset, -y_offset), (-x_offset, y_offset), (x_offset, -y_offset), (x_offset, y_offset))
    ):
        name = f"{table_spec['leg_geom_prefix']}_{index}"
        tracker.claim("geom", name, "scene.yaml", (dx, dy))
        ET.SubElement(
            table_body,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": _format_numbers((table_center[0] + dx, table_center[1] + dy, underside_z / 2.0)),
                "size": _format_numbers((leg_width / 2.0, leg_width / 2.0, underside_z / 2.0)),
                **_scene_geom_attributes(config, physics.table_material.name, masks["table"]),
            },
        )
        leg_names.append(name)

    cylinder_spec = bundle.scene["cylinder"]
    shape = str(cylinder_spec.get("shape", "cylinder"))
    _require(shape in {"cylinder", "square_prism"}, f"unsupported scene object shape {shape}", PatchSchemaError)
    cylinder_name = str(cylinder_spec["body"])
    tracker.claim("body", cylinder_name, "scene.yaml", cylinder_spec)
    randomization = config.randomization
    radius = (randomization.radius.minimum + randomization.radius.maximum) / 2.0
    height = (randomization.height.minimum + randomization.height.maximum) / 2.0
    mass = (randomization.mass.minimum + randomization.mass.maximum) / 2.0
    top_z = table_center[2] + table_size[2] / 2.0
    cylinder_position = (
        scene.cylinder_origin_xy_m[0],
        scene.cylinder_origin_xy_m[1],
        top_z + scene.cylinder_table_clearance_m + height / 2.0,
    )
    cylinder_body = ET.SubElement(
        worldbody,
        "body",
        {"name": cylinder_name, "pos": _format_numbers(cylinder_position)},
    )
    joint_name = str(cylinder_spec["joint"])
    tracker.claim("joint", joint_name, "scene.yaml", cylinder_spec)
    ET.SubElement(cylinder_body, "freejoint", {"name": joint_name})
    geom_name = str(cylinder_spec["geom"])
    tracker.claim("geom", geom_name, "scene.yaml", cylinder_spec)
    half_side = radius / math.sqrt(2.0)
    geom_size = (half_side, half_side, height / 2.0) if shape == "square_prism" else (radius, height / 2.0)
    cylinder_attributes = {
        **_scene_geom_attributes(config, physics.cylinder_material.name, masks["cylinder"]),
    }
    if "contact_condim" in cylinder_spec:
        condim = int(cylinder_spec["contact_condim"])
        _require(condim in {1, 3, 4, 6}, "scene.cylinder.contact_condim must be 1, 3, 4 or 6", PatchSchemaError)
        cylinder_attributes["condim"] = str(condim)
    ET.SubElement(
        cylinder_body,
        "geom",
        {
            "name": geom_name,
            "type": "box" if shape == "square_prism" else "cylinder",
            "size": _format_numbers(geom_size),
            "mass": _format_number(mass),
            # The task object should not create an early contact shell around
            # itself; hand mesh geoms also use margin=0.
            **cylinder_attributes,
        },
    )
    site_name = "cylinder_axis_site"
    tracker.claim("site", site_name, "scene.yaml", cylinder_spec)
    ET.SubElement(
        cylinder_body,
        "site",
        {
            "name": site_name,
            "type": "cylinder",
            "size": _format_numbers((radius * 0.2, height / 2.0)),
            "rgba": "1 1 0 0.5",
            "group": "4",
        },
    )
    return {
        "floor": floor_summary,
        "table_body": table_body_name,
        "table_geoms": [top_name, *leg_names],
        "cylinder_body": cylinder_name,
        "cylinder_joint": joint_name,
        "cylinder_geom": geom_name,
        "cylinder_nominal_radius_m": radius,
        "cylinder_nominal_height_m": height,
        "cylinder_nominal_mass_kg": mass,
        "cylinder_initial_position_m": cylinder_position,
    }


def _add_home_keyframe(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    roles: Mapping[str, str],
    fixed_positions: Mapping[str, float],
    scene_summary: Mapping[str, Any],
) -> None:
    arm_home = dict(zip(config.task.action.arm_joint_names, config.task.action.center_joint_positions_rad, strict=True))
    qpos: list[float] = []
    worldbody = _not_none(root.find("worldbody"))
    for element in worldbody.iter():
        if element.tag == "joint":
            name = _not_none(element.get("name"))
            if name in roles:
                qpos.append(float(arm_home.get(name, fixed_positions.get(name, 0.0))))
            elif element.get("type") == "free":
                qpos.extend((*scene_summary["cylinder_initial_position_m"], 1.0, 0.0, 0.0, 0.0))
            else:
                raise PatchSchemaError(f"home keyframe has unknown joint {name}")
        elif element.tag == "freejoint":
            name = _not_none(element.get("name"))
            _require(name == scene_summary["cylinder_joint"], f"unknown freejoint {name}", PatchSchemaError)
            qpos.extend((*scene_summary["cylinder_initial_position_m"], 1.0, 0.0, 0.0, 0.0))
    _require(len(qpos) == EXPECTED_SOURCE_MOVABLE_JOINTS + 7, f"home qpos has wrong length {len(qpos)}")
    name = str(bundle.main["joint_posture"]["home_keyframe"])
    tracker.claim("key", name, "joint_posture", qpos)
    keyframe = _ensure_section(root, "keyframe")
    actuator_targets = [
        float(arm_home.get(joint_name, 0.0))
        for joint_name in (
            *config.physics.actuator_calibration.arm.joint_names,
            *config.physics.actuator_calibration.hand.joint_names,
        )
    ]
    _require(len(actuator_targets) == 13, "home keyframe must define 13 actuator targets")
    ET.SubElement(
        keyframe,
        "key",
        {
            "name": name,
            "qpos": _format_numbers(qpos),
            "ctrl": _format_numbers(actuator_targets),
        },
    )


def _apply_metadata(
    root: ET.Element,
    bundle: PatchBundle,
    tracker: PatchTracker,
    config: ResolvedRunConfig,
    roles: Mapping[str, str],
) -> None:
    _custom_text(root, tracker, "asset.generator_version", GENERATOR_VERSION, "metadata")
    _custom_text(root, tracker, "asset.patch_version", str(bundle.main["patch_version"]), "metadata")
    _custom_text(root, tracker, "asset.contract_version", config.asset.asset_version, "metadata")
    _custom_text(root, tracker, "asset.source_joint_hierarchy", JOINT_HIERARCHY_NAME, "metadata")
    for role in ("active_arm", "active_hand", "mimic_follower", "fixed_non_task"):
        names = [name for name, joint_role in roles.items() if joint_role == role]
        _custom_text(root, tracker, f"asset.joint_role.{role}", ",".join(names), "metadata")


def _canonicalize_xml(root: ET.Element) -> bytes:
    preferred = {
        name: index
        for index, name in enumerate(
            (
                "name",
                "class",
                "type",
                "joint",
                "joint1",
                "joint2",
                "body",
                "pos",
                "quat",
                "axis",
                "fromto",
                "size",
                "range",
                "ctrlrange",
                "forcerange",
                "kp",
                "kv",
            )
        )
    }
    for element in root.iter():
        attributes = sorted(element.attrib.items(), key=lambda item: (preferred.get(item[0], len(preferred)), item[0]))
        element.attrib.clear()
        element.attrib.update(attributes)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True) + b"\n"


def _remove_visual_meshes(root: ET.Element) -> None:
    """Remove only STL assets and mesh geoms from a headless runtime tree.

    MuJoCo loads and preprocesses every declared STL before applying the
    ``discardvisual`` compiler option.  The archived meshes are exceptionally
    dense, so merely setting that option still retains gigabytes of compiler
    allocation per independently loaded model.  The committed viewer MJCF is
    the byte-traceable, full-appearance product; this derived runtime tree
    removes exactly the visual-only declarations and keeps all physical XML
    byte-equivalent after those declarations are ignored.
    """

    asset = root.find("asset")
    if asset is None:
        raise AssetPipelineError("patched viewer MJCF has no asset section")
    meshes = [element for element in list(asset) if element.tag == "mesh"]
    _require(len(meshes) == EXPECTED_MESHES, f"expected {EXPECTED_MESHES} visual mesh assets, found {len(meshes)}")
    worldbody = _not_none(root.find("worldbody"))
    physical_mesh_names = {
        geom.get("mesh")
        for geom in worldbody.iter("geom")
        if geom.get("type") == "mesh" and (geom.get("contype") != "0" or geom.get("conaffinity") != "0")
    }
    _require(None not in physical_mesh_names, "collision mesh geom is missing its mesh asset name")
    for element in meshes:
        if element.get("name") not in physical_mesh_names:
            asset.remove(element)

    visual_mesh_geom_count = sum(
        1
        for geom in worldbody.iter("geom")
        if geom.get("type") == "mesh" and geom.get("contype") == "0" and geom.get("conaffinity") == "0"
    )
    removed_geoms = 0
    for body in worldbody.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("type") != "mesh":
                continue
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                body.remove(geom)
                removed_geoms += 1
    _require(
        removed_geoms == visual_mesh_geom_count,
        f"expected to remove {visual_mesh_geom_count} visual mesh geoms, removed {removed_geoms}",
    )
    for geom in worldbody.iter("geom"):
        if geom.get("type") == "mesh":
            _require(
                geom.get("contype") != "0" or geom.get("conaffinity") != "0",
                f"headless runtime retained a visual-only mesh geom {geom.get('name')!r}",
            )


def _compile_final(
    xml_bytes: bytes,
    inventory: SourceInventory,
    expected_version: str,
    *,
    include_visual_assets: bool,
) -> Any:
    import mujoco

    _require(mujoco.__version__ == expected_version, "MuJoCo version changed during generation")
    assets = {path.name: path.read_bytes() for path in inventory.mesh_paths} if include_visual_assets else None
    try:
        return mujoco.MjModel.from_xml_string(xml_bytes.decode("utf-8"), assets)
    except ValueError as exc:
        raise AssetPipelineError(f"generated runtime MJCF does not compile: {exc}") from exc


def _home_robot_table_clearance(
    model: Any,
    contact_groups: Mapping[str, Mapping[str, Any]],
) -> float:
    """Return the minimum signed home clearance across task robot/table geoms."""

    import mujoco
    import numpy as np

    # The hand can intentionally rest close to the tabletop in the home pose;
    # the clearance gate is for the arm and TCP, whose table penetration would
    # destabilize every reset.
    robot_group_names = ("right_arm", "tcp")
    robot_geoms = {str(name) for group_name in robot_group_names for name in contact_groups[group_name]["members"]}
    table_geoms = {str(name) for name in contact_groups["table"]["members"]}
    data = mujoco.MjData(model)
    _require(model.nkey == 1, "runtime model must expose exactly one home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    fromto = np.zeros(6, dtype=np.float64)
    clearances = []
    for robot_name in sorted(robot_geoms):
        robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, robot_name)
        for table_name in sorted(table_geoms):
            table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, table_name)
            clearances.append(float(mujoco.mj_geomDistance(model, data, robot_id, table_id, 2.0, fromto)))
    _require(bool(clearances), "home-clearance validation found no robot/table geom pairs")
    return min(clearances)


def _runtime_names(model: Any, object_type: Any, count: int) -> list[str]:
    import mujoco

    names: list[str] = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        _require(name is not None, f"runtime object {object_type} id {index} is unnamed")
        names.append(name)
    return names


def _build_name_index(
    model: Any,
    site_declarations: Sequence[Mapping[str, Any]],
    contact_groups: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    import mujoco

    kinds = {
        "bodies": (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joints": (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "geoms": (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        "sites": (mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "actuators": (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "equalities": (mujoco.mjtObj.mjOBJ_EQUALITY, model.neq),
        "keys": (mujoco.mjtObj.mjOBJ_KEY, model.nkey),
    }
    indices: dict[str, dict[str, int]] = {}
    for label, (object_type, count) in kinds.items():
        names = _runtime_names(model, object_type, count)
        _require(len(names) == len(set(names)), f"runtime {label} contain duplicate names")
        indices[label] = {name: index for index, name in enumerate(names)}
    frames = {
        str(item["name"]): {
            "body": str(item["body"]),
            "runtime_object": None,
            "note": "MJCF frame is a compile-time coordinate transform; named sites provide runtime lookup",
        }
        for item in site_declarations
        if item["kind"] == "frame"
    }
    return {
        "schema_version": "tienkung2_pro_name_index_v1",
        "indices": indices,
        "declared_frames": frames,
        "contact_groups": contact_groups,
    }


def _build_joint_hierarchy(inventory: SourceInventory, roles: Mapping[str, str], model: Any) -> Mapping[str, Any]:
    import mujoco

    runtime_joint_names = _runtime_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    runtime_ids = {name: index for index, name in enumerate(runtime_joint_names)}
    entries = []
    for joint in inventory.joints:
        _require(joint.name in runtime_ids, f"runtime lost source joint {joint.name}")
        entries.append(
            {**asdict(joint), "runtime_joint_id": runtime_ids[joint.name], "runtime_role": roles[joint.name]}
        )
    return {
        "schema_version": "tienkung2_pro_joint_hierarchy_v1",
        "source": f"source/{inventory.urdf_path.name}",
        "movable_joint_count": len(entries),
        "joints": entries,
    }


def _patch_hashes(bundle: PatchBundle) -> Mapping[str, str]:
    return {path.name: sha256_file(path) for path in sorted(bundle.file_paths, key=lambda item: item.name)}


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def generate_assets(
    config: ResolvedRunConfig,
    *,
    output_root: Path = ASSET_ROOT,
    urdf_path: Path | None = None,
    mesh_dir: Path | None = None,
    patch_dir: Path = PATCH_DIR,
) -> GenerationResult:
    """Generate every committed M02 output into ``output_root``."""

    source_urdf = urdf_path or REPOSITORY_ROOT / config.asset.source_urdf_path
    source_meshes = mesh_dir or REPOSITORY_ROOT / config.asset.visual_mesh_directory
    inventory = inspect_source_assets(source_urdf, source_meshes)
    bundle = load_patch_bundle(patch_dir)
    _require(
        bundle.main["asset_contract_version"] == config.asset.asset_version,
        "patch asset contract differs from M01 AssetConfig",
        PatchSchemaError,
    )
    source_manifest_bytes = _json_bytes(build_source_manifest(inventory))
    base_root = _import_urdf_with_mujoco(inventory, config.platform.mujoco_version)
    base_bytes = _canonicalize_xml(base_root)

    viewer_root = ET.fromstring(base_bytes)
    viewer_root.set("model", "tienkung2_pro_reach_then_close")
    tracker = PatchTracker()
    _apply_fixed_base(viewer_root, bundle, tracker)
    roles, fixed_positions = _apply_joint_posture(viewer_root, bundle, tracker, config, inventory)
    gravity_compensated_bodies = _apply_gravity_compensation(
        viewer_root,
        bundle,
        tracker,
        config,
        inventory,
    )
    _apply_visual_only(viewer_root, _mesh_collision_bodies(bundle))
    collision_names = _apply_collisions(viewer_root, bundle, tracker, config)
    contact_groups = _apply_contact_groups(viewer_root, bundle, tracker, collision_names)
    site_declarations = _apply_sites(viewer_root, bundle, tracker)
    actuator_names = _apply_actuators(viewer_root, bundle, tracker, config, inventory)
    mimic = _apply_mimic(viewer_root, bundle, tracker, config, inventory)
    scene_summary = _apply_scene(viewer_root, bundle, tracker, config)
    _add_home_keyframe(
        viewer_root,
        bundle,
        tracker,
        config,
        roles,
        fixed_positions,
        scene_summary,
    )
    _apply_metadata(viewer_root, bundle, tracker, config, roles)
    viewer_bytes = _canonicalize_xml(viewer_root)
    viewer_model = _compile_final(
        viewer_bytes,
        inventory,
        config.platform.mujoco_version,
        include_visual_assets=True,
    )

    runtime_root = ET.fromstring(viewer_bytes)
    _remove_visual_meshes(runtime_root)
    runtime_bytes = _canonicalize_xml(runtime_root)
    model = _compile_final(
        runtime_bytes,
        inventory,
        config.platform.mujoco_version,
        include_visual_assets=True,
    )
    for label, compiled in (("headless", model), ("viewer", viewer_model)):
        _require(compiled.nu == 13, f"compiled {label} model has {compiled.nu} actuators instead of 13")
        _require(
            compiled.njnt == EXPECTED_SOURCE_MOVABLE_JOINTS + 1,
            f"compiled {label} model joint count is wrong",
        )
        _require(compiled.nsite >= 5, f"compiled {label} model is missing required sites")
    for field_name in ("nbody", "njnt", "nq", "nv", "nsite", "nu", "neq", "nkey"):
        _require(
            int(getattr(model, field_name)) == int(getattr(viewer_model, field_name)),
            f"headless/viewer physical model mismatch for {field_name}",
        )
    mesh_collision_count = _physical_mesh_collision_count(bundle)
    _require(model.nmesh == mesh_collision_count, "headless runtime mesh collision count is wrong")
    _require(viewer_model.nmesh == EXPECTED_MESHES, "viewer runtime must compile all source visual meshes")
    minimum_home_clearance = float(bundle.main["joint_posture"]["minimum_robot_table_clearance_m"])
    measured_home_clearance = _home_robot_table_clearance(model, contact_groups)
    _require(
        measured_home_clearance >= minimum_home_clearance,
        f"home robot/table clearance {measured_home_clearance} is below required {minimum_home_clearance}",
        PatchSchemaError,
    )

    name_index_bytes = _json_bytes(_build_name_index(model, site_declarations, contact_groups))
    hierarchy_bytes = _json_bytes(_build_joint_hierarchy(inventory, roles, model))
    output_payloads = {
        f"source/{SOURCE_MANIFEST_NAME}": source_manifest_bytes,
        f"mjcf/{BASE_MJCF_NAME}": base_bytes,
        f"mjcf/{RUNTIME_MJCF_NAME}": runtime_bytes,
        f"mjcf/{VIEWER_MJCF_NAME}": viewer_bytes,
        f"mjcf/{NAME_INDEX_NAME}": name_index_bytes,
        f"mjcf/{JOINT_HIERARCHY_NAME}": hierarchy_bytes,
    }
    output_hashes = {name: _sha256_bytes(data) for name, data in output_payloads.items()}

    def summary(compiled: Any) -> dict[str, int]:
        return {
            "nbody": int(compiled.nbody),
            "njnt": int(compiled.njnt),
            "nq": int(compiled.nq),
            "nv": int(compiled.nv),
            "ngeom": int(compiled.ngeom),
            "nmesh": int(compiled.nmesh),
            "nsite": int(compiled.nsite),
            "nu": int(compiled.nu),
            "neq": int(compiled.neq),
            "nkey": int(compiled.nkey),
        }

    model_summary = summary(model)
    viewer_model_summary = summary(viewer_model)
    manifest = {
        "schema_version": "tienkung2_pro_generation_manifest_v2",
        "asset_version": config.asset.asset_version,
        "generator_version": GENERATOR_VERSION,
        "tool": {
            "name": "mujoco",
            "version": config.platform.mujoco_version,
            "urdf_import_options": URDF_IMPORT_OPTIONS,
            "base_numeric_format": "MuJoCo mj_saveLastXML 3.11.0 canonical output",
            "patch_numeric_format": FLOAT_FORMAT,
            "input_order": "UTF-8 path byte order",
            "generator_sha256": sha256_file(Path(__file__)),
        },
        "source_manifest_sha256": _sha256_bytes(source_manifest_bytes),
        "source_urdf_sha256": sha256_file(inventory.urdf_path),
        "visual_meshes_aggregate_sha256": _sha256_bytes(
            b"".join(f"{path.name}\0{sha256_file(path)}\n".encode() for path in inventory.mesh_paths)
        ),
        "patch_version": bundle.main["patch_version"],
        "patch_hashes": _patch_hashes(bundle),
        "collision_calibration": dict(bundle.collisions["calibration"]),
        "output_hashes": output_hashes,
        "source_contract": {
            "movable_joint_count": len(inventory.joints),
            "mimic_relation_count": len(inventory.mimic),
            "right_mimic_relation_count": len(mimic),
            "fixed_non_task_joint_count": len(fixed_positions),
            "active_arm_joint_count": len(config.physics.actuator_calibration.arm.joint_names),
            "active_hand_joint_count": len(config.physics.actuator_calibration.hand.joint_names),
            "gravity_compensated_root_body": gravity_compensated_bodies[0],
            "gravity_compensated_body_count": len(gravity_compensated_bodies),
            "visual_mesh_count": len(inventory.mesh_paths),
            "runtime_actuator_names": actuator_names,
        },
        "scene": {
            **scene_summary,
            "minimum_home_robot_table_clearance_m": minimum_home_clearance,
            "measured_home_robot_table_clearance_m": measured_home_clearance,
        },
        "runtime_products": {
            "headless": {
                "path": f"mjcf/{RUNTIME_MJCF_NAME}",
                "purpose": "training_evaluation_headless_independent_model",
                "visual_meshes_compiled": False,
                "collision_meshes_compiled": mesh_collision_count,
                "model_summary": model_summary,
            },
            "viewer": {
                "path": f"mjcf/{VIEWER_MJCF_NAME}",
                "purpose": "single_environment_full_visual_viewer",
                "visual_meshes_compiled": True,
                "collision_meshes_compiled": mesh_collision_count,
                "model_summary": viewer_model_summary,
            },
            "physical_equivalence": "viewer minus visual-only mesh assets/geoms equals headless",
        },
        "model_summary": model_summary,
        "calibration_status": config.asset.calibration_status,
    }
    manifest_bytes = _json_bytes(manifest)
    output_payloads[GENERATION_MANIFEST_NAME] = manifest_bytes
    hashes = {name: _sha256_bytes(data) for name, data in output_payloads.items()}
    for relative, data in output_payloads.items():
        _write(output_root / relative, data)
    return GenerationResult(output_root=output_root, hashes=hashes, model_summary=model_summary)


def check_generated_assets(config: ResolvedRunConfig, *, committed_root: Path = ASSET_ROOT) -> None:
    """Regenerate in a temporary directory and fail on any committed drift."""

    with tempfile.TemporaryDirectory(prefix="tienkung-asset-check-") as temporary:
        candidate_root = Path(temporary) / "tienkung2_pro"
        generate_assets(config, output_root=candidate_root)
        drift: list[str] = []
        for relative in GENERATED_RELATIVE_PATHS:
            expected = committed_root / relative
            actual = candidate_root / relative
            if not expected.is_file():
                drift.append(f"missing committed output: {relative.as_posix()}")
            elif expected.read_bytes() != actual.read_bytes():
                drift.append(
                    f"content drift: {relative.as_posix()} "
                    f"(committed={sha256_file(expected)}, regenerated={sha256_file(actual)})"
                )
        if drift:
            raise AssetPipelineError("generated asset check failed:\n" + "\n".join(drift))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministically generate or check the TienKung 2 Pro MuJoCo asset")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="write generated assets (default)")
    mode.add_argument("--check", action="store_true", help="regenerate in temporary storage and compare")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="M01 run configuration")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="strict typed M01 configuration override (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_run_config(args.config, overrides=tuple(args.overrides))
        if args.check:
            check_generated_assets(config)
            print(f"[PASS] generated assets match {ASSET_ROOT}")
        else:
            result = generate_assets(config)
            print(f"[PASS] generated {len(result.hashes)} deterministic outputs in {result.output_root}")
            for name, digest in sorted(result.hashes.items()):
                print(f"{digest}  {name}")
    except (AssetPipelineError, ValueError, OSError) as exc:
        parser.exit(1, f"asset pipeline error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
