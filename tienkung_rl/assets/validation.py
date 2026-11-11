"""Deterministic structural and dynamic validation for the committed asset.

The validator deliberately reads both the archived URDF and the compiled
MuJoCo model.  The generated hierarchy and name-index files are useful
evidence, but are never treated as an independent source of truth.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco  # type: ignore[import-untyped]  # MuJoCo 3.11 does not publish a PEP 561 marker.
import numpy as np
import yaml  # type: ignore[import-untyped]  # PyYAML does not publish type information.

from tienkung_rl.assets import pipeline
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import MujocoEnvironmentBackend, ResetSpec

REPORT_SCHEMA_VERSION = "tienkung_asset_validation_report_v1"
VALIDATOR_VERSION = "tienkung_asset_validator_v1"
AXIS_TOLERANCE = 1.0e-10
LIMIT_TOLERANCE_RAD = 1.0e-5
INERTIA_EIGENVALUE_TOLERANCE_KG_M2 = 1.0e-12
TRIANGLE_INEQUALITY_TOLERANCE_KG_M2 = 1.0e-10
INITIAL_PENETRATION_TOLERANCE_M = 1.0e-8
COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT = 0.20
SETTLE_DURATION_S = 2.0
FIXED_JOINT_SETTLE_TOLERANCE_RAD = 0.01
CYLINDER_HORIZONTAL_DRIFT_TOLERANCE_M = 0.002
CYLINDER_VERTICAL_DRIFT_TOLERANCE_M = 0.005
CYLINDER_TILT_TOLERANCE_RAD = math.radians(2.0)
ARM_TRACKING_TARGET_DELTA_RAD = 0.02
ARM_TRACKING_SETTLE_S = 1.0
ARM_TRACKING_OBSERVE_S = 1.0
ARM_TRACKING_STEADY_ERROR_LIMIT_RAD = 0.15
ARM_TRACKING_OVERSHOOT_LIMIT_RAD = 0.10
HAND_HOLD_DURATION_S = 1.0
HAND_ACTIVE_ERROR_LIMIT_RAD = 0.02
HAND_MIMIC_ERROR_LIMIT_RAD = 0.01
HAND_JOINT_LIMIT_TOLERANCE_RAD = 0.01
INITIAL_CONTACT_ALLOWLIST_VERSION = "initial_contact_allowlist_v1"
INITIAL_CONTACT_ALLOWLIST: Mapping[tuple[str, str], str] = {
    ("cylinder", "table"): "the cylinder is initialized on the table surface",
}

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """One stable, machine-readable validation diagnostic."""

    code: str
    severity: Severity
    message: str
    object_name: str | None = None
    explanation: str | None = None
    measured: float | int | str | bool | None = None
    expected: float | int | str | bool | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "object_name": self.object_name,
            "explanation": self.explanation,
            "measured": self.measured,
            "expected": self.expected,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ValidationCheck:
    """Result of one independently named validation check."""

    check_id: str
    description: str
    passed: bool
    issues: tuple[ValidationIssue, ...]
    measurements: Mapping[str, object] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "measurements": _json_ready(self.measurements),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Top-level report returned by both the Python and command-line APIs."""

    asset_version: str
    runtime_mjcf_sha256: str
    checks: tuple[ValidationCheck, ...]
    artifacts: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = REPORT_SCHEMA_VERSION
    validator_version: str = VALIDATOR_VERSION

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def error_count(self) -> int:
        return sum(check.error_count for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(check.warning_count for check in self.checks)

    @property
    def unexplained_warning_count(self) -> int:
        return sum(
            issue.severity == "warning" and not issue.explanation for check in self.checks for issue in check.issues
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
            "asset_version": self.asset_version,
            "runtime_mjcf_sha256": self.runtime_mjcf_sha256,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "unexplained_warning_count": self.unexplained_warning_count,
            "artifacts": dict(sorted(self.artifacts.items())),
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"

    def terminal_lines(self) -> tuple[str, ...]:
        lines = []
        for check in self.checks:
            state = "PASS" if check.passed else "FAIL"
            lines.append(
                f"[{state}] {check.check_id}: {check.description} "
                f"(errors={check.error_count}, warnings={check.warning_count})"
            )
            for issue in check.issues:
                lines.append(f"  {issue.severity.upper()} {issue.code}: {issue.message}")
        state = "PASS" if self.passed else "FAIL"
        lines.append(
            f"[{state}] asset validation: checks={len(self.checks)}, "
            f"errors={self.error_count}, warnings={self.warning_count}"
        )
        return tuple(lines)


class _CheckBuilder:
    def __init__(self, check_id: str, description: str) -> None:
        self.check_id = check_id
        self.description = description
        self.issues: list[ValidationIssue] = []
        self.measurements: dict[str, object] = {}

    def error(
        self,
        code: str,
        message: str,
        *,
        object_name: str | None = None,
        measured: float | int | str | bool | None = None,
        expected: float | int | str | bool | None = None,
        unit: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity="error",
                message=message,
                object_name=object_name,
                measured=measured,
                expected=expected,
                unit=unit,
            )
        )

    def warning(
        self,
        code: str,
        message: str,
        *,
        explanation: str,
        object_name: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity="warning",
                message=message,
                object_name=object_name,
                explanation=explanation,
            )
        )

    def finish(self) -> ValidationCheck:
        return ValidationCheck(
            check_id=self.check_id,
            description=self.description,
            passed=not any(issue.severity == "error" for issue in self.issues),
            issues=tuple(self.issues),
            measurements=dict(sorted(self.measurements.items())),
        )


@dataclass(frozen=True)
class RuntimeJoint:
    name: str
    joint_type: str
    parent_body: str
    child_body: str
    axis_xyz: tuple[float, float, float]
    limited: bool
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class _ValidationContext:
    config: ResolvedRunConfig
    repository_root: Path
    asset_root: Path
    source_urdf: Path
    visual_mesh_dir: Path
    runtime_mjcf: Path
    viewer_mjcf: Path
    collision_manifest: Path
    generation_manifest_path: Path
    inventory: pipeline.SourceInventory
    runtime_root: ET.Element
    viewer_root: ET.Element
    model: mujoco.MjModel
    viewer_model: mujoco.MjModel
    manifest: Mapping[str, object]
    name_index: Mapping[str, object]
    hierarchy: Mapping[str, object]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} must contain a JSON object with string keys")
    return value


def _asset_path(repository_root: Path, configured: str) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else repository_root / path


def _make_context(
    config: ResolvedRunConfig,
    *,
    repository_root: Path = pipeline.REPOSITORY_ROOT,
    runtime_mjcf: Path | None = None,
) -> _ValidationContext:
    source_urdf = _asset_path(repository_root, config.asset.source_urdf_path)
    visual_mesh_dir = _asset_path(repository_root, config.asset.visual_mesh_directory)
    runtime = runtime_mjcf or _asset_path(repository_root, config.asset.runtime_mjcf_path)
    viewer = _asset_path(repository_root, config.asset.viewer_mjcf_path)
    collision_manifest = _asset_path(repository_root, config.asset.collision_manifest_path)
    generation_manifest_path = _asset_path(repository_root, config.asset.generation_manifest_path)
    asset_root = generation_manifest_path.parent
    inventory = pipeline.inspect_source_assets(source_urdf, visual_mesh_dir)
    return _ValidationContext(
        config=config,
        repository_root=repository_root,
        asset_root=asset_root,
        source_urdf=source_urdf,
        visual_mesh_dir=visual_mesh_dir,
        runtime_mjcf=runtime,
        viewer_mjcf=viewer,
        collision_manifest=collision_manifest,
        generation_manifest_path=generation_manifest_path,
        inventory=inventory,
        runtime_root=ET.parse(runtime).getroot(),
        viewer_root=ET.parse(viewer).getroot(),
        model=mujoco.MjModel.from_xml_path(str(runtime.resolve())),
        viewer_model=mujoco.MjModel.from_xml_path(str(viewer.resolve())),
        manifest=_read_json_mapping(generation_manifest_path),
        name_index=_read_json_mapping(asset_root / "mjcf" / pipeline.NAME_INDEX_NAME),
        hierarchy=_read_json_mapping(asset_root / "mjcf" / pipeline.JOINT_HIERARCHY_NAME),
    )


def _parse_vector(raw: str | None, length: int, default: Sequence[float]) -> tuple[float, ...]:
    values = tuple(default) if raw is None else tuple(float(item) for item in raw.split())
    if len(values) != length or not all(math.isfinite(item) for item in values):
        raise ValueError(f"expected {length} finite numbers, got {raw!r}")
    return values


def extract_runtime_joints(root: ET.Element) -> dict[str, RuntimeJoint]:
    """Extract the runtime joint tree without trusting a generated index."""

    result: dict[str, RuntimeJoint] = {}

    def visit(body: ET.Element, parent_name: str) -> None:
        child_name = body.get("name")
        if child_name is None:
            raise ValueError("every runtime body must have a name")
        for element in (*body.findall("joint"), *body.findall("freejoint")):
            name = element.get("name")
            if name is None or name in result:
                raise ValueError(f"runtime joint name is absent or duplicated: {name!r}")
            joint_type = "free" if element.tag == "freejoint" else element.get("type", "hinge")
            axis = _parse_vector(element.get("axis"), 3, (0.0, 0.0, 1.0))
            range_values = element.get("range")
            limited_raw = element.get("limited", "auto")
            limited = range_values is not None and limited_raw != "false"
            lower: float | None = None
            upper: float | None = None
            if range_values is not None:
                lower, upper = _parse_vector(range_values, 2, (0.0, 0.0))
            result[name] = RuntimeJoint(
                name=name,
                joint_type=joint_type,
                parent_body=parent_name,
                child_body=child_name,
                axis_xyz=(axis[0], axis[1], axis[2]),
                limited=limited,
                lower=lower,
                upper=upper,
            )
        for child in body.findall("body"):
            visit(child, child_name)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("runtime MJCF has no worldbody")
    for body in worldbody.findall("body"):
        visit(body, "world")
    return result


def compare_joint_structure(inventory: pipeline.SourceInventory, runtime_root: ET.Element) -> ValidationCheck:
    builder = _CheckBuilder(
        "joint_structure",
        "URDF and MJCF movable-joint names, hierarchy and types agree",
    )
    try:
        runtime = extract_runtime_joints(runtime_root)
    except (ValueError, TypeError) as error:
        builder.error("joint.runtime_parse", str(error))
        return builder.finish()
    source_names = {joint.name for joint in inventory.joints}
    runtime_robot_names = set(runtime) - {"cylinder_free_joint"}
    for missing in sorted(source_names - runtime_robot_names):
        builder.error("joint.missing", f"source joint {missing!r} is absent from runtime MJCF", object_name=missing)
    for extra in sorted(runtime_robot_names - source_names):
        builder.error("joint.extra", f"runtime robot joint {extra!r} is absent from source URDF", object_name=extra)
    type_map = {"revolute": "hinge", "continuous": "hinge", "prismatic": "slide"}
    for source in inventory.joints:
        target = runtime.get(source.name)
        if target is None:
            continue
        expected_type = type_map[source.joint_type]
        if target.joint_type != expected_type:
            builder.error(
                "joint.type",
                f"joint {source.name!r} type differs",
                object_name=source.name,
                measured=target.joint_type,
                expected=expected_type,
            )
        if target.parent_body != source.parent_link or target.child_body != source.child_link:
            builder.error(
                "joint.hierarchy",
                f"joint {source.name!r} parent/child hierarchy differs",
                object_name=source.name,
                measured=f"{target.parent_body}->{target.child_body}",
                expected=f"{source.parent_link}->{source.child_link}",
            )
    builder.measurements.update(
        source_movable_joint_count=len(source_names),
        runtime_robot_joint_count=len(runtime_robot_names),
        hierarchy_unit="named body edge",
    )
    return builder.finish()


def compare_joint_axes_and_limits(inventory: pipeline.SourceInventory, runtime_root: ET.Element) -> ValidationCheck:
    builder = _CheckBuilder(
        "joint_axes_limits",
        "joint axes and radian limits agree within explicit tolerances",
    )
    try:
        runtime = extract_runtime_joints(runtime_root)
    except (ValueError, TypeError) as error:
        builder.error("joint.runtime_parse", str(error))
        return builder.finish()
    for source in inventory.joints:
        target = runtime.get(source.name)
        if target is None:
            continue
        axis_error = float(np.max(np.abs(np.asarray(source.axis_xyz) - np.asarray(target.axis_xyz))))
        if axis_error > AXIS_TOLERANCE:
            builder.error(
                "joint.axis",
                f"joint {source.name!r} axis differs or is flipped",
                object_name=source.name,
                measured=axis_error,
                expected=AXIS_TOLERANCE,
                unit="absolute axis component",
            )
        if source.joint_type == "continuous":
            if target.limited:
                builder.error(
                    "joint.unexpected_limit",
                    f"continuous joint {source.name!r} became limited",
                    object_name=source.name,
                )
            continue
        if not target.limited or target.lower is None or target.upper is None:
            builder.error(
                "joint.limit_missing", f"joint {source.name!r} lost its finite limit", object_name=source.name
            )
            continue
        limit_error = max(abs(target.lower - source.lower), abs(target.upper - source.upper))
        if limit_error > LIMIT_TOLERANCE_RAD:
            builder.error(
                "joint.limit",
                f"joint {source.name!r} radian range differs",
                object_name=source.name,
                measured=limit_error,
                expected=LIMIT_TOLERANCE_RAD,
                unit="rad",
            )
    builder.measurements.update(
        axis_tolerance=AXIS_TOLERANCE,
        axis_unit="dimensionless local-frame unit vector",
        limit_tolerance_rad=LIMIT_TOLERANCE_RAD,
        limit_unit="rad",
    )
    return builder.finish()


def _float_tuple(element: ET.Element, attribute: str, count: int) -> tuple[float, ...]:
    return _parse_vector(element.get(attribute), count, ())


def _safe_control_range(joint: pipeline.SourceJoint, inventory: pipeline.SourceInventory) -> tuple[float, float]:
    lower, upper = joint.lower, joint.upper
    for follower, (leader, multiplier, offset) in inventory.mimic.items():
        if leader != joint.name:
            continue
        follower_joint = inventory.joint_by_name[follower]
        values = (
            (follower_joint.lower - offset) / multiplier,
            (follower_joint.upper - offset) / multiplier,
        )
        lower = max(lower, min(values))
        upper = min(upper, max(values))
    return lower, upper


def _check_actuators(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "actuators",
        "seven right-arm and six active-hand position actuators preserve order and limits",
    )
    elements = context.runtime_root.findall("./actuator/position")
    expected_groups = (
        context.config.physics.actuator_calibration.arm,
        context.config.physics.actuator_calibration.hand,
    )
    expected_names = tuple(name for group in expected_groups for name in group.joint_names)
    actual_names = tuple(element.get("joint") for element in elements)
    if actual_names != expected_names:
        builder.error(
            "actuator.order",
            "runtime actuator joint order is not 7 right-arm followed by 6 active-hand joints",
            measured=str(actual_names),
            expected=str(expected_names),
        )
    for index, (element, joint_name) in enumerate(zip(elements, expected_names, strict=False)):
        group = expected_groups[0] if index < 7 else expected_groups[1]
        group_index = index if index < 7 else index - 7
        expected_effort = group.effort_limit_n_m[group_index]
        expected_velocity = group.velocity_limit_rad_s[group_index]
        expected_kp = group.stiffness_n_m_per_rad[group_index]
        expected_kd = group.damping_n_m_s_per_rad[group_index]
        try:
            ctrlrange = _float_tuple(element, "ctrlrange", 2)
            forcerange = _float_tuple(element, "forcerange", 2)
            user = _float_tuple(element, "user", 4)
            kp = float(element.get("kp", "nan"))
            kd = float(element.get("kv", "nan"))
        except ValueError as error:
            builder.error(
                "actuator.metadata", f"actuator {joint_name!r} metadata is invalid: {error}", object_name=joint_name
            )
            continue
        expected_ctrl = _safe_control_range(context.inventory.joint_by_name[joint_name], context.inventory)
        comparisons = (
            (
                "control_range",
                max(abs(ctrlrange[0] - expected_ctrl[0]), abs(ctrlrange[1] - expected_ctrl[1])),
                LIMIT_TOLERANCE_RAD,
                "rad",
            ),
            (
                "effort_range",
                max(abs(forcerange[0] + expected_effort), abs(forcerange[1] - expected_effort)),
                1.0e-10,
                "N*m",
            ),
            ("velocity_limit", abs(user[0] - expected_velocity), 1.0e-10, "rad/s"),
            ("user_effort", abs(user[1] - expected_effort), 1.0e-10, "N*m"),
            ("stiffness", max(abs(kp - expected_kp), abs(user[2] - expected_kp)), 1.0e-10, "N*m/rad"),
            ("damping", max(abs(kd - expected_kd), abs(user[3] - expected_kd)), 1.0e-10, "N*m*s/rad"),
        )
        for field_name, deviation, tolerance, unit in comparisons:
            if deviation > tolerance:
                builder.error(
                    f"actuator.{field_name}",
                    f"actuator {joint_name!r} {field_name} differs from its source contract",
                    object_name=joint_name,
                    measured=deviation,
                    expected=tolerance,
                    unit=unit,
                )
    if len(elements) != 13 or context.model.nu != 13:
        builder.error(
            "actuator.count",
            "compiled runtime must expose exactly 13 physical actuators",
            measured=context.model.nu,
            expected=13,
        )
    builder.measurements.update(
        physics_version=context.config.physics.version,
        physics_calibration_status=context.config.physics.calibration_status,
        actuator_calibration_version=context.config.physics.actuator_calibration.version,
        actuator_calibration_status=context.config.physics.actuator_calibration.status,
        actuator_count=context.model.nu,
        arm_actuator_count=7,
        active_hand_actuator_count=6,
        actuator_order=list(expected_names),
        actuator_parameters=[
            {
                "joint": joint_name,
                "stiffness_n_m_per_rad": group.stiffness_n_m_per_rad[group_index],
                "damping_n_m_s_per_rad": group.damping_n_m_s_per_rad[group_index],
                "effort_limit_n_m": group.effort_limit_n_m[group_index],
                "velocity_limit_rad_s": group.velocity_limit_rad_s[group_index],
            }
            for group in expected_groups
            for group_index, joint_name in enumerate(group.joint_names)
        ],
    )
    return builder.finish()


def _check_hand_mimic(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "hand_mimic",
        "right-hand active/observed joints and all source mimic relations remain explicit",
    )
    active = context.config.physics.actuator_calibration.hand.joint_names
    observed = context.config.task.observation.observed_joint_names[7:]
    if len(active) != 6 or len(set(active)) != 6:
        builder.error("hand.active", "right hand must expose six unique active joints")
    if len(observed) != 12 or len(set(observed)) != 12:
        builder.error("hand.observed", "right hand must expose twelve unique observed joints")
    right_source = {name: relation for name, relation in context.inventory.mimic.items() if name.startswith("right_")}
    equality_elements = {
        element.get("joint1"): element
        for element in context.runtime_root.findall("./equality/joint")
        if element.get("name", "").startswith("mimic__")
    }
    if set(equality_elements) != set(right_source):
        builder.error(
            "mimic.right_inventory",
            "runtime right-hand mimic followers do not exactly match the source URDF",
            measured=str(sorted(name for name in equality_elements if name is not None)),
            expected=str(sorted(right_source)),
        )
    for follower, (leader, multiplier, offset) in sorted(right_source.items()):
        element = equality_elements.get(follower)
        if element is None:
            continue
        try:
            polycoef = _float_tuple(element, "polycoef", 5)
        except ValueError as error:
            builder.error("mimic.polycoef", str(error), object_name=follower)
            continue
        if element.get("joint2") != leader or not np.allclose(
            polycoef,
            (offset, multiplier, 0.0, 0.0, 0.0),
            rtol=0.0,
            atol=1.0e-12,
        ):
            builder.error(
                "mimic.mapping", f"mimic relationship for {follower!r} differs from URDF", object_name=follower
            )
    left_source = {name for name in context.inventory.mimic if name.startswith("left_")}
    fixed_equalities = {
        element.get("joint1")
        for element in context.runtime_root.findall("./equality/joint")
        if element.get("name", "").startswith("fixed__")
    }
    if not left_source <= fixed_equalities:
        builder.error(
            "mimic.fixed_non_task",
            "left-hand source mimic joints must remain present and be fixed as non-task joints",
            measured=str(sorted(name for name in left_source if name not in fixed_equalities)),
            expected="none missing",
        )
    builder.measurements.update(
        active_right_hand_joints=list(active),
        observed_right_hand_joints=list(observed),
        source_mimic_relation_count=len(context.inventory.mimic),
        runtime_right_mimic_relation_count=len(equality_elements),
        fixed_left_mimic_relation_count=len(left_source & fixed_equalities),
    )
    return builder.finish()


def _check_named_objects(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "named_objects",
        "TCP, grasp, target, approach and contact-group names resolve uniquely",
    )
    requirements = {
        "body": ("right_tcp_link",),
        "site": ("right_tcp_site", "grasp_center_site", "target_tcp_site", "approach_direction_site"),
    }
    object_types = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE}
    for kind, names in requirements.items():
        xml_names = [element.get("name") for element in context.runtime_root.iter(kind)]
        for name in names:
            if xml_names.count(name) != 1:
                builder.error(
                    "name.unique",
                    f"{kind} {name!r} must occur exactly once in runtime MJCF",
                    object_name=name,
                    measured=xml_names.count(name),
                    expected=1,
                )
            if mujoco.mj_name2id(context.model, object_types[kind], name) < 0:
                builder.error("name.compiled", f"compiled model cannot resolve {kind} {name!r}", object_name=name)
    contact_groups_raw = context.name_index.get("contact_groups")
    if not isinstance(contact_groups_raw, dict):
        builder.error("contact_groups.schema", "name index has no contact_groups mapping")
        contact_groups_raw = {}
    required_groups = (
        "right_hand_observation_links",
        "thumb_side",
        "four_finger_opposing",
        "right_arm",
        "tcp",
        "reach_diagnostic",
        "robot",
        "table",
        "cylinder",
        "non_task_body",
    )
    for required_group_name in required_groups:
        if required_group_name not in contact_groups_raw:
            builder.error(
                "contact_groups.missing",
                f"required contact group {required_group_name!r} is absent",
                object_name=required_group_name,
            )
    compiled_member_count = 0
    for group_name, raw_specification in sorted(contact_groups_raw.items()):
        if not isinstance(group_name, str) or not isinstance(raw_specification, dict):
            builder.error("contact_groups.schema", "each contact group must be a named mapping")
            continue
        group_kind = raw_specification.get("kind")
        raw_members = raw_specification.get("members")
        if not isinstance(raw_members, list) or not all(isinstance(member, str) for member in raw_members):
            builder.error(
                "contact_groups.schema",
                f"contact group {group_name!r} must contain a string member list",
                object_name=group_name,
            )
            continue
        members = tuple(raw_members)
        if len(set(members)) != len(members):
            builder.error(
                "contact_groups.duplicate_member",
                f"contact group {group_name!r} contains duplicate members",
                object_name=group_name,
            )
        if group_kind == "collision_mask":
            if members != ("robot",):
                builder.error(
                    "contact_groups.collision_mask",
                    f"virtual collision-mask group {group_name!r} has an unexpected member contract",
                    object_name=group_name,
                )
            continue
        if not isinstance(group_kind, str):
            builder.error(
                "contact_groups.kind",
                f"contact group {group_name!r} has unsupported kind {group_kind!r}",
                object_name=group_name,
            )
            continue
        object_type = {
            "body": mujoco.mjtObj.mjOBJ_BODY,
            "geom": mujoco.mjtObj.mjOBJ_GEOM,
        }.get(group_kind)
        if object_type is None:
            builder.error(
                "contact_groups.kind",
                f"contact group {group_name!r} has unsupported kind {group_kind!r}",
                object_name=group_name,
            )
            continue
        xml_tag = str(group_kind)
        xml_names = [element.get("name") for element in context.runtime_root.iter(xml_tag)]
        for member in members:
            if xml_names.count(member) != 1 or mujoco.mj_name2id(context.model, object_type, member) < 0:
                builder.error(
                    "contact_groups.member",
                    f"{group_kind} {member!r} in contact group {group_name!r} must compile exactly once",
                    object_name=member,
                    measured=xml_names.count(member),
                    expected=1,
                )
            else:
                compiled_member_count += 1
    builder.measurements.update(
        required_contact_groups=list(required_groups),
        named_frame_count=4,
        compiled_contact_group_member_count=compiled_member_count,
    )
    return builder.finish()


def _binary_stl_triangle_count(path: Path) -> int | None:
    size = path.stat().st_size
    if size < 84:
        return None
    with path.open("rb") as stream:
        stream.seek(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
    return triangle_count if 84 + 50 * triangle_count == size else None


def _stl_loadable(path: Path) -> tuple[bool, int]:
    binary_count = _binary_stl_triangle_count(path)
    if binary_count is not None and binary_count > 0:
        return True, binary_count
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return False, 0
    facet_count = text.lower().count("facet normal")
    return text.lstrip().lower().startswith("solid") and facet_count > 0, facet_count


def _check_stl_visual_only(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "stl_visual_only",
        "every STL is loadable and only declared right-hand mesh geometry participates in collision",
    )
    total_triangles = 0
    for path in context.inventory.mesh_paths:
        loadable, triangles = _stl_loadable(path)
        total_triangles += triangles
        if not loadable:
            builder.error(
                "stl.load", f"STL {path.name!r} is not a valid non-empty binary or ASCII STL", object_name=path.name
            )
    try:
        declarations = _load_collision_declarations(context.collision_manifest)
    except (OSError, ValueError, yaml.YAMLError) as error:
        builder.error("stl.manifest", str(error))
        declarations = ()
    declared_mesh_names = {
        str(item.get("name"))
        for item in declarations
        if item.get("type") == "mesh" and str(item.get("mode", "physical")) == "physical"
    }
    headless_mesh_geoms = [element for element in context.runtime_root.iter("geom") if element.get("type") == "mesh"]
    for element in headless_mesh_geoms:
        if element.get("name") not in declared_mesh_names or element.get("contype") == "0":
            builder.error(
                "stl.headless_visual",
                f"headless runtime contains undeclared or visual-only mesh {element.get('name')!r}",
                object_name=element.get("name"),
            )
    mesh_geoms = [element for element in context.viewer_root.iter("geom") if element.get("type") == "mesh"]
    for element in mesh_geoms:
        if (element.get("contype") != "0" or element.get("conaffinity") != "0") and element.get(
            "name"
        ) not in declared_mesh_names:
            builder.error(
                "stl.collision",
                f"undeclared mesh geom {element.get('name')!r} is collision-enabled",
                object_name=element.get("name"),
            )
    compiled_collision_meshes = [
        mujoco.mj_id2name(context.viewer_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        for geom_id in range(context.viewer_model.ngeom)
        if context.viewer_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        and (context.viewer_model.geom_contype[geom_id] != 0 or context.viewer_model.geom_conaffinity[geom_id] != 0)
    ]
    for name in compiled_collision_meshes:
        if name not in declared_mesh_names:
            builder.error("stl.compiled_collision", f"undeclared mesh {name!r} is collision-enabled", object_name=name)
    builder.measurements.update(
        stl_count=len(context.inventory.mesh_paths),
        headless_visual_mesh_geom_count=len(headless_mesh_geoms),
        visual_mesh_geom_count=len(mesh_geoms),
        total_stl_triangles=total_triangles,
        compiled_collision_mesh_count=len(compiled_collision_meshes),
        declared_collision_mesh_count=len(declared_mesh_names),
    )
    return builder.finish()


def _physical_xml_signature(element: ET.Element, physical_meshes: set[str] | None = None) -> object:
    """Canonical XML tree after removing only declared visual-mesh content."""

    if physical_meshes is None:
        physical_meshes = {
            child.get("mesh")
            for child in element.iter("geom")
            if child.get("type") == "mesh" and (child.get("contype") != "0" or child.get("conaffinity") != "0")
        }
    children = []
    for child in element:
        if element.tag == "asset" and child.tag == "mesh" and child.get("name") not in physical_meshes:
            continue
        if (
            child.tag == "geom"
            and child.get("type") == "mesh"
            and child.get("contype") == "0"
            and child.get("conaffinity") == "0"
        ):
            continue
        children.append(_physical_xml_signature(child, physical_meshes))
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(children),
    )


def _check_runtime_variants(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "runtime_variants",
        "headless and viewer MJCFs differ only by visual mesh assets and geoms",
    )
    if _physical_xml_signature(context.runtime_root) != _physical_xml_signature(context.viewer_root):
        builder.error(
            "variants.physics_xml",
            "headless and viewer physical XML contracts differ after removing visual meshes",
            object_name=context.viewer_mjcf.name,
            expected="viewer minus asset/mesh and geom[type=mesh] equals headless",
        )
    count_fields = ("nq", "nv", "nu", "nbody", "njnt", "neq", "nsite")
    count_mismatches = {
        name: (int(getattr(context.model, name)), int(getattr(context.viewer_model, name)))
        for name in count_fields
        if int(getattr(context.model, name)) != int(getattr(context.viewer_model, name))
    }
    if count_mismatches:
        builder.error(
            "variants.compiled_counts",
            "headless and viewer compiled physical dimensions differ",
            measured=json.dumps(count_mismatches, sort_keys=True),
            expected="identical physical dimensions",
        )
    if context.model.nmesh == 0:
        builder.error(
            "variants.headless_meshes",
            "headless runtime compiled no declared collision meshes",
            measured=0,
            expected=1,
        )
    declared_mesh_count = sum(
        1
        for item in _load_collision_declarations(context.collision_manifest)
        if item.get("type") == "mesh" and str(item.get("mode", "physical")) == "physical"
    )
    if context.model.nmesh != declared_mesh_count:
        builder.error(
            "variants.headless_mesh_count",
            "headless runtime mesh count differs from declared collision meshes",
            measured=context.model.nmesh,
            expected=declared_mesh_count,
        )
    if context.viewer_model.nmesh != len(context.inventory.mesh_paths):
        builder.error(
            "variants.viewer_meshes",
            "viewer runtime did not compile every immutable source STL",
            measured=context.viewer_model.nmesh,
            expected=len(context.inventory.mesh_paths),
        )
    builder.measurements.update(
        equivalence_rule="viewer minus visual-only mesh assets/geoms equals headless",
        compiled_count_fields={name: int(getattr(context.model, name)) for name in count_fields},
        headless_mesh_count=context.model.nmesh,
        viewer_mesh_count=context.viewer_model.nmesh,
        viewer_mjcf_sha256=pipeline.sha256_file(context.viewer_mjcf),
    )
    return builder.finish()


def _urdf_inertials(path: Path) -> Iterable[tuple[str, float, np.ndarray, np.ndarray]]:
    root = ET.parse(path).getroot()
    for link in root.findall("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_element = inertial.find("mass")
        inertia_element = inertial.find("inertia")
        if mass_element is None or inertia_element is None:
            raise ValueError(f"link {name!r} has incomplete inertial data")
        mass = float(mass_element.get("value", "nan"))
        origin = inertial.find("origin")
        com = np.asarray(_parse_vector(None if origin is None else origin.get("xyz"), 3, (0.0, 0.0, 0.0)))
        ixx = float(inertia_element.get("ixx", "nan"))
        ixy = float(inertia_element.get("ixy", "nan"))
        ixz = float(inertia_element.get("ixz", "nan"))
        iyy = float(inertia_element.get("iyy", "nan"))
        iyz = float(inertia_element.get("iyz", "nan"))
        izz = float(inertia_element.get("izz", "nan"))
        tensor = np.asarray(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)), dtype=np.float64)
        yield name, mass, com, tensor


def _validate_inertia_values(
    builder: _CheckBuilder,
    name: str,
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
) -> None:
    if not math.isfinite(mass) or mass <= 0.0:
        builder.error(
            "inertia.mass",
            f"{name!r} mass must be finite and positive",
            object_name=name,
            measured=mass,
            expected="> 0",
            unit="kg",
        )
        return
    if not np.isfinite(com).all() or not np.isfinite(inertia).all():
        builder.error("inertia.nonfinite", f"{name!r} COM or inertia contains a non-finite value", object_name=name)
        return
    eigenvalues = np.linalg.eigvalsh(inertia)
    if float(np.min(eigenvalues)) <= INERTIA_EIGENVALUE_TOLERANCE_KG_M2:
        builder.error(
            "inertia.positive_definite",
            f"{name!r} inertia is not positive definite",
            object_name=name,
            measured=float(np.min(eigenvalues)),
            expected=INERTIA_EIGENVALUE_TOLERANCE_KG_M2,
            unit="kg*m^2",
        )
    principal = np.sort(eigenvalues)
    violation = float(principal[2] - principal[0] - principal[1])
    if violation > TRIANGLE_INEQUALITY_TOLERANCE_KG_M2:
        builder.error(
            "inertia.triangle",
            f"{name!r} principal inertia violates the triangle inequality",
            object_name=name,
            measured=violation,
            expected=TRIANGLE_INEQUALITY_TOLERANCE_KG_M2,
            unit="kg*m^2",
        )


def _check_inertials(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "inertials",
        "source and runtime mass, COM and inertia values are finite and physically admissible",
    )
    source_count = 0
    try:
        for name, mass, com, tensor in _urdf_inertials(context.source_urdf):
            source_count += 1
            _validate_inertia_values(builder, f"urdf:{name}", mass, com, tensor)
    except ValueError as error:
        builder.error("inertia.urdf_parse", str(error))
    runtime_count = 0
    for body_id in range(1, context.model.nbody):
        name = mujoco.mj_id2name(context.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        runtime_count += 1
        _validate_inertia_values(
            builder,
            f"mjcf:{name}",
            float(context.model.body_mass[body_id]),
            np.asarray(context.model.body_ipos[body_id]),
            np.diag(np.asarray(context.model.body_inertia[body_id])),
        )
    builder.measurements.update(source_inertial_count=source_count, runtime_dynamic_body_count=runtime_count)
    return builder.finish()


def _reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey != 1:
        raise ValueError(f"runtime model must have exactly one home keyframe, got {model.nkey}")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def _quat_tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    # Public backend snapshots are intentionally read-only.  Own the small
    # vector before normalization so validation never mutates simulator state.
    quaternion = np.array(quaternion_wxyz, dtype=np.float64, copy=True)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        return math.inf
    quaternion /= norm
    _, x, y, _ = quaternion
    rotation_zz = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(float(np.clip(rotation_zz, -1.0, 1.0)))


def _check_initial_pose(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "initial_pose",
        "fixed base, non-task posture, upright cylinder, table height and task frames agree",
    )
    model = context.model
    data = mujoco.MjData(model)
    try:
        _reset_home(model, data)
    except ValueError as error:
        builder.error("pose.keyframe", str(error))
        return builder.finish()
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    pelvis_joints = np.flatnonzero(model.jnt_bodyid == pelvis_id)
    if pelvis_joints.size != 0:
        builder.error(
            "pose.fixed_base",
            "pelvis/root body unexpectedly owns a movable joint",
            measured=int(pelvis_joints.size),
            expected=0,
        )
    expected_pelvis_position = np.asarray((0.0, 0.0, 0.8908))
    pelvis_error = float(np.max(np.abs(data.xpos[pelvis_id] - expected_pelvis_position)))
    if pelvis_error > 1.0e-12:
        builder.error(
            "pose.base_position",
            "fixed base position differs from its versioned patch",
            measured=pelvis_error,
            expected=1.0e-12,
            unit="m",
        )
    hierarchy_joints = context.hierarchy.get("joints")
    if not isinstance(hierarchy_joints, list):
        builder.error("pose.hierarchy_schema", "generated hierarchy has no joints list")
        hierarchy_joints = []
    fixed_differences: list[float] = []
    for entry in hierarchy_joints:
        if not isinstance(entry, dict) or entry.get("runtime_role") != "fixed_non_task":
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_address = int(model.jnt_qposadr[joint_id])
        equality_name = f"fixed__{name}"
        equality = next(
            (item for item in context.runtime_root.findall("./equality/joint") if item.get("name") == equality_name),
            None,
        )
        if equality is None:
            builder.error("pose.fixed_equality", f"non-task joint {name!r} has no fixed equality", object_name=name)
            continue
        expected = _float_tuple(equality, "polycoef", 5)[0]
        fixed_differences.append(abs(float(data.qpos[qpos_address]) - expected))
    cylinder_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free_joint")
    cylinder_qpos = int(model.jnt_qposadr[cylinder_joint])
    cylinder_position = np.asarray(data.qpos[cylinder_qpos : cylinder_qpos + 3])
    cylinder_quaternion = np.asarray(data.qpos[cylinder_qpos + 3 : cylinder_qpos + 7])
    scene = context.config.task.scene
    cylinder_xy_error = float(np.max(np.abs(cylinder_position[:2] - np.asarray(scene.cylinder_origin_xy_m))))
    if cylinder_xy_error > 1.0e-12:
        builder.error(
            "pose.cylinder_xy",
            "cylinder XY position differs from the versioned task-scene origin",
            measured=cylinder_xy_error,
            expected=1.0e-12,
            unit="m",
        )
    tabletop_z = scene.table_center_m[2] + scene.table_size_m[2] / 2.0
    nominal_height = (context.config.randomization.height.minimum + context.config.randomization.height.maximum) / 2.0
    expected_z = tabletop_z + scene.cylinder_table_clearance_m + nominal_height / 2.0
    if abs(float(cylinder_position[2]) - expected_z) > 1.0e-12:
        builder.error(
            "pose.cylinder_height",
            "cylinder center is inconsistent with table top, clearance and height",
            measured=float(cylinder_position[2]),
            expected=expected_z,
            unit="m",
        )
    tilt = _quat_tilt_rad(cylinder_quaternion)
    if tilt > 1.0e-12:
        builder.error(
            "pose.cylinder_vertical", "initial cylinder is not vertical", measured=tilt, expected=1.0e-12, unit="rad"
        )
    table_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    table_top_measured = float(data.geom_xpos[table_geom, 2] + model.geom_size[table_geom, 2])
    if abs(table_top_measured - tabletop_z) > 1.0e-12:
        builder.error(
            "pose.table_height",
            "compiled tabletop height differs from SceneConfig",
            measured=table_top_measured,
            expected=tabletop_z,
            unit="m",
        )
    target_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_tcp_site")
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center_site")
    target_offset = np.asarray(data.site_xpos[target_site] - cylinder_position)
    # The target TCP is defined relative to the grasp center, not the
    # cylinder's top surface.  The grasp-center site declaration is the
    # versioned TCP-to-grasp offset used by the task frame.  Configured lateral
    # approach corrections in X/Y are intentionally preserved.
    target_rotation = np.asarray(data.site_xmat[target_site]).reshape(3, 3)
    expected_target_offset = target_rotation @ np.asarray(model.site_pos[grasp_site])
    expected_target_offset[0] = target_offset[0]
    expected_target_offset[1] = target_offset[1]
    target_error = float(np.linalg.norm(target_offset - expected_target_offset))
    if target_error > 1.0e-12:
        builder.error(
            "pose.task_frame",
            "initial target TCP frame does not preserve the calibrated TCP/grasp-center offset",
            measured=target_error,
            expected=1.0e-12,
            unit="m",
        )
    builder.measurements.update(
        fixed_base=True,
        fixed_non_task_joint_count=len(fixed_differences),
        maximum_fixed_posture_error_rad=max(fixed_differences, default=0.0),
        tabletop_height_m=table_top_measured,
        cylinder_center_m=cylinder_position.tolist(),
        cylinder_xy_origin_error_m=cylinder_xy_error,
        cylinder_tilt_rad=tilt,
        target_tcp_offset_from_cylinder_center_m=target_offset.tolist(),
        target_tcp_offset_error_m=target_error,
    )
    return builder.finish()


def _primitive_outside_distance(
    points_world: np.ndarray,
    geom_id: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    local = (points_world - data.geom_xpos[geom_id]) @ rotation
    sizes = np.asarray(model.geom_size[geom_id])
    geom_type = int(model.geom_type[geom_id])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.maximum(np.linalg.norm(local, axis=1) - sizes[0], 0.0)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.linalg.norm(np.maximum(np.abs(local) - sizes[:3], 0.0), axis=1)
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radial = np.linalg.norm(local[:, :2], axis=1)
        axial = np.maximum(np.abs(local[:, 2]) - sizes[1], 0.0)
        return np.maximum(np.sqrt(radial * radial + axial * axial) - sizes[0], 0.0)
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radial = np.maximum(np.linalg.norm(local[:, :2], axis=1) - sizes[0], 0.0)
        axial = np.maximum(np.abs(local[:, 2]) - sizes[1], 0.0)
        return np.sqrt(radial * radial + axial * axial)
    raise ValueError(f"unsupported simplified collision type {geom_type}")


def _visual_vertices_world(body_id: int, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    batches: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != body_id or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count])
        rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
        batches.append(vertices @ rotation.T + data.geom_xpos[geom_id])
    if not batches:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(batches, axis=0)


def _load_collision_declarations(path: Path) -> tuple[Mapping[str, object], ...]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("geometries"), list):
        raise ValueError("collision manifest must contain a geometries list")
    result: list[Mapping[str, object]] = []
    for item in value["geometries"]:
        if not isinstance(item, dict):
            raise ValueError("each collision declaration must be a mapping")
        result.append(item)
    return tuple(result)


def _coverage_reason(geom_type: str) -> str:
    return {
        "mesh": "source STL is used directly for right-hand collision fidelity",
        "box": "box approximates a broad rigid volume with a deterministic low-cost primitive",
        "sphere": "sphere approximates a compact joint or terminal volume without orientation-sensitive faces",
        "capsule": "capsule follows an elongated limb or finger while keeping contact convex and inexpensive",
        "cylinder": "cylinder preserves the dominant axial geometry with a low-complexity primitive",
    }[geom_type]


def _check_collision_coverage(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "collision_coverage",
        "simplified primitive coverage is measured against each target visual mesh",
    )
    # Coverage needs the full visual mesh product; collision dynamics and all
    # task execution continue to use the lightweight headless runtime.
    model = context.viewer_model
    data = mujoco.MjData(model)
    _reset_home(model, data)
    records: list[dict[str, object]] = []
    try:
        declarations = _load_collision_declarations(context.collision_manifest)
    except (OSError, ValueError, yaml.YAMLError) as error:
        builder.error("coverage.manifest", str(error))
        return builder.finish()
    # A visual body may be covered by multiple simplified primitives (for
    # example, the palm root and front-palm boxes).  Coverage is therefore
    # measured against their union rather than requiring every primitive to
    # cover the entire body on its own.
    body_geom_ids: dict[str, list[int]] = {}
    for declaration in declarations:
        body = declaration.get("body")
        name = declaration.get("name")
        if isinstance(body, str) and isinstance(name, str) and declaration.get("type") != "mesh":
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                body_geom_ids.setdefault(body, []).append(geom_id)
    for declaration in declarations:
        name = declaration.get("name")
        body = declaration.get("body")
        geom_type = declaration.get("type")
        if not isinstance(name, str) or not isinstance(body, str) or not isinstance(geom_type, str):
            builder.error("coverage.schema", f"invalid collision declaration {declaration!r}")
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if body_id < 0 or geom_id < 0:
            builder.error("coverage.object", f"coverage target {name!r}/{body!r} does not compile", object_name=name)
            continue
        vertices = _visual_vertices_world(body_id, model, data)
        if vertices.shape[0] == 0:
            builder.error("coverage.visual", f"body {body!r} has no visual mesh vertices", object_name=body)
            continue
        if geom_type == "mesh":
            records.append(
                {
                    "body": body,
                    "collision_geom": name,
                    "primitive_type": geom_type,
                    "visual_vertex_count": int(vertices.shape[0]),
                    "visual_diagonal_m": float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))),
                    "outside_rms_m": 0.0,
                    "outside_max_m": 0.0,
                    "normalized_rms_error": 0.0,
                    "inside_fraction_at_0_1mm": 1.0,
                    "selection_reason": _coverage_reason(geom_type),
                }
            )
            continue
        distances = np.min(
            np.vstack(
                [_primitive_outside_distance(vertices, candidate, model, data) for candidate in body_geom_ids[body]]
            ),
            axis=0,
        )
        visual_diagonal = float(np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0)))
        rms_error = float(np.sqrt(np.mean(distances * distances)))
        normalized_error = rms_error / visual_diagonal if visual_diagonal > 0.0 else math.inf
        inside_fraction = float(np.mean(distances <= 1.0e-4))
        record = {
            "body": body,
            "collision_geom": name,
            "primitive_type": geom_type,
            "visual_vertex_count": int(vertices.shape[0]),
            "visual_diagonal_m": visual_diagonal,
            "outside_rms_m": rms_error,
            "outside_max_m": float(np.max(distances)),
            "normalized_rms_error": normalized_error,
            "inside_fraction_at_0_1mm": inside_fraction,
            "selection_reason": _coverage_reason(geom_type),
        }
        records.append(record)
        if not math.isfinite(normalized_error) or normalized_error > COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT:
            builder.error(
                "coverage.error",
                f"simplified collision {name!r} exceeds its normalized RMS coverage-error limit",
                object_name=name,
                measured=normalized_error,
                expected=COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT,
                unit="outside RMS / visual AABB diagonal",
            )
    builder.measurements.update(
        metric="RMS distance of visual vertices outside the simplified primitive divided by visual AABB diagonal",
        normalized_rms_limit=COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT,
        target_count=len(records),
        targets=records,
    )
    return builder.finish()


def write_collision_coverage_svg(check: ValidationCheck, path: Path) -> None:
    """Write a deterministic, dependency-free visual summary of coverage error."""

    targets = check.measurements.get("targets")
    if not isinstance(targets, list):
        targets = []
    width = 960
    row_height = 24
    height = 70 + row_height * len(targets)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="monospace" font-size="16">TienKung 2 Pro simplified-collision coverage</text>',
        f'<text x="20" y="48" font-family="monospace" font-size="12">normalized RMS limit = {COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT:.3f}</text>',
    ]
    for index, raw in enumerate(targets):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("collision_geom", "unknown"))
        value = float(raw.get("normalized_rms_error", math.nan))
        y = 68 + index * row_height
        bar_width = max(0.0, min(500.0, value / COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT * 500.0))
        colour = "#248f4b" if math.isfinite(value) and value <= COLLISION_COVERAGE_NORMALIZED_RMS_LIMIT else "#c9362b"
        lines.append(f'<text x="20" y="{y}" font-family="monospace" font-size="11">{name}</text>')
        lines.append(f'<rect x="360" y="{y - 11}" width="500" height="12" fill="#eeeeee"/>')
        lines.append(f'<rect x="360" y="{y - 11}" width="{bar_width:.3f}" height="12" fill="{colour}"/>')
        lines.append(f'<text x="870" y="{y}" font-family="monospace" font-size="11">{value:.6f}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _contact_pair_names(model: mujoco.MjModel, contact: mujoco.MjContact) -> tuple[str, str]:
    first = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or f"geom_{contact.geom1}"
    second = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or f"geom_{contact.geom2}"
    return first, second


def _group_members(context: _ValidationContext) -> dict[str, set[str]]:
    raw = context.name_index.get("contact_groups")
    result: dict[str, set[str]] = {}
    if not isinstance(raw, dict):
        return result
    for name, spec in raw.items():
        if not isinstance(name, str) or not isinstance(spec, dict):
            continue
        members = spec.get("members")
        if isinstance(members, list) and all(isinstance(member, str) for member in members):
            result[name] = set(members)
    return result


def _groups_for_geom(name: str, groups: Mapping[str, set[str]]) -> set[str]:
    return {group for group, members in groups.items() if name in members}


def _check_initial_penetration(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "initial_penetration",
        "initial penetrating contact pairs are absent or explained by a versioned allowlist",
    )
    data = mujoco.MjData(context.model)
    _reset_home(context.model, data)
    groups = _group_members(context)
    contacts: list[dict[str, object]] = []
    illegal_count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = _contact_pair_names(context.model, contact)
        first_groups = _groups_for_geom(first, groups)
        second_groups = _groups_for_geom(second, groups)
        allowed_reason: str | None = None
        for left in first_groups:
            for right in second_groups:
                normalized_pair = (left, right) if left <= right else (right, left)
                allowed_reason = INITIAL_CONTACT_ALLOWLIST.get(normalized_pair)
                if allowed_reason is not None:
                    break
            if allowed_reason is not None:
                break
        penetrating = float(contact.dist) < -INITIAL_PENETRATION_TOLERANCE_M
        illegal = penetrating and allowed_reason is None
        illegal_count += int(illegal)
        contacts.append(
            {
                "contact_index": index,
                "geom1": first,
                "geom2": second,
                "distance_m": float(contact.dist),
                "penetrating": penetrating,
                "allowlist_reason": allowed_reason,
                "illegal": illegal,
            }
        )
        if illegal:
            builder.error(
                "penetration.illegal",
                f"illegal initial penetration between {first!r} and {second!r}",
                object_name=f"{first}|{second}",
                measured=float(contact.dist),
                expected=-INITIAL_PENETRATION_TOLERANCE_M,
                unit="m (must be >= expected)",
            )
    builder.measurements.update(
        tolerance_m=INITIAL_PENETRATION_TOLERANCE_M,
        allowlist_version=INITIAL_CONTACT_ALLOWLIST_VERSION,
        allowlist=[
            {"groups": list(pair), "reason": reason} for pair, reason in sorted(INITIAL_CONTACT_ALLOWLIST.items())
        ],
        initial_contact_count=data.ncon,
        illegal_penetration_count=illegal_count,
        contacts=contacts,
    )
    return builder.finish()


def _mujoco_warnings(data: mujoco.MjData) -> list[dict[str, int]]:
    return [
        {"warning_index": index, "count": int(warning.number), "last_info": int(warning.lastinfo)}
        for index, warning in enumerate(data.warning)
        if warning.number
    ]


def _source_roles(context: _ValidationContext) -> dict[str, str]:
    raw = context.hierarchy.get("joints")
    result: dict[str, str] = {}
    if not isinstance(raw, list):
        return result
    for entry in raw:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and isinstance(entry.get("runtime_role"), str)
        ):
            result[str(entry["name"])] = str(entry["runtime_role"])
    return result


def tracking_response_metrics(
    positions_rad: Sequence[float],
    *,
    settled_position_rad: float,
    target_position_rad: float,
    direction: float,
) -> tuple[float, float]:
    """Return directional peak response and overshoot relative to settle.

    Expressing both the trajectory and the commanded step relative to the
    measured settled position avoids treating a non-zero absolute joint angle
    as overshoot.  The same equation applies to positive and negative trials.
    """

    if direction not in (-1.0, 1.0):
        raise ValueError(f"tracking direction must be -1.0 or 1.0, got {direction}")
    positions = np.asarray(tuple(positions_rad), dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0 or not np.isfinite(positions).all():
        raise ValueError("tracking positions must be a non-empty finite one-dimensional sequence")
    directional_peak = float(np.max(direction * (positions - settled_position_rad)))
    commanded_step = direction * (target_position_rad - settled_position_rad)
    return directional_peak, max(0.0, directional_peak - commanded_step)


def _check_zero_action_settle(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "zero_action_settle",
        "a real zero-action MuJoCo rollout keeps fixed state, object pose and numerics stable",
    )
    model = context.model
    data = mujoco.MjData(model)
    _reset_home(model, data)
    initial_qpos = np.asarray(data.qpos).copy()
    initial_pelvis = np.asarray(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]).copy()
    step_count = round(SETTLE_DURATION_S / model.opt.timestep)
    policy_steps = round(SETTLE_DURATION_S / context.config.physics.policy_dt_s)
    with MujocoEnvironmentBackend(context.config, mjcf_path=context.runtime_mjcf) as backend:
        initial_backend_state = backend.reset(ResetSpec())
        backend.apply_targets(np.array(initial_backend_state.actuator_targets_rad, dtype=np.float64, order="C"))
        last_result = None
        for _ in range(policy_steps):
            last_result = backend.step_substeps(context.config.physics.decimation)
        if last_result is None:
            raise ValueError("settle duration must include at least one policy step")
        backend_state = last_result.state
        backend_diagnostics = last_result.diagnostics
    # The public backend intentionally exposes only the 19 observed joints.
    # A parallel direct rollout measures the 35 fixed non-task joints, while
    # all task-facing/object/numeric checks below use the public M04 boundary.
    for _ in range(step_count):
        mujoco.mj_step(model, data)
    roles = _source_roles(context)
    fixed_drifts: list[float] = []
    for name, role in roles.items():
        if role != "fixed_non_task":
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        address = int(model.jnt_qposadr[joint_id])
        fixed_drifts.append(abs(float(data.qpos[address] - initial_qpos[address])))
    maximum_fixed_drift = max(fixed_drifts, default=math.inf)
    if maximum_fixed_drift > FIXED_JOINT_SETTLE_TOLERANCE_RAD:
        builder.error(
            "settle.fixed_joint",
            "a fixed non-task joint drifted during zero-action settle",
            measured=maximum_fixed_drift,
            expected=FIXED_JOINT_SETTLE_TOLERANCE_RAD,
            unit="rad",
        )
    pelvis_drift = float(
        np.linalg.norm(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")] - initial_pelvis)
    )
    if pelvis_drift > 1.0e-12:
        builder.error(
            "settle.fixed_base",
            "fixed base moved during zero-action settle",
            measured=pelvis_drift,
            expected=1.0e-12,
            unit="m",
        )
    horizontal_drift = float(
        np.linalg.norm(backend_state.cylinder_position_m[:2] - initial_backend_state.cylinder_position_m[:2])
    )
    vertical_drift = abs(float(backend_state.cylinder_position_m[2] - initial_backend_state.cylinder_position_m[2]))
    tilt = _quat_tilt_rad(backend_state.cylinder_quaternion_wxyz)
    for code, value, limit, unit in (
        ("horizontal", horizontal_drift, CYLINDER_HORIZONTAL_DRIFT_TOLERANCE_M, "m"),
        ("vertical", vertical_drift, CYLINDER_VERTICAL_DRIFT_TOLERANCE_M, "m"),
        ("tilt", tilt, CYLINDER_TILT_TOLERANCE_RAD, "rad"),
    ):
        if value > limit:
            builder.error(
                f"settle.cylinder_{code}",
                f"cylinder {code} drift exceeds the settle limit",
                measured=value,
                expected=limit,
                unit=unit,
            )
    finite = bool(
        np.isfinite(backend_state.joint_positions_rad).all()
        and np.isfinite(backend_state.joint_velocities_rad_s).all()
        and np.isfinite(backend_state.cylinder_position_m).all()
        and np.isfinite(backend_state.cylinder_quaternion_wxyz).all()
        and np.isfinite(backend_state.actuator_forces_n_m).all()
    )
    if not finite:
        builder.error("settle.nonfinite", "zero-action settle produced a non-finite state")
    warnings = [
        {
            "warning_index": warning.code,
            "count": warning.occurrence_count,
            "last_info": warning.last_info,
        }
        for warning in backend_diagnostics.warnings
    ]
    if not backend_diagnostics.healthy:
        builder.error(
            "settle.backend_diagnostics",
            "backend diagnostics reported contact capacity, solver or non-finite failure",
        )
    for warning in warnings:
        builder.error("settle.mujoco_warning", f"MuJoCo warning during settle: {warning}")
    builder.measurements.update(
        duration_s=SETTLE_DURATION_S,
        physics_steps=step_count,
        policy_steps=policy_steps,
        dynamic_backend="MujocoEnvironmentBackend",
        maximum_fixed_non_task_drift_rad=maximum_fixed_drift,
        fixed_base_drift_m=pelvis_drift,
        cylinder_horizontal_drift_m=horizontal_drift,
        cylinder_vertical_drift_m=vertical_drift,
        cylinder_tilt_rad=tilt,
        finite=finite,
        mujoco_warnings=warnings,
    )
    return builder.finish()


def _check_arm_tracking(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "arm_tracking",
        "each right-arm joint tracks real positive and negative small MuJoCo position targets",
    )
    model = context.model
    settle_steps = round(ARM_TRACKING_SETTLE_S / context.config.physics.policy_dt_s)
    observe_steps = round(ARM_TRACKING_OBSERVE_S / context.config.physics.policy_dt_s)
    trials: list[dict[str, object]] = []
    paired_final_positions: dict[str, dict[float, float]] = {}
    scene = context.config.task.scene
    isolated_cylinder_position = (
        scene.table_center_m[0] + 0.25,
        scene.table_center_m[1] + 0.25,
        ResetSpec().cylinder_position_m[2],
    )
    for actuator_id, joint_name in enumerate(context.config.physics.actuator_calibration.arm.joint_names):
        effort_limit = context.config.physics.actuator_calibration.arm.effort_limit_n_m[actuator_id]
        velocity_limit = context.config.physics.actuator_calibration.arm.velocity_limit_rad_s[actuator_id]
        paired_final_positions[joint_name] = {}
        for direction in (-1.0, 1.0):
            with MujocoEnvironmentBackend(context.config, mjcf_path=context.runtime_mjcf) as backend:
                initial_state = backend.reset(ResetSpec(cylinder_position_m=isolated_cylinder_position))
                initial_targets = np.array(initial_state.actuator_targets_rad, dtype=np.float64, order="C")
                backend.apply_targets(initial_targets)
                state = initial_state
                for _ in range(settle_steps):
                    state = backend.step_substeps(context.config.physics.decimation).state
                joint_index = state.joint_names.index(joint_name)
                start_position = float(state.joint_positions_rad[joint_index])
                targets = initial_targets.copy()
                # Measure a true local positive/negative response from the
                # gravity-loaded settled posture, not from the keyframe's
                # controller target which may differ from that posture.
                target = float(start_position + direction * ARM_TRACKING_TARGET_DELTA_RAD)
                target = float(
                    np.clip(
                        target,
                        model.actuator_ctrlrange[actuator_id, 0],
                        model.actuator_ctrlrange[actuator_id, 1],
                    )
                )
                targets[actuator_id] = target
                backend.apply_targets(np.array(targets, dtype=np.float64, order="C"))
                positions: list[float] = []
                velocities: list[float] = []
                forces: list[float] = []
                result = None
                for _ in range(observe_steps):
                    result = backend.step_substeps(context.config.physics.decimation)
                    state = result.state
                    positions.append(float(state.joint_positions_rad[joint_index]))
                    velocities.append(float(state.joint_velocities_rad_s[joint_index]))
                    forces.append(float(state.actuator_forces_n_m[actuator_id]))
                if result is None:
                    raise ValueError("arm tracking duration must contain at least one policy step")
                diagnostics = result.diagnostics
            final_position = positions[-1]
            paired_final_positions[joint_name][direction] = final_position
            steady_error = abs(final_position - target)
            displacement = final_position - start_position
            commanded_displacement = target - start_position
            directional_peak_response, overshoot = tracking_response_metrics(
                positions,
                settled_position_rad=start_position,
                target_position_rad=target,
                direction=direction,
            )
            maximum_velocity = float(np.max(np.abs(velocities)))
            maximum_force = float(np.max(np.abs(forces)))
            warnings = [
                {
                    "warning_index": warning.code,
                    "count": warning.occurrence_count,
                    "last_info": warning.last_info,
                }
                for warning in diagnostics.warnings
            ]
            trial: dict[str, object] = {
                "joint": joint_name,
                "direction": "negative" if direction < 0 else "positive",
                "target_rad": target,
                "start_rad": start_position,
                "final_rad": final_position,
                "signed_displacement_rad": displacement,
                "commanded_displacement_rad": commanded_displacement,
                "directional_peak_response_rad": directional_peak_response,
                "steady_state_error_rad": steady_error,
                "overshoot_rad": overshoot,
                "maximum_velocity_rad_s": maximum_velocity,
                "velocity_limit_rad_s": velocity_limit,
                "maximum_actuator_force_n_m": maximum_force,
                "effort_limit_n_m": effort_limit,
                "mujoco_warnings": warnings,
            }
            trials.append(trial)
            if direction * displacement <= 0.0:
                builder.error(
                    "tracking.response_direction",
                    f"joint {joint_name!r} terminal response opposed its local target direction",
                    object_name=joint_name,
                    measured=displacement,
                    expected="negative for a negative target; positive for a positive target",
                    unit="rad",
                )
            if steady_error > ARM_TRACKING_STEADY_ERROR_LIMIT_RAD:
                builder.error(
                    "tracking.steady_error",
                    f"joint {joint_name!r} steady-state error exceeds its measured calibration gate",
                    object_name=joint_name,
                    measured=steady_error,
                    expected=ARM_TRACKING_STEADY_ERROR_LIMIT_RAD,
                    unit="rad",
                )
            if overshoot > ARM_TRACKING_OVERSHOOT_LIMIT_RAD:
                builder.error(
                    "tracking.overshoot",
                    f"joint {joint_name!r} overshoot exceeds its measured calibration gate",
                    object_name=joint_name,
                    measured=overshoot,
                    expected=ARM_TRACKING_OVERSHOOT_LIMIT_RAD,
                    unit="rad",
                )
            if maximum_velocity > velocity_limit + 1.0e-9:
                builder.error(
                    "tracking.velocity",
                    f"joint {joint_name!r} exceeded the configured velocity limit",
                    object_name=joint_name,
                    measured=maximum_velocity,
                    expected=velocity_limit,
                    unit="rad/s",
                )
            if maximum_force > effort_limit + 1.0e-9:
                builder.error(
                    "tracking.effort",
                    f"joint {joint_name!r} exceeded the configured effort limit",
                    object_name=joint_name,
                    measured=maximum_force,
                    expected=effort_limit,
                    unit="N*m",
                )
            for warning in warnings:
                builder.error(
                    "tracking.mujoco_warning",
                    f"MuJoCo warning during {joint_name!r} tracking: {warning}",
                    object_name=joint_name,
                )
            if not diagnostics.healthy:
                builder.error(
                    "tracking.backend_diagnostics",
                    f"backend diagnostics were unhealthy during {joint_name!r} tracking",
                    object_name=joint_name,
                )
    for joint_name, final_positions in paired_final_positions.items():
        negative = final_positions.get(-1.0)
        positive = final_positions.get(1.0)
        if negative is None or positive is None or positive <= negative:
            builder.error(
                "tracking.paired_response",
                f"joint {joint_name!r} positive target did not finish above its negative target",
                object_name=joint_name,
                measured=f"negative={negative}, positive={positive}",
                expected="positive > negative",
                unit="rad",
            )
    builder.measurements.update(
        target_delta_rad=ARM_TRACKING_TARGET_DELTA_RAD,
        pre_settle_duration_s=ARM_TRACKING_SETTLE_S,
        observation_duration_s=ARM_TRACKING_OBSERVE_S,
        dynamic_backend="MujocoEnvironmentBackend",
        isolated_cylinder_position_m=list(isolated_cylinder_position),
        steady_error_limit_rad=ARM_TRACKING_STEADY_ERROR_LIMIT_RAD,
        overshoot_limit_rad=ARM_TRACKING_OVERSHOOT_LIMIT_RAD,
        trials=trials,
    )
    return builder.finish()


def _check_hand_motion(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "hand_motion",
        "right-hand open, half-close and full-close targets remain continuous, limited and mimic-consistent",
    )
    # Compile from the runtime path for this dynamic check instead of reusing
    # the structural-check model.  This makes the hand gate exercise the exact
    # committed runtime that each backend instance below independently loads.
    model = mujoco.MjModel.from_xml_path(str(context.runtime_mjcf.resolve()))
    hand_slice = slice(7, 13)
    hold_steps = round(HAND_HOLD_DURATION_S / context.config.physics.policy_dt_s)
    scene = context.config.task.scene
    isolated_cylinder_position = (
        scene.table_center_m[0] + 0.25,
        scene.table_center_m[1] + 0.25,
        ResetSpec().cylinder_position_m[2],
    )
    poses: list[dict[str, object]] = []
    previous_targets: np.ndarray | None = None
    for label, fraction in (("open", 0.0), ("half_close", 0.5), ("full_close", 1.0)):
        lower = np.asarray(model.actuator_ctrlrange[hand_slice, 0])
        upper = np.asarray(model.actuator_ctrlrange[hand_slice, 1])
        targets = lower + fraction * (upper - lower)
        transition_steps = max(
            1,
            round(context.config.task.option.closing_time_s / context.config.physics.policy_dt_s),
        )
        with MujocoEnvironmentBackend(context.config, mjcf_path=context.runtime_mjcf) as backend:
            # T14 isolates actuator/mimic motion from object contact.  Contact
            # behaviour for the standard and endpoint cylinders is exercised
            # separately and explicitly by T15.
            state = backend.reset(ResetSpec(cylinder_position_m=isolated_cylinder_position))
            physical_targets = np.array(state.actuator_targets_rad, dtype=np.float64, order="C")
            initial = physical_targets[hand_slice].copy()
            maximum_target_step = 0.0
            for step in range(transition_steps):
                alpha = (step + 1) / transition_steps
                next_targets = initial + alpha * (targets - initial)
                maximum_target_step = max(
                    maximum_target_step,
                    float(np.max(np.abs(next_targets - physical_targets[hand_slice]))),
                )
                physical_targets[hand_slice] = next_targets
                backend.apply_targets(np.array(physical_targets, dtype=np.float64, order="C"))
                state = backend.step_substeps(context.config.physics.decimation).state
            result = None
            for _ in range(hold_steps):
                result = backend.step_substeps(context.config.physics.decimation)
                state = result.state
            if result is None:
                raise ValueError("hand hold duration must contain at least one policy step")
            diagnostics = result.diagnostics
        active_errors: list[float] = []
        active_positions: list[float] = []
        active_names = context.config.physics.actuator_calibration.hand.joint_names
        for actuator_id, joint_name in enumerate(active_names, start=7):
            position = float(state.joint_positions_rad[state.joint_names.index(joint_name)])
            active_positions.append(position)
            active_errors.append(abs(position - float(state.actuator_targets_rad[actuator_id])))
        mimic_errors: list[float] = []
        for follower_name, (leader_name, multiplier, offset) in sorted(
            (item for item in context.inventory.mimic.items() if item[0].startswith("right_")),
        ):
            follower = float(state.joint_positions_rad[state.joint_names.index(follower_name)])
            leader = float(state.joint_positions_rad[state.joint_names.index(leader_name)])
            expected = multiplier * leader + offset
            mimic_errors.append(abs(follower - expected))
        maximum_active_error = max(active_errors, default=math.inf)
        maximum_mimic_error = max(mimic_errors, default=math.inf)
        finite = bool(
            np.isfinite(state.joint_positions_rad).all()
            and np.isfinite(state.joint_velocities_rad_s).all()
            and np.isfinite(state.actuator_forces_n_m).all()
        )
        observed_hand_names = context.config.task.observation.observed_joint_names[7:]
        observed_hand_positions = {
            name: float(state.joint_positions_rad[state.joint_names.index(name)]) for name in observed_hand_names
        }
        limited = all(
            context.inventory.joint_by_name[name].lower - HAND_JOINT_LIMIT_TOLERANCE_RAD
            <= position
            <= context.inventory.joint_by_name[name].upper + HAND_JOINT_LIMIT_TOLERANCE_RAD
            for name, position in observed_hand_positions.items()
        )
        monotonic_targets = previous_targets is None or bool(np.all(targets >= previous_targets - 1.0e-12))
        warnings = [
            {
                "warning_index": warning.code,
                "count": warning.occurrence_count,
                "last_info": warning.last_info,
            }
            for warning in diagnostics.warnings
        ]
        poses.append(
            {
                "pose": label,
                "fraction": fraction,
                "targets_rad": targets.tolist(),
                "active_positions_rad": active_positions,
                "observed_hand_positions_rad": observed_hand_positions,
                "maximum_active_error_rad": maximum_active_error,
                "maximum_mimic_error_rad": maximum_mimic_error,
                "maximum_target_step_rad": maximum_target_step,
                "within_limits": limited,
                "finite": finite,
                "mujoco_warnings": warnings,
            }
        )
        if maximum_active_error > HAND_ACTIVE_ERROR_LIMIT_RAD:
            builder.error(
                "hand.active_tracking",
                f"{label} active-hand target error exceeds the gate",
                object_name=label,
                measured=maximum_active_error,
                expected=HAND_ACTIVE_ERROR_LIMIT_RAD,
                unit="rad",
            )
        if maximum_mimic_error > HAND_MIMIC_ERROR_LIMIT_RAD:
            builder.error(
                "hand.mimic_tracking",
                f"{label} mimic relationship error exceeds the gate",
                object_name=label,
                measured=maximum_mimic_error,
                expected=HAND_MIMIC_ERROR_LIMIT_RAD,
                unit="rad",
            )
        if not limited:
            builder.error(
                "hand.limit",
                f"{label} produced an active or mimic hand joint outside its source limit",
                object_name=label,
            )
        if not monotonic_targets:
            builder.error(
                "hand.continuity",
                f"{label} targets are not monotonic relative to the preceding pose",
                object_name=label,
            )
        if not finite:
            builder.error("hand.nonfinite", f"{label} produced a non-finite state", object_name=label)
        for warning in warnings:
            builder.error("hand.mujoco_warning", f"MuJoCo warning during {label}: {warning}", object_name=label)
        if not diagnostics.healthy:
            builder.error(
                "hand.backend_diagnostics",
                f"backend diagnostics were unhealthy during {label}",
                object_name=label,
            )
        previous_targets = targets
    builder.measurements.update(
        transition_duration_s=context.config.task.option.closing_time_s,
        hold_duration_s=HAND_HOLD_DURATION_S,
        active_error_limit_rad=HAND_ACTIVE_ERROR_LIMIT_RAD,
        mimic_error_limit_rad=HAND_MIMIC_ERROR_LIMIT_RAD,
        joint_limit_tolerance_rad=HAND_JOINT_LIMIT_TOLERANCE_RAD,
        dynamic_backend="MujocoEnvironmentBackend",
        isolated_cylinder_position_m=list(isolated_cylinder_position),
        poses=poses,
    )
    return builder.finish()


def _set_cylinder_properties_and_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    radius_m: float,
    height_m: float,
    mass_kg: float,
    friction: float,
    position_m: np.ndarray,
) -> None:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cylinder_collision")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cylinder_free_joint")
    half_side = radius_m / np.sqrt(2.0)
    model.geom_size[geom_id, :3] = (half_side, half_side, height_m / 2.0)
    model.geom_friction[geom_id, 0] = friction
    model.body_mass[body_id] = mass_kg
    model.body_inertia[body_id] = (
        mass_kg * (2.0 * radius_m * radius_m + height_m * height_m) / 12.0,
        mass_kg * (2.0 * radius_m * radius_m + height_m * height_m) / 12.0,
        (1.0 / 3.0) * mass_kg * radius_m * radius_m,
    )
    qpos_address = int(model.jnt_qposadr[joint_id])
    dof_address = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_address : qpos_address + 3] = position_m
    data.qpos[qpos_address + 3 : qpos_address + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_address : dof_address + 6] = 0.0
    mujoco.mj_forward(model, data)


def _cylinder_contacted_groups(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    groups: Mapping[str, set[str]],
) -> set[str]:
    cylinder_name = "cylinder_collision"
    contacted: set[str] = set()
    for index in range(data.ncon):
        first, second = _contact_pair_names(model, data.contact[index])
        if first == cylinder_name:
            contacted.update(_groups_for_geom(second, groups))
        elif second == cylinder_name:
            contacted.update(_groups_for_geom(first, groups))
    return contacted


def _check_contact_probes(context: _ValidationContext) -> ValidationCheck:
    builder = _CheckBuilder(
        "contact_probes",
        "standard and endpoint cylinders expose observation and bilateral grasp groups",
    )
    groups = _group_members(context)
    required_groups = {"right_hand_observation_links", "thumb_side", "four_finger_opposing", "cylinder"}
    if not required_groups <= set(groups):
        builder.error(
            "probe.groups",
            "contact group index is incomplete",
            measured=str(sorted(required_groups - set(groups))),
            expected="none missing",
        )
        return builder.finish()
    cases = (
        ("standard", 0.015, 0.100, 0.115, 0.9),
        (
            "minimum_endpoint",
            context.config.randomization.radius.minimum,
            context.config.randomization.height.minimum,
            context.config.randomization.mass.minimum,
            context.config.randomization.hand_sliding_friction.minimum,
        ),
        (
            "maximum_endpoint",
            context.config.randomization.radius.maximum,
            context.config.randomization.height.maximum,
            context.config.randomization.mass.maximum,
            context.config.randomization.hand_sliding_friction.maximum,
        ),
    )
    records: list[dict[str, object]] = []
    for label, radius, height, mass, friction in cases:
        model = mujoco.MjModel.from_xml_path(str(context.runtime_mjcf.resolve()))
        data = mujoco.MjData(model)
        _reset_home(model, data)
        thumb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_thumb_4_collision")
        opposing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_index_1_collision")
        bilateral_position = (np.asarray(data.geom_xpos[thumb_id]) + np.asarray(data.geom_xpos[opposing_id])) / 2.0
        _set_cylinder_properties_and_pose(
            model,
            data,
            radius_m=radius,
            height_m=height,
            mass_kg=mass,
            friction=friction,
            position_m=bilateral_position,
        )
        bilateral_groups = _cylinder_contacted_groups(model, data, groups)
        bilateral = "thumb_side" in bilateral_groups and "four_finger_opposing" in bilateral_groups
        observation_names = tuple(sorted(groups["right_hand_observation_links"]))
        observed_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in observation_names]
        observation_query = (
            len(observed_body_ids) == 12
            and len(set(observed_body_ids)) == 12
            and all(body_id >= 0 for body_id in observed_body_ids)
        )
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        records.append(
            {
                "case": label,
                "radius_m": radius,
                "height_m": height,
                "mass_kg": mass,
                "hand_sliding_friction": friction,
                "bilateral_groups": sorted(bilateral_groups),
                "bilateral_grasp_query": bilateral,
                "observation_link_query": observation_query,
                "observation_body_names": list(observation_names),
                "finite": finite,
            }
        )
        if not bilateral:
            builder.error(
                "probe.bilateral",
                f"{label} cylinder did not produce queryable thumb/opposing contacts",
                object_name=label,
            )
        if not observation_query:
            builder.error(
                "probe.observation", f"{label} cannot resolve all twelve observation contact bodies", object_name=label
            )
        if not finite:
            builder.error("probe.nonfinite", f"{label} contact probe produced non-finite state", object_name=label)
    builder.measurements.update(cases=records, case_count=len(records))
    return builder.finish()


def _preflight_failure(config: ResolvedRunConfig, error: Exception) -> ValidationReport:
    builder = _CheckBuilder("preflight", "asset files and MuJoCo model load successfully")
    builder.error("preflight.load", f"asset validation could not initialize: {type(error).__name__}: {error}")
    return ValidationReport(
        asset_version=config.asset.asset_version,
        runtime_mjcf_sha256="unavailable",
        checks=(builder.finish(),),
    )


def validate_assets(
    config: ResolvedRunConfig,
    *,
    repository_root: Path = pipeline.REPOSITORY_ROOT,
    runtime_mjcf: Path | None = None,
) -> ValidationReport:
    """Run every M03 check in a stable order using real MuJoCo execution."""

    try:
        context = _make_context(config, repository_root=repository_root, runtime_mjcf=runtime_mjcf)
    except (OSError, ValueError, ET.ParseError, pipeline.AssetPipelineError) as error:
        return _preflight_failure(config, error)
    checks: tuple[Callable[[_ValidationContext], ValidationCheck], ...] = (
        lambda current: compare_joint_structure(current.inventory, current.runtime_root),
        lambda current: compare_joint_axes_and_limits(current.inventory, current.runtime_root),
        _check_actuators,
        _check_hand_mimic,
        _check_named_objects,
        _check_runtime_variants,
        _check_stl_visual_only,
        _check_inertials,
        _check_initial_pose,
        _check_collision_coverage,
        _check_initial_penetration,
        _check_zero_action_settle,
        _check_arm_tracking,
        _check_hand_motion,
        _check_contact_probes,
    )
    results: list[ValidationCheck] = []
    for check in checks:
        try:
            results.append(check(context))
        except (
            OSError,
            ValueError,
            TypeError,
            IndexError,
            KeyError,
            FloatingPointError,
            pipeline.AssetPipelineError,
        ) as error:
            builder = _CheckBuilder(
                getattr(check, "__name__", "validation_check"), "validation check completed without an internal error"
            )
            builder.error("validator.internal", f"{type(error).__name__}: {error}")
            results.append(builder.finish())
    return ValidationReport(
        asset_version=config.asset.asset_version,
        runtime_mjcf_sha256=pipeline.sha256_file(context.runtime_mjcf),
        checks=tuple(results),
    )


def run(config: ResolvedRunConfig, args: argparse.Namespace) -> int:
    """CLI runtime: print terminal results and optionally write canonical JSON/SVG."""

    report = validate_assets(config)
    report_path = getattr(args, "report", None)
    artifacts: dict[str, str] = {}
    if isinstance(report_path, Path):
        coverage = next((check for check in report.checks if check.check_id == "collision_coverage"), None)
        if coverage is not None:
            svg_path = report_path.with_name(f"{report_path.stem}.collision_coverage.svg")
            write_collision_coverage_svg(coverage, svg_path)
            artifacts["collision_coverage_svg"] = svg_path.name
        report = ValidationReport(
            asset_version=report.asset_version,
            runtime_mjcf_sha256=report.runtime_mjcf_sha256,
            checks=report.checks,
            artifacts=artifacts,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report.to_json(), encoding="utf-8")
    for line in report.terminal_lines():
        print(line)
    return 0 if report.passed and report.unexplained_warning_count == 0 else 1
