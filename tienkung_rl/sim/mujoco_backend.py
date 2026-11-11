"""Native MuJoCo CPU implementation of one headless physics environment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

import mujoco  # type: ignore[import-untyped]  # MuJoCo 3.11 does not ship a PEP 561 marker.
import numpy as np
import yaml  # type: ignore[import-untyped]  # PyYAML has no bundled type information.
from numpy.typing import NDArray

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import sha256_fingerprint
from tienkung_rl.paths import PROJECT_ROOT as REPOSITORY_ROOT

from .backend import EnvironmentBackend
from .state import (
    ActuatorControlRanges,
    ContactFrame,
    ContactSample,
    CylinderParameters,
    FloatArray,
    PhysicsState,
    PhysicsStepResult,
    PhysicsSubstepTrace,
    ResetSpec,
    SimulationDiagnostics,
    SimulationWarning,
    SiteHingeKinematics,
    is_illegal_contact,
    mujoco_quaternion_to_wxyz,
)

_HOME_KEY_NAME = "robot_and_scene_home"
_CYLINDER_BODY_NAME = "cylinder"
_CYLINDER_GEOM_NAME = "cylinder_collision"
_CYLINDER_JOINT_NAME = "cylinder_free_joint"
_TCP_SITE_NAME = "right_tcp_site"
_GRASP_CENTER_SITE_NAME = "grasp_center_site"
_REQUIRED_SITE_ORDER = (
    "target_tcp_site",
    "right_tcp_site",
    "grasp_center_site",
    "approach_direction_site",
    "cylinder_axis_site",
)
_EXPECTED_WARNING_COUNT = int(mujoco.mjtWarning.mjNWARNING)
_SOLVER_WARNING_CODES = frozenset(
    {
        int(mujoco.mjtWarning.mjWARN_INERTIA),
        int(mujoco.mjtWarning.mjWARN_CNSTRFULL),
        int(mujoco.mjtWarning.mjWARN_BADQPOS),
        int(mujoco.mjtWarning.mjWARN_BADQVEL),
        int(mujoco.mjtWarning.mjWARN_BADQACC),
        int(mujoco.mjtWarning.mjWARN_BADCTRL),
    }
)


class BackendError(RuntimeError):
    """Base class for failures at the named physics boundary."""


class BackendContractError(BackendError):
    """The model, name index, config, reset, or target contract is invalid."""


class BackendClosedError(BackendError):
    """A method requiring live MuJoCo state was used after ``close``."""


class NonFinitePhysicsError(BackendError):
    """MuJoCo exposed a non-finite state instead of a valid snapshot."""

    def __init__(self, message: str, diagnostics: SimulationDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class SimulationStepError(BackendError):
    """MuJoCo raised while advancing one physical substep."""

    def __init__(self, message: str, diagnostics: SimulationDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class ModelNameCache:
    """Validated names and integer addresses resolved exactly once at startup."""

    object_ids: Mapping[str, Mapping[str, int]]
    observed_joint_names: tuple[str, ...]
    observed_joint_qpos_addresses: tuple[int, ...]
    observed_joint_dof_addresses: tuple[int, ...]
    body_names: tuple[str, ...]
    body_ids: tuple[int, ...]
    site_names: tuple[str, ...]
    site_ids: tuple[int, ...]
    actuator_names: tuple[str, ...]
    actuator_ids: tuple[int, ...]
    home_key_id: int
    cylinder_body_id: int
    cylinder_geom_id: int
    cylinder_joint_id: int
    tcp_site_id: int
    grasp_center_site_id: int


@dataclass(frozen=True, slots=True)
class _ContactGroups:
    geom_groups: Mapping[str, tuple[str, ...]]
    observed_hand_body_names: tuple[str, ...]
    observed_hand_body_ids: tuple[int, ...]
    illegal_geom_groups: frozenset[str]
    hand_geom_ids: tuple[int, ...]
    table_geom_ids: tuple[int, ...]


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BackendContractError(f"cannot read name index {path}: {error}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BackendContractError(f"name index {path} must contain a string-keyed object")
    return cast(Mapping[str, object], value)


def _string_int_mapping(value: object, path: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise BackendContractError(f"{path} must be an object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise BackendContractError(f"{path} names must be non-empty strings")
        if type(item) is not int or item < 0:
            raise BackendContractError(f"{path}.{key} must be a non-negative integer ID")
        result[key] = item
    if len(set(result.values())) != len(result):
        raise BackendContractError(f"{path} contains duplicate IDs")
    return result


def _strict_name_cache(
    model: mujoco.MjModel,
    config: ResolvedRunConfig,
    name_index_path: Path,
    *,
    allow_visual_geoms: bool = False,
) -> tuple[ModelNameCache, Mapping[str, object]]:
    index = _read_json_mapping(name_index_path)
    if index.get("schema_version") != "tienkung2_pro_name_index_v1":
        raise BackendContractError("unsupported or missing name-index schema_version")
    indices_value = index.get("indices")
    if not isinstance(indices_value, dict):
        raise BackendContractError("name index is missing indices")

    specifications: tuple[tuple[str, mujoco.mjtObj, int], ...] = (
        ("bodies", mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        ("joints", mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        ("geoms", mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        ("sites", mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        ("actuators", mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        ("equalities", mujoco.mjtObj.mjOBJ_EQUALITY, model.neq),
        ("keys", mujoco.mjtObj.mjOBJ_KEY, model.nkey),
    )
    object_ids: dict[str, Mapping[str, int]] = {}
    for namespace, object_type, count in specifications:
        expected = _string_int_mapping(indices_value.get(namespace), f"indices.{namespace}")
        actual: dict[str, int] = {}
        actual_order: list[str] = []
        for object_id in range(count):
            name = mujoco.mj_id2name(model, object_type, object_id)
            if name is None:
                raise BackendContractError(f"runtime model has unnamed {namespace[:-1]} at ID {object_id}")
            actual[name] = object_id
            actual_order.append(name)
        if namespace == "geoms" and allow_visual_geoms:
            expected_order = tuple(name for name, _ in sorted(expected.items(), key=lambda item: item[1]))
            # Viewer-only mesh geoms can be interleaved with physical geoms
            # inside the same body.  Compare the ordered subsequence of names
            # from the headless contract instead of relying on a name suffix.
            physical_order = tuple(name for name in actual_order if name in expected)
            visual_names = tuple(name for name in actual_order if name not in expected)
            if physical_order != expected_order or not visual_names:
                raise BackendContractError(
                    "viewer geoms must preserve the headless physical order and add named visual geoms"
                )
            for name in visual_names:
                geom_id = actual[name]
                if (
                    int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH)
                    or int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                    or int(model.geom_group[geom_id]) != 1
                ):
                    raise BackendContractError(f"viewer visual geom {name!r} participates in physics")
            object_ids[namespace] = MappingProxyType(actual)
            continue
        if expected != actual:
            missing = tuple(sorted(set(expected) - set(actual)))
            extra = tuple(sorted(set(actual) - set(expected)))
            wrong_ids = tuple(sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name]))
            raise BackendContractError(
                f"runtime {namespace} do not match versioned name index: "
                f"missing={missing}, extra={extra}, wrong_ids={wrong_ids}"
            )
        expected_order = tuple(name for name, _ in sorted(expected.items(), key=lambda item: item[1]))
        if expected_order != tuple(actual_order):
            raise BackendContractError(f"runtime {namespace} order differs from name index")
        object_ids[namespace] = MappingProxyType(expected)

    # Sensors are an explicit zero-length contract in the v1 runtime asset.
    if model.nsensor != 0:
        sensor_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, index) for index in range(model.nsensor)
        )
        raise BackendContractError(f"runtime sensors must be empty, got {sensor_names}")

    observed_joint_names = config.task.observation.observed_joint_names
    joint_ids = object_ids["joints"]
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in observed_joint_names:
        if name not in joint_ids:
            raise BackendContractError(f"observed joint {name!r} is absent from runtime model")
        joint_id = joint_ids[name]
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise BackendContractError(f"observed joint {name!r} must be a scalar hinge")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))

    body_names = tuple(name for name, _ in sorted(object_ids["bodies"].items(), key=lambda item: item[1]))
    body_ids = tuple(object_ids["bodies"][name] for name in body_names)
    site_names = tuple(name for name, _ in sorted(object_ids["sites"].items(), key=lambda item: item[1]))
    if site_names != _REQUIRED_SITE_ORDER:
        raise BackendContractError(f"runtime site order must be {_REQUIRED_SITE_ORDER}, got {site_names}")
    site_ids = tuple(object_ids["sites"][name] for name in site_names)
    expected_actuator_names = tuple(
        f"{name}_position"
        for name in (
            *config.physics.actuator_calibration.arm.joint_names,
            *config.physics.actuator_calibration.hand.joint_names,
        )
    )
    actuator_names = tuple(name for name, _ in sorted(object_ids["actuators"].items(), key=lambda item: item[1]))
    if actuator_names != expected_actuator_names or len(actuator_names) != 13:
        raise BackendContractError("runtime actuator order must be the configured 7 arm then 6 active-hand actuators")
    actuator_ids = tuple(object_ids["actuators"][name] for name in actuator_names)

    cache = ModelNameCache(
        object_ids=MappingProxyType(object_ids),
        observed_joint_names=observed_joint_names,
        observed_joint_qpos_addresses=tuple(qpos_addresses),
        observed_joint_dof_addresses=tuple(dof_addresses),
        body_names=body_names,
        body_ids=body_ids,
        site_names=site_names,
        site_ids=site_ids,
        actuator_names=actuator_names,
        actuator_ids=actuator_ids,
        home_key_id=object_ids["keys"][_HOME_KEY_NAME],
        cylinder_body_id=object_ids["bodies"][_CYLINDER_BODY_NAME],
        cylinder_geom_id=object_ids["geoms"][_CYLINDER_GEOM_NAME],
        cylinder_joint_id=object_ids["joints"][_CYLINDER_JOINT_NAME],
        tcp_site_id=object_ids["sites"][_TCP_SITE_NAME],
        grasp_center_site_id=object_ids["sites"][_GRASP_CENTER_SITE_NAME],
    )
    return cache, index


def _read_contact_phase_rules(path: Path) -> Mapping[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BackendContractError(f"cannot read contact groups {path}: {error}") from error
    if not isinstance(value, dict):
        raise BackendContractError("contact group declaration must be an object")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BackendContractError(f"{path} must be a sequence of non-empty names")
    result = cast(tuple[str, ...], tuple(value))
    if len(set(result)) != len(result):
        raise BackendContractError(f"{path} must not contain duplicates")
    return result


def _contact_groups(
    model: mujoco.MjModel,
    cache: ModelNameCache,
    name_index: Mapping[str, object],
    declaration_path: Path,
) -> _ContactGroups:
    raw_groups = name_index.get("contact_groups")
    if not isinstance(raw_groups, dict):
        raise BackendContractError("name index is missing contact_groups")
    geom_groups_by_name: dict[str, list[str]] = {}
    group_members: dict[str, tuple[str, ...]] = {}
    group_kinds: dict[str, str] = {}
    for group_name, raw_group in raw_groups.items():
        if not isinstance(group_name, str) or not isinstance(raw_group, dict):
            raise BackendContractError("contact groups must be named objects")
        kind = raw_group.get("kind")
        if not isinstance(kind, str):
            raise BackendContractError(f"contact group {group_name!r} has no kind")
        members = _string_sequence(raw_group.get("members"), f"contact_groups.{group_name}.members")
        group_members[group_name] = members
        group_kinds[group_name] = kind
        if kind == "geom":
            for member in members:
                if member not in cache.object_ids["geoms"]:
                    raise BackendContractError(f"contact group {group_name!r} names unknown geom {member!r}")
                geom_groups_by_name.setdefault(member, []).append(group_name)
        elif kind == "body":
            for member in members:
                if member not in cache.object_ids["bodies"]:
                    raise BackendContractError(f"contact group {group_name!r} names unknown body {member!r}")
        elif kind != "collision_mask":
            raise BackendContractError(f"contact group {group_name!r} has unsupported kind {kind!r}")

    declaration = _read_contact_phase_rules(declaration_path)
    if declaration.get("schema_version") != "tienkung2_pro_contact_groups_v1":
        raise BackendContractError("unsupported contact-group declaration version")
    declared_groups = declaration.get("groups")
    if not isinstance(declared_groups, dict) or declared_groups != raw_groups:
        raise BackendContractError("runtime name-index contact groups differ from versioned declaration")
    phase_rules = declaration.get("phase_rules")
    if not isinstance(phase_rules, dict) or set(phase_rules) != {"reach", "script"}:
        raise BackendContractError("contact declaration must contain reach and script phase rules")
    illegal_groups: set[str] = set()
    for phase_name in ("reach", "script"):
        raw_rule = phase_rules.get(phase_name)
        if not isinstance(raw_rule, dict):
            raise BackendContractError(f"contact phase rule {phase_name!r} must be an object")
        for group_name in _string_sequence(
            raw_rule.get("illegal_robot_groups"),
            f"phase_rules.{phase_name}.illegal_robot_groups",
        ):
            if group_name not in group_members or group_kinds[group_name] != "geom":
                raise BackendContractError(f"illegal contact group {group_name!r} is not a declared geom group")
            illegal_groups.add(group_name)

    hand_names = group_members.get("right_hand_observation_links")
    if hand_names is None or group_kinds.get("right_hand_observation_links") != "body":
        raise BackendContractError("right_hand_observation_links must be a declared body group")
    if len(hand_names) != 12:
        raise BackendContractError("right_hand_observation_links must contain exactly 12 bodies")
    hand_ids = tuple(cache.object_ids["bodies"][name] for name in hand_names)
    hand_geom_names = tuple(
        dict.fromkeys(
            (
                *group_members["thumb_side"],
                *group_members["four_finger_opposing"],
            )
        )
    )
    table_geom_names = group_members.get("table")
    if table_geom_names is None:
        raise BackendContractError("table contact group is missing")
    geom_groups: dict[str, tuple[str, ...]] = {
        name: tuple(sorted(groups)) for name, groups in geom_groups_by_name.items()
    }
    return _ContactGroups(
        geom_groups=MappingProxyType(geom_groups),
        observed_hand_body_names=hand_names,
        observed_hand_body_ids=hand_ids,
        illegal_geom_groups=frozenset(illegal_groups),
        hand_geom_ids=tuple(cache.object_ids["geoms"][name] for name in hand_geom_names),
        table_geom_ids=tuple(cache.object_ids["geoms"][name] for name in table_geom_names),
    )


class MujocoEnvironmentBackend(EnvironmentBackend):
    """One independent native MuJoCo model/data pair with no render objects."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        mjcf_path: str | Path | None = None,
        name_index_path: str | Path | None = None,
        contact_groups_path: str | Path | None = None,
        config_fingerprint: str | None = None,
    ) -> None:
        config.validate()
        configured_path = Path(config.asset.runtime_mjcf_path)
        runtime_path = Path(mjcf_path) if mjcf_path is not None else configured_path
        if not runtime_path.is_absolute():
            runtime_path = REPOSITORY_ROOT / runtime_path
        runtime_path = runtime_path.resolve()
        index_path = Path(name_index_path) if name_index_path is not None else runtime_path.with_name("name_index.json")
        if not index_path.is_absolute():
            index_path = REPOSITORY_ROOT / index_path
        declaration_path = (
            Path(contact_groups_path)
            if contact_groups_path is not None
            else runtime_path.parents[1] / "patches" / "contact_groups.yaml"
        )
        if not declaration_path.is_absolute():
            declaration_path = REPOSITORY_ROOT / declaration_path
        try:
            model = mujoco.MjModel.from_xml_path(str(runtime_path))
            data = mujoco.MjData(model)
        except (OSError, ValueError) as error:
            raise BackendContractError(f"cannot load MuJoCo runtime model {runtime_path}: {error}") from error

        self._config = config
        self._runtime_path = runtime_path
        self._model: mujoco.MjModel | None = model
        self._data: mujoco.MjData | None = data
        self._config_fingerprint = config_fingerprint if config_fingerprint is not None else sha256_fingerprint(config)
        if not self._config_fingerprint:
            raise BackendContractError("config_fingerprint must not be empty")
        configured_viewer = Path(config.asset.viewer_mjcf_path)
        if not configured_viewer.is_absolute():
            configured_viewer = REPOSITORY_ROOT / configured_viewer
        cache, name_index = _strict_name_cache(
            model,
            config,
            index_path.resolve(),
            allow_visual_geoms=runtime_path == configured_viewer.resolve(),
        )
        self._names = cache
        self._contact_groups = _contact_groups(
            model,
            cache,
            name_index,
            declaration_path.resolve(),
        )
        self._baseline_geom_friction = np.array(model.geom_friction, dtype=np.float64, copy=True)
        self._cylinder_parameters = CylinderParameters(
            # ``radius_m`` remains the square prism circumradius so existing
            # randomization/manifest semantics stay stable.
            radius_m=float(np.sqrt(2.0) * model.geom_size[cache.cylinder_geom_id, 0]),
            height_m=2.0 * float(model.geom_size[cache.cylinder_geom_id, 2]),
            mass_kg=float(model.body_mass[cache.cylinder_body_id]),
            hand_sliding_friction=config.physics.cylinder_material.sliding_friction,
            table_sliding_friction=config.physics.cylinder_material.sliding_friction,
            torsional_friction=config.physics.cylinder_material.torsional_friction,
            rolling_friction=config.physics.cylinder_material.rolling_friction,
        )
        self._episode_physics_steps = 0
        self._total_physics_steps = 0
        self._warning_counts = [0] * _EXPECTED_WARNING_COUNT
        self._warning_events: list[SimulationWarning] = []
        self._contact_capacity_exceeded = False
        self._solver_anomaly = False
        self._nonfinite_state = False
        self._last_contacts = self._empty_contact_frame()
        self.reset(ResetSpec(cylinder=self._cylinder_parameters))

    @property
    def is_closed(self) -> bool:
        return self._model is None

    @property
    def render_context_created(self) -> bool:
        """Headless backends never allocate a camera, viewer, or GL context."""

        return False

    @property
    def runtime_path(self) -> Path:
        return self._runtime_path

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def name_cache(self) -> ModelNameCache:
        """Expose immutable resolved names, never the mutable MuJoCo arrays."""

        return self._names

    def native_viewer_handles(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        """Return the live pair solely for ``mujoco.viewer.launch_passive``.

        Task and diagnostic code must continue to consume detached named
        snapshots.  The viewer may synchronize rendering but must not mutate
        either object.
        """

        return self._live()

    def _live(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = self._model
        data = self._data
        if model is None or data is None:
            raise BackendClosedError("MuJoCo backend is closed")
        return model, data

    def _empty_contact_frame(self) -> ContactFrame:
        time_s = 0.0
        if self._data is not None:
            time_s = float(self._data.time)
        return ContactFrame(
            contacts=(),
            substeps_collected=0,
            hand_link_names=self._contact_groups.observed_hand_body_names
            if hasattr(self, "_contact_groups")
            else self._config.task.observation.contact_link_names,
            hand_contact_force_norms_n=np.zeros(12, dtype=np.float64),
            maximum_contact_force_n=0.0,
            maximum_contact_impulse_n_s=0.0,
            maximum_illegal_contact_impulse_n_s=0.0,
            contact_capacity=self._config.physics.contact.maximum_contacts,
            maximum_contacts_seen=0,
            physics_step=self._total_physics_steps,
            simulation_time_s=time_s,
        )

    def _apply_cylinder_parameters(self, parameters: CylinderParameters) -> None:
        model, data = self._live()
        randomization = self._config.randomization
        ranges = (
            ("radius_m", parameters.radius_m, randomization.radius.minimum, randomization.radius.maximum),
            ("height_m", parameters.height_m, randomization.height.minimum, randomization.height.maximum),
            ("mass_kg", parameters.mass_kg, randomization.mass.minimum, randomization.mass.maximum),
            (
                "hand_sliding_friction",
                parameters.hand_sliding_friction,
                randomization.hand_sliding_friction.minimum,
                randomization.hand_sliding_friction.maximum,
            ),
            (
                "table_sliding_friction",
                parameters.table_sliding_friction,
                randomization.table_sliding_friction.minimum,
                randomization.table_sliding_friction.maximum,
            ),
        )
        for name, value, lower, upper in ranges:
            if not lower <= value <= upper:
                raise BackendContractError(f"cylinder {name}={value} lies outside configured [{lower}, {upper}]")
        if not np.isclose(
            parameters.torsional_friction,
            randomization.fixed_torsional_friction,
            rtol=0.0,
            atol=1e-15,
        ) or not np.isclose(
            parameters.rolling_friction,
            randomization.fixed_rolling_friction,
            rtol=0.0,
            atol=1e-15,
        ):
            raise BackendContractError("cylinder torsional and rolling friction must match fixed config values")
        geom_id = self._names.cylinder_geom_id
        body_id = self._names.cylinder_body_id
        half_side = parameters.radius_m / np.sqrt(2.0)
        model.geom_size[geom_id, :3] = (half_side, half_side, 0.5 * parameters.height_m)
        model.body_mass[body_id] = parameters.mass_kg
        model.body_inertia[body_id] = np.asarray(
            parameters.solid_inertia_kg_m2,
            dtype=np.float64,
        )

        # With equal MuJoCo geom priorities, pair sliding friction is the
        # component-wise maximum.  Setting the cylinder to the smaller sample
        # and the two counterpart groups to their samples realizes independent
        # cylinder-hand and cylinder-table coefficients without hidden pairs.
        model.geom_friction[:] = self._baseline_geom_friction
        cylinder_sliding = min(
            parameters.hand_sliding_friction,
            parameters.table_sliding_friction,
        )
        counterpart_ids = (
            *self._contact_groups.hand_geom_ids,
            *self._contact_groups.table_geom_ids,
        )
        priorities = {int(model.geom_priority[index]) for index in (geom_id, *counterpart_ids)}
        if len(priorities) != 1:
            raise BackendContractError("independent cylinder friction requires equal MuJoCo geom priorities")
        model.geom_friction[geom_id] = (
            cylinder_sliding,
            parameters.torsional_friction,
            parameters.rolling_friction,
        )
        for hand_geom_id in self._contact_groups.hand_geom_ids:
            model.geom_friction[hand_geom_id] = (
                parameters.hand_sliding_friction,
                parameters.torsional_friction,
                parameters.rolling_friction,
            )
        for table_geom_id in self._contact_groups.table_geom_ids:
            model.geom_friction[table_geom_id] = (
                parameters.table_sliding_friction,
                parameters.torsional_friction,
                parameters.rolling_friction,
            )
        mujoco.mj_setConst(model, data)
        self._cylinder_parameters = parameters

    def reset(self, reset_spec: ResetSpec) -> PhysicsState:
        if not isinstance(reset_spec, ResetSpec):
            raise TypeError(f"reset_spec must be ResetSpec, got {type(reset_spec).__name__}")
        model, data = self._live()
        validated_overrides: list[tuple[int, int | None, float]] = []
        for joint_name, joint_position in reset_spec.robot_joint_positions_rad.items():
            joint_id = self._names.object_ids["joints"].get(joint_name)
            if joint_id is None:
                raise BackendContractError(f"reset names unknown robot joint {joint_name!r}")
            if joint_name == _CYLINDER_JOINT_NAME:
                raise BackendContractError("cylinder free joint must be set through cylinder pose fields")
            if int(model.jnt_type[joint_id]) not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                raise BackendContractError(f"reset joint {joint_name!r} is not scalar")
            lower, upper = model.jnt_range[joint_id]
            if bool(model.jnt_limited[joint_id]) and not lower <= joint_position <= upper:
                raise BackendContractError(
                    f"reset joint {joint_name!r} target {joint_position} is outside [{lower}, {upper}]"
                )
            actuator_id = self._names.object_ids["actuators"].get(f"{joint_name}_position")
            validated_overrides.append((joint_id, actuator_id, joint_position))

        self._apply_cylinder_parameters(reset_spec.cylinder)
        mujoco.mj_resetDataKeyframe(model, data, self._names.home_key_id)
        for joint_id, actuator_id, joint_position in validated_overrides:
            data.qpos[int(model.jnt_qposadr[joint_id])] = joint_position
            if actuator_id is not None:
                data.ctrl[actuator_id] = joint_position

        cylinder_qpos_address = int(model.jnt_qposadr[self._names.cylinder_joint_id])
        cylinder_dof_address = int(model.jnt_dofadr[self._names.cylinder_joint_id])
        data.qpos[cylinder_qpos_address : cylinder_qpos_address + 3] = reset_spec.cylinder_position_m
        data.qpos[cylinder_qpos_address + 3 : cylinder_qpos_address + 7] = reset_spec.cylinder_quaternion_wxyz
        data.qvel[cylinder_dof_address : cylinder_dof_address + 3] = reset_spec.cylinder_linear_velocity_m_s
        data.qvel[cylinder_dof_address + 3 : cylinder_dof_address + 6] = reset_spec.cylinder_angular_velocity_rad_s
        data.qacc.fill(0.0)
        data.qacc_warmstart.fill(0.0)
        data.qfrc_applied.fill(0.0)
        data.xfrc_applied.fill(0.0)
        mujoco.mj_forward(model, data)
        data.qacc_warmstart.fill(0.0)

        self._episode_physics_steps = 0
        self._warning_counts = [0] * _EXPECTED_WARNING_COUNT
        self._warning_events.clear()
        self._contact_capacity_exceeded = False
        self._solver_anomaly = False
        self._nonfinite_state = False
        self._last_contacts = self._empty_contact_frame()
        self._synchronize_warnings()
        return self.read_state()

    def apply_targets(self, targets_rad: NDArray[np.float64]) -> None:
        if not isinstance(targets_rad, np.ndarray):
            raise TypeError("physical targets must be a CPU numpy.ndarray; device tensors are not accepted")
        if targets_rad.dtype != np.float64:
            raise TypeError(f"physical targets must have dtype float64, got {targets_rad.dtype}")
        if targets_rad.shape != (13,):
            raise ValueError(f"physical targets must have shape (13,), got {targets_rad.shape}")
        if not targets_rad.flags.c_contiguous:
            raise ValueError("physical targets must be C-contiguous")
        if not np.isfinite(targets_rad).all():
            raise ValueError("physical targets contain a non-finite value")
        model, data = self._live()
        lower = model.actuator_ctrlrange[np.asarray(self._names.actuator_ids), 0]
        upper = model.actuator_ctrlrange[np.asarray(self._names.actuator_ids), 1]
        below = np.flatnonzero(targets_rad < lower)
        above = np.flatnonzero(targets_rad > upper)
        if below.size or above.size:
            indices = tuple(int(index) for index in np.concatenate((below, above)))
            names = tuple(self._names.actuator_names[index] for index in indices)
            raise ValueError(f"physical targets exceed actuator ranges for {names}")
        data.ctrl[np.asarray(self._names.actuator_ids)] = targets_rad

    def read_actuator_control_ranges(self) -> ActuatorControlRanges:
        """Return detached control limits in the validated actuator order."""

        model, _ = self._live()
        actuator_ids = np.asarray(self._names.actuator_ids, dtype=np.int64)
        return ActuatorControlRanges(
            actuator_names=self._names.actuator_names,
            ranges_rad=np.asarray(model.actuator_ctrlrange[actuator_ids, :], dtype=np.float64),
        )

    def read_site_hinge_kinematics(self, site_name: str, joint_name: str) -> SiteHingeKinematics:
        """Read a finite named site Jacobian column without exposing MuJoCo arrays."""

        if not isinstance(site_name, str) or not site_name:
            raise TypeError("site_name must be a non-empty string")
        if not isinstance(joint_name, str) or not joint_name:
            raise TypeError("joint_name must be a non-empty string")
        model, data = self._live()
        site_id = self._names.object_ids["sites"].get(site_name)
        joint_id = self._names.object_ids["joints"].get(joint_name)
        if site_id is None:
            raise KeyError(f"unknown site {site_name!r}")
        if joint_id is None:
            raise KeyError(f"unknown joint {joint_name!r}")
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise BackendContractError(f"joint {joint_name!r} is not a scalar hinge")
        if not bool(model.jnt_limited[joint_id]):
            raise BackendContractError(f"hinge {joint_name!r} has no finite range")
        dof_address = int(model.jnt_dofadr[joint_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        jacobian = np.ascontiguousarray(
            np.concatenate((jacobian_position[:, dof_address], jacobian_rotation[:, dof_address])),
            dtype=np.float64,
        )
        return SiteHingeKinematics(
            site_name=site_name,
            joint_name=joint_name,
            joint_position_rad=float(data.qpos[qpos_address]),
            joint_lower_rad=float(model.jnt_range[joint_id, 0]),
            joint_upper_rad=float(model.jnt_range[joint_id, 1]),
            site_jacobian_6d=jacobian,
        )

    def read_arm_site_jacobian(self, site_name: str) -> tuple[FloatArray, FloatArray]:
        """Return detached arm positions and a world-frame ``6 x 7`` site Jacobian."""

        if not isinstance(site_name, str) or not site_name:
            raise TypeError("site_name must be a non-empty string")
        model, data = self._live()
        site_id = self._names.object_ids["sites"].get(site_name)
        if site_id is None:
            raise KeyError(f"unknown site {site_name!r}")
        joint_names = self._config.physics.actuator_calibration.arm.joint_names
        dof_addresses: list[int] = []
        qpos_addresses: list[int] = []
        for name in joint_names:
            joint_id = self._names.object_ids["joints"][name]
            if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise BackendContractError(f"arm joint {name!r} is not a scalar hinge")
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
        jacobian = np.ascontiguousarray(
            np.vstack((jacobian_position[:, dof_addresses], jacobian_rotation[:, dof_addresses])),
            dtype=np.float64,
        )
        positions = np.ascontiguousarray(data.qpos[qpos_addresses], dtype=np.float64)
        jacobian.setflags(write=False)
        positions.setflags(write=False)
        return positions, jacobian

    def step_substeps(self, count: int) -> PhysicsStepResult:
        if type(count) is not int:
            raise TypeError(f"substep count must be int, got {type(count).__name__}")
        expected = self._config.physics.decimation
        if count != expected:
            raise ValueError(f"one policy step requires exactly {expected} physical substeps")
        model, data = self._live()
        contacts: list[ContactSample] = []
        maximum_hand_forces = np.zeros(12, dtype=np.float64)
        maximum_force = 0.0
        maximum_impulse = 0.0
        maximum_illegal_impulse = 0.0
        maximum_contacts_seen = 0
        trace_positions = np.empty((4, 3), dtype=np.float64)
        trace_quaternions = np.empty((4, 4), dtype=np.float64)
        trace_linear_velocities = np.empty((4, 3), dtype=np.float64)
        trace_angular_velocities = np.empty((4, 3), dtype=np.float64)
        trace_finite = np.empty(4, dtype=np.bool_)
        trace_times = np.empty(4, dtype=np.float64)
        cylinder_qpos = int(model.jnt_qposadr[self._names.cylinder_joint_id])
        cylinder_dof = int(model.jnt_dofadr[self._names.cylinder_joint_id])
        for substep_index in range(count):
            try:
                mujoco.mj_step(model, data)
            except Exception as error:
                self._solver_anomaly = True
                self._synchronize_warnings()
                raise SimulationStepError(
                    f"MuJoCo step failed at physical step {self._total_physics_steps}: {error}",
                    self.read_warnings(),
                ) from error
            self._episode_physics_steps += 1
            self._total_physics_steps += 1
            self._synchronize_warnings()
            if not self._raw_state_is_finite():
                self._nonfinite_state = True
            trace_positions[substep_index] = data.qpos[cylinder_qpos : cylinder_qpos + 3]
            trace_quaternions[substep_index] = data.qpos[cylinder_qpos + 3 : cylinder_qpos + 7]
            trace_linear_velocities[substep_index] = data.qvel[cylinder_dof : cylinder_dof + 3]
            trace_angular_velocities[substep_index] = data.qvel[cylinder_dof + 3 : cylinder_dof + 6]
            trace_times[substep_index] = float(data.time)
            trace_finite[substep_index] = not (self._solver_anomaly or self._nonfinite_state) and bool(
                np.isfinite(
                    (
                        *trace_positions[substep_index],
                        *trace_quaternions[substep_index],
                        *trace_linear_velocities[substep_index],
                        *trace_angular_velocities[substep_index],
                        trace_times[substep_index],
                    )
                ).all()
            )
            samples, hand_forces, force, impulse, illegal_impulse = self._collect_contacts(substep_index)
            contacts.extend(samples)
            maximum_hand_forces = np.maximum(maximum_hand_forces, hand_forces)
            maximum_force = max(maximum_force, force)
            maximum_impulse = max(maximum_impulse, impulse)
            maximum_illegal_impulse = max(maximum_illegal_impulse, illegal_impulse)
            maximum_contacts_seen = max(maximum_contacts_seen, int(data.ncon))
            if int(data.ncon) >= self._config.physics.contact.maximum_contacts:
                self._contact_capacity_exceeded = True

        self._last_contacts = ContactFrame(
            contacts=tuple(contacts),
            substeps_collected=count,
            hand_link_names=self._contact_groups.observed_hand_body_names,
            hand_contact_force_norms_n=maximum_hand_forces,
            maximum_contact_force_n=maximum_force,
            maximum_contact_impulse_n_s=maximum_impulse,
            maximum_illegal_contact_impulse_n_s=maximum_illegal_impulse,
            contact_capacity=self._config.physics.contact.maximum_contacts,
            maximum_contacts_seen=maximum_contacts_seen,
            physics_step=self._total_physics_steps,
            simulation_time_s=float(data.time),
        )
        state = self.read_state()
        return PhysicsStepResult(
            state=state,
            contacts=self._last_contacts,
            diagnostics=self.read_warnings(),
            substeps_executed=count,
            substep_trace=PhysicsSubstepTrace(
                cylinder_positions_m=trace_positions,
                cylinder_quaternions_wxyz=trace_quaternions,
                cylinder_linear_velocities_m_s=trace_linear_velocities,
                cylinder_angular_velocities_rad_s=trace_angular_velocities,
                finite=trace_finite,
                simulation_times_s=trace_times,
            ),
        )

    def _collect_contacts(
        self,
        substep_index: int,
    ) -> tuple[list[ContactSample], FloatArray, float, float, float]:
        model, data = self._live()
        samples: list[ContactSample] = []
        hand_body_index = {body_id: index for index, body_id in enumerate(self._contact_groups.observed_hand_body_ids)}
        hand_net_forces = np.zeros((12, 3), dtype=np.float64)
        maximum_force = 0.0
        maximum_impulse = 0.0
        maximum_illegal_impulse = 0.0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1 = self._required_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2 = self._required_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            body1_id = int(model.geom_bodyid[geom1_id])
            body2_id = int(model.geom_bodyid[geom2_id])
            body1 = self._required_name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2 = self._required_name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            force_torque = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force_torque)
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            normal = np.array(frame[0], dtype=np.float64, copy=True)
            force_on_geom2 = np.array(frame.T @ force_torque[:3], dtype=np.float64, copy=True)
            force_norm = float(np.linalg.norm(force_on_geom2))
            normal_force = abs(float(force_torque[0]))
            impulse = force_norm * self._config.physics.physics_dt_s
            signed_distance = float(contact.dist)
            penetration = max(0.0, -signed_distance)
            groups1 = self._contact_groups.geom_groups.get(geom1, ())
            groups2 = self._contact_groups.geom_groups.get(geom2, ())
            is_illegal = is_illegal_contact(
                groups1,
                groups2,
                force_norm,
                right_arm_table_force_limit_n=self._config.task.termination.right_arm_table_contact_force_limit_n,
                illegal_groups=self._contact_groups.illegal_geom_groups,
            )
            if is_illegal:
                maximum_illegal_impulse = max(maximum_illegal_impulse, impulse)
            if body1_id in hand_body_index:
                hand_net_forces[hand_body_index[body1_id]] -= force_on_geom2
            if body2_id in hand_body_index:
                hand_net_forces[hand_body_index[body2_id]] += force_on_geom2
            maximum_force = max(maximum_force, force_norm)
            maximum_impulse = max(maximum_impulse, impulse)
            samples.append(
                ContactSample(
                    substep_index=substep_index,
                    physics_step=self._total_physics_steps,
                    geom1=geom1,
                    geom2=geom2,
                    body1=body1,
                    body2=body2,
                    geom1_groups=groups1,
                    geom2_groups=groups2,
                    position_m=np.asarray(contact.pos, dtype=np.float64),
                    normal_from_geom1_to_geom2=normal,
                    force_on_geom2_world_n=force_on_geom2,
                    normal_force_n=normal_force,
                    impulse_n_s=impulse,
                    signed_distance_m=signed_distance,
                    penetration_m=penetration,
                )
            )
        hand_norms = np.linalg.norm(hand_net_forces, axis=1).astype(np.float64, copy=False)
        return samples, hand_norms, maximum_force, maximum_impulse, maximum_illegal_impulse

    @staticmethod
    def _required_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
        name = mujoco.mj_id2name(model, object_type, object_id)
        if name is None:
            raise BackendContractError(f"runtime contact references unnamed object ID {object_id}")
        return name

    @staticmethod
    def _site_quaternion(data: mujoco.MjData, site_id: int) -> FloatArray:
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
        return mujoco_quaternion_to_wxyz(quaternion)

    @staticmethod
    def _object_velocity(
        model: mujoco.MjModel,
        data: mujoco.MjData,
        object_type: mujoco.mjtObj,
        object_id: int,
    ) -> tuple[FloatArray, FloatArray]:
        velocity = np.empty(6, dtype=np.float64)
        mujoco.mj_objectVelocity(model, data, object_type, object_id, velocity, 0)
        angular = np.array(velocity[:3], dtype=np.float64, copy=True)
        linear = np.array(velocity[3:], dtype=np.float64, copy=True)
        return linear, angular

    def _raw_state_is_finite(self) -> bool:
        _, data = self._live()
        arrays = (
            data.qpos,
            data.qvel,
            data.qacc,
            data.ctrl,
            data.actuator_force,
            data.xpos,
            data.xquat,
            data.site_xpos,
            data.site_xmat,
        )
        return all(np.isfinite(array).all() for array in arrays)

    def read_state(self) -> PhysicsState:
        model, data = self._live()
        if not self._raw_state_is_finite():
            self._nonfinite_state = True
            raise NonFinitePhysicsError(
                f"non-finite MuJoCo state at physical step {self._total_physics_steps}",
                self.read_warnings(),
            )
        joint_positions = np.asarray(
            [data.qpos[address] for address in self._names.observed_joint_qpos_addresses],
            dtype=np.float64,
        )
        joint_velocities = np.asarray(
            [data.qvel[address] for address in self._names.observed_joint_dof_addresses],
            dtype=np.float64,
        )
        body_positions = np.asarray(data.xpos[np.asarray(self._names.body_ids)], dtype=np.float64)
        body_quaternions = np.asarray(data.xquat[np.asarray(self._names.body_ids)], dtype=np.float64)
        site_positions = np.asarray(data.site_xpos[np.asarray(self._names.site_ids)], dtype=np.float64)
        site_quaternions = np.vstack([self._site_quaternion(data, site_id) for site_id in self._names.site_ids])
        site_linear_velocities = np.empty((len(self._names.site_ids), 3), dtype=np.float64)
        site_angular_velocities = np.empty((len(self._names.site_ids), 3), dtype=np.float64)
        for index, site_id in enumerate(self._names.site_ids):
            linear, angular = self._object_velocity(
                model,
                data,
                mujoco.mjtObj.mjOBJ_SITE,
                site_id,
            )
            site_linear_velocities[index] = linear
            site_angular_velocities[index] = angular
        cylinder_linear, cylinder_angular = self._object_velocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._names.cylinder_body_id,
        )
        tcp_index = self._names.site_ids.index(self._names.tcp_site_id)
        grasp_index = self._names.site_ids.index(self._names.grasp_center_site_id)
        cylinder_body_index = self._names.body_ids.index(self._names.cylinder_body_id)
        return PhysicsState(
            joint_names=self._names.observed_joint_names,
            joint_positions_rad=joint_positions,
            joint_velocities_rad_s=joint_velocities,
            body_names=self._names.body_names,
            body_positions_m=body_positions,
            body_quaternions_wxyz=body_quaternions,
            site_names=self._names.site_names,
            site_positions_m=site_positions,
            site_quaternions_wxyz=site_quaternions,
            site_linear_velocities_m_s=site_linear_velocities,
            site_angular_velocities_rad_s=site_angular_velocities,
            tcp_position_m=site_positions[tcp_index],
            tcp_quaternion_wxyz=site_quaternions[tcp_index],
            grasp_center_position_m=site_positions[grasp_index],
            grasp_center_quaternion_wxyz=site_quaternions[grasp_index],
            cylinder_position_m=body_positions[cylinder_body_index],
            cylinder_quaternion_wxyz=body_quaternions[cylinder_body_index],
            cylinder_linear_velocity_m_s=cylinder_linear,
            cylinder_angular_velocity_rad_s=cylinder_angular,
            actuator_names=self._names.actuator_names,
            actuator_targets_rad=np.asarray(data.ctrl[np.asarray(self._names.actuator_ids)], dtype=np.float64),
            actuator_forces_n_m=np.asarray(data.actuator_force[np.asarray(self._names.actuator_ids)], dtype=np.float64),
            cylinder_parameters=self._cylinder_parameters,
            episode_physics_steps=self._episode_physics_steps,
            total_physics_steps=self._total_physics_steps,
            simulation_time_s=float(data.time),
            total_simulation_time_s=(self._total_physics_steps * self._config.physics.physics_dt_s),
        )

    def read_contacts(self) -> ContactFrame:
        self._live()
        return self._last_contacts

    def _synchronize_warnings(self) -> None:
        _, data = self._live()
        if len(data.warning) != _EXPECTED_WARNING_COUNT:
            raise BackendContractError(
                f"MuJoCo warning count changed: expected {_EXPECTED_WARNING_COUNT}, got {len(data.warning)}"
            )
        for code in range(_EXPECTED_WARNING_COUNT):
            warning = data.warning[code]
            count = int(warning.number)
            if count < self._warning_counts[code]:
                # MuJoCo may clear warning counters during an internal reset.
                self._warning_counts[code] = 0
            increment = count - self._warning_counts[code]
            if increment <= 0:
                continue
            last_info = int(warning.lastinfo)
            warning_name = mujoco.mjtWarning(code).name
            self._warning_events.append(
                SimulationWarning(
                    code=code,
                    name=warning_name,
                    message=mujoco.mju_warningText(code, last_info),
                    occurrence_count=increment,
                    last_info=last_info,
                    physics_step=self._total_physics_steps,
                    config_fingerprint=self._config_fingerprint,
                )
            )
            self._warning_counts[code] = count
            if code == int(mujoco.mjtWarning.mjWARN_CONTACTFULL):
                self._contact_capacity_exceeded = True
            if code in _SOLVER_WARNING_CODES:
                self._solver_anomaly = True

    def read_warnings(self) -> SimulationDiagnostics:
        _, data = self._live()
        self._synchronize_warnings()
        if not self._raw_state_is_finite():
            self._nonfinite_state = True
        return SimulationDiagnostics(
            warnings=tuple(self._warning_events),
            contact_capacity_exceeded=self._contact_capacity_exceeded,
            solver_anomaly=self._solver_anomaly,
            nonfinite_state=self._nonfinite_state,
            physics_step=self._total_physics_steps,
            simulation_time_s=float(data.time),
            config_fingerprint=self._config_fingerprint,
        )

    def close(self) -> None:
        """Release Python's only references to the independent model/data pair."""

        if self._model is None:
            return
        self._data = None
        self._model = None

    def __enter__(self) -> MujocoEnvironmentBackend:
        self._live()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
