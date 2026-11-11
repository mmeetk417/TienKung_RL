"""Typed, immutable snapshots exchanged across the simulation boundary.

MuJoCo owns mutable ``float64`` buffers.  Objects in this module deliberately
contain copies with NumPy's write flag disabled so task code can never mutate
live simulator state by retaining a view.  Positions use metres, angles use
radians, forces use newtons, and impulses use newton-seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_ILLEGAL_CONTACT_GROUPS = frozenset(("right_arm", "tcp", "non_task_body"))


def is_illegal_contact(
    geom1_groups: Sequence[str],
    geom2_groups: Sequence[str],
    force_n: float,
    *,
    right_arm_table_force_limit_n: float,
    illegal_groups: Sequence[str] = _ILLEGAL_CONTACT_GROUPS,
) -> bool:
    """Classify one contact, allowing low-force right-arm/table contact."""

    if not np.isfinite(force_n) or force_n < 0.0:
        raise ValueError("contact force must be finite and non-negative")
    if not np.isfinite(right_arm_table_force_limit_n) or right_arm_table_force_limit_n < 0.0:
        raise ValueError("right-arm/table force limit must be finite and non-negative")
    first = set(geom1_groups)
    second = set(geom2_groups)
    illegal_overlap = set(illegal_groups).intersection(first | second)
    if not illegal_overlap:
        return False
    right_arm_table = ("right_arm" in first and "table" in second) or ("right_arm" in second and "table" in first)
    return not (right_arm_table and illegal_overlap == {"right_arm"} and force_n <= right_arm_table_force_limit_n)


def readonly_float64(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    """Return an owning, C-contiguous, finite, read-only ``float64`` copy."""

    array = np.asarray(value)
    if array.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {array.dtype}")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class ActuatorControlRanges:
    """Named, immutable actuator control bounds from one live model."""

    actuator_names: tuple[str, ...]
    ranges_rad: FloatArray

    def __post_init__(self) -> None:
        count = len(self.actuator_names)
        if len(set(self.actuator_names)) != count:
            raise ValueError("actuator control-range names must be unique")
        object.__setattr__(self, "ranges_rad", readonly_float64(self.ranges_rad, (count, 2), "ranges_rad"))
        if np.any(self.ranges_rad[:, 0] >= self.ranges_rad[:, 1]):
            raise ValueError("actuator control lower bounds must be strictly below upper bounds")

    @property
    def lower_rad(self) -> FloatArray:
        result = self.ranges_rad[:, 0]
        result.flags.writeable = False
        return result

    @property
    def upper_rad(self) -> FloatArray:
        result = self.ranges_rad[:, 1]
        result.flags.writeable = False
        return result


@dataclass(frozen=True, slots=True, eq=False)
class SiteHingeKinematics:
    """Finite read-only kinematics for one named site and scalar hinge."""

    site_name: str
    joint_name: str
    joint_position_rad: float
    joint_lower_rad: float
    joint_upper_rad: float
    site_jacobian_6d: FloatArray

    def __post_init__(self) -> None:
        if not self.site_name or not self.joint_name:
            raise ValueError("site and joint names must not be empty")
        numeric = (self.joint_position_rad, self.joint_lower_rad, self.joint_upper_rad)
        if not np.isfinite(numeric).all():
            raise ValueError("site-hinge kinematics scalars must be finite")
        if self.joint_lower_rad > self.joint_position_rad or self.joint_position_rad > self.joint_upper_rad:
            raise ValueError("hinge position must lie within its joint range")
        object.__setattr__(
            self,
            "site_jacobian_6d",
            readonly_float64(self.site_jacobian_6d, (6,), "site_jacobian_6d"),
        )


def mujoco_quaternion_to_wxyz(value: object) -> FloatArray:
    """Normalize a MuJoCo quaternion and choose one deterministic ``wxyz`` sign.

    MuJoCo already stores quaternions in ``wxyz`` order.  Normalization makes
    the boundary robust to accumulated floating-point error.  Since ``q`` and
    ``-q`` describe the same rotation, the first non-zero component is made
    non-negative (normally this is the scalar component).
    """

    quaternion = np.asarray(value)
    if quaternion.dtype != np.float64:
        raise TypeError(f"MuJoCo quaternion must have dtype float64, got {quaternion.dtype}")
    if quaternion.shape != (4,):
        raise ValueError(f"MuJoCo quaternion must have shape (4,), got {quaternion.shape}")
    if not np.isfinite(quaternion).all():
        raise ValueError("MuJoCo quaternion contains a non-finite value")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("MuJoCo quaternion has zero norm")
    normalized = np.array(quaternion / norm, dtype=np.float64, copy=True)
    for component in normalized:
        if abs(float(component)) > np.finfo(np.float64).eps:
            if component < 0.0:
                normalized *= -1.0
            break
    normalized.setflags(write=False)
    return normalized


def _quaternion_rows(value: object, rows: int, name: str) -> FloatArray:
    source = np.asarray(value)
    if source.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {source.dtype}")
    if source.shape != (rows, 4):
        raise ValueError(f"{name} must have shape {(rows, 4)}, got {source.shape}")
    result = np.empty_like(source)
    for index in range(rows):
        result[index] = mujoco_quaternion_to_wxyz(source[index])
    result.setflags(write=False)
    return result


def _freeze_mapping(value: Mapping[str, float], name: str) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        scalar = float(item)
        if not np.isfinite(scalar):
            raise ValueError(f"{name}[{key!r}] must be finite")
        result[key] = scalar
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class CylinderParameters:
    """One episode's physical square-prism parameters, all in SI units."""

    radius_m: float = 0.015
    height_m: float = 0.100
    mass_kg: float = 0.115
    hand_sliding_friction: float = 0.9
    table_sliding_friction: float = 0.9
    torsional_friction: float = 0.005
    rolling_friction: float = 0.0001

    def __post_init__(self) -> None:
        for name in ("radius_m", "height_m", "mass_kg"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"cylinder {name} must be finite and positive")
        for name in (
            "hand_sliding_friction",
            "table_sliding_friction",
            "torsional_friction",
            "rolling_friction",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"cylinder {name} must be finite and non-negative")

    @property
    def solid_inertia_kg_m2(self) -> tuple[float, float, float]:
        """Principal inertia of a square prism with circumradius ``radius_m``."""

        half_side_sq = 0.5 * self.radius_m**2
        transverse = self.mass_kg * (4.0 * half_side_sq + self.height_m**2) / 12.0
        axial = (2.0 / 3.0) * self.mass_kg * half_side_sq
        return (transverse, transverse, axial)


@dataclass(frozen=True, slots=True)
class ResetSpec:
    """Complete episode reset request.

    ``robot_joint_positions_rad`` is an optional sparse override on top of the
    versioned ``robot_and_scene_home`` keyframe.  The cylinder pose and twist
    refer to its free joint and are expressed in world coordinates.
    """

    robot_joint_positions_rad: Mapping[str, float] = field(default_factory=dict)
    cylinder_position_m: tuple[float, float, float] = (0.5252919, -0.26195379, 0.8728)
    cylinder_quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    cylinder_linear_velocity_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cylinder_angular_velocity_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cylinder: CylinderParameters = field(default_factory=CylinderParameters)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "robot_joint_positions_rad",
            _freeze_mapping(self.robot_joint_positions_rad, "robot_joint_positions_rad"),
        )
        position = readonly_float64(
            np.asarray(self.cylinder_position_m, dtype=np.float64),
            (3,),
            "cylinder_position_m",
        )
        quaternion = mujoco_quaternion_to_wxyz(np.asarray(self.cylinder_quaternion_wxyz, dtype=np.float64))
        linear_velocity = readonly_float64(
            np.asarray(self.cylinder_linear_velocity_m_s, dtype=np.float64),
            (3,),
            "cylinder_linear_velocity_m_s",
        )
        angular_velocity = readonly_float64(
            np.asarray(self.cylinder_angular_velocity_rad_s, dtype=np.float64),
            (3,),
            "cylinder_angular_velocity_rad_s",
        )
        object.__setattr__(self, "cylinder_position_m", tuple(float(item) for item in position))
        object.__setattr__(
            self,
            "cylinder_quaternion_wxyz",
            tuple(float(item) for item in quaternion),
        )
        object.__setattr__(
            self,
            "cylinder_linear_velocity_m_s",
            tuple(float(item) for item in linear_velocity),
        )
        object.__setattr__(
            self,
            "cylinder_angular_velocity_rad_s",
            tuple(float(item) for item in angular_velocity),
        )


@dataclass(frozen=True, slots=True, eq=False)
class PhysicsState:
    """Read-only named snapshot of one MuJoCo environment.

    Array shapes are ``(J,)`` for named scalar joints, ``(B, 3/4)`` for body
    poses, ``(S, 3/4)`` for site poses/twists, and ``(A,)`` for actuators.
    All arrays are C-contiguous ``float64`` values and do not alias MuJoCo.
    """

    joint_names: tuple[str, ...]
    joint_positions_rad: FloatArray
    joint_velocities_rad_s: FloatArray
    body_names: tuple[str, ...]
    body_positions_m: FloatArray
    body_quaternions_wxyz: FloatArray
    site_names: tuple[str, ...]
    site_positions_m: FloatArray
    site_quaternions_wxyz: FloatArray
    site_linear_velocities_m_s: FloatArray
    site_angular_velocities_rad_s: FloatArray
    tcp_position_m: FloatArray
    tcp_quaternion_wxyz: FloatArray
    grasp_center_position_m: FloatArray
    grasp_center_quaternion_wxyz: FloatArray
    cylinder_position_m: FloatArray
    cylinder_quaternion_wxyz: FloatArray
    cylinder_linear_velocity_m_s: FloatArray
    cylinder_angular_velocity_rad_s: FloatArray
    actuator_names: tuple[str, ...]
    actuator_targets_rad: FloatArray
    actuator_forces_n_m: FloatArray
    cylinder_parameters: CylinderParameters
    episode_physics_steps: int
    total_physics_steps: int
    simulation_time_s: float
    total_simulation_time_s: float

    def __post_init__(self) -> None:
        joint_count = len(self.joint_names)
        body_count = len(self.body_names)
        site_count = len(self.site_names)
        actuator_count = len(self.actuator_names)
        if len(set(self.joint_names)) != joint_count:
            raise ValueError("PhysicsState joint_names must be unique")
        if len(set(self.body_names)) != body_count:
            raise ValueError("PhysicsState body_names must be unique")
        if len(set(self.site_names)) != site_count:
            raise ValueError("PhysicsState site_names must be unique")
        if len(set(self.actuator_names)) != actuator_count:
            raise ValueError("PhysicsState actuator_names must be unique")
        for field_name, shape in (
            ("joint_positions_rad", (joint_count,)),
            ("joint_velocities_rad_s", (joint_count,)),
            ("body_positions_m", (body_count, 3)),
            ("site_positions_m", (site_count, 3)),
            ("site_linear_velocities_m_s", (site_count, 3)),
            ("site_angular_velocities_rad_s", (site_count, 3)),
            ("tcp_position_m", (3,)),
            ("grasp_center_position_m", (3,)),
            ("cylinder_position_m", (3,)),
            ("cylinder_linear_velocity_m_s", (3,)),
            ("cylinder_angular_velocity_rad_s", (3,)),
            ("actuator_targets_rad", (actuator_count,)),
            ("actuator_forces_n_m", (actuator_count,)),
        ):
            object.__setattr__(
                self,
                field_name,
                readonly_float64(getattr(self, field_name), shape, field_name),
            )
        for field_name, rows in (
            ("body_quaternions_wxyz", body_count),
            ("site_quaternions_wxyz", site_count),
        ):
            object.__setattr__(
                self,
                field_name,
                _quaternion_rows(getattr(self, field_name), rows, field_name),
            )
        for field_name in (
            "tcp_quaternion_wxyz",
            "grasp_center_quaternion_wxyz",
            "cylinder_quaternion_wxyz",
        ):
            object.__setattr__(
                self,
                field_name,
                mujoco_quaternion_to_wxyz(getattr(self, field_name)),
            )
        if self.episode_physics_steps < 0 or self.total_physics_steps < 0:
            raise ValueError("physics step counters must be non-negative")
        if self.episode_physics_steps > self.total_physics_steps:
            raise ValueError("episode physics steps cannot exceed total physics steps")
        if not np.isfinite((self.simulation_time_s, self.total_simulation_time_s)).all():
            raise ValueError("simulation times must be finite")
        if self.simulation_time_s < 0.0 or self.total_simulation_time_s < 0.0:
            raise ValueError("simulation times must be non-negative")

    def joint_index(self, name: str) -> int:
        """Return a named joint's snapshot index, failing closed if unknown."""

        try:
            return self.joint_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown state joint {name!r}") from error

    def body_index(self, name: str) -> int:
        try:
            return self.body_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown state body {name!r}") from error

    def site_index(self, name: str) -> int:
        try:
            return self.site_names.index(name)
        except ValueError as error:
            raise KeyError(f"unknown state site {name!r}") from error


@dataclass(frozen=True, slots=True, eq=False)
class ContactSample:
    """One contact measured after one physical substep."""

    substep_index: int
    physics_step: int
    geom1: str
    geom2: str
    body1: str
    body2: str
    geom1_groups: tuple[str, ...]
    geom2_groups: tuple[str, ...]
    position_m: FloatArray
    normal_from_geom1_to_geom2: FloatArray
    force_on_geom2_world_n: FloatArray
    normal_force_n: float
    impulse_n_s: float
    signed_distance_m: float
    penetration_m: float

    def __post_init__(self) -> None:
        for field_name in (
            "position_m",
            "normal_from_geom1_to_geom2",
            "force_on_geom2_world_n",
        ):
            object.__setattr__(
                self,
                field_name,
                readonly_float64(getattr(self, field_name), (3,), field_name),
            )
        if not np.isclose(
            np.linalg.norm(self.normal_from_geom1_to_geom2),
            1.0,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("contact normal must have unit length")
        if self.substep_index < 0 or self.physics_step < 0:
            raise ValueError("contact step indices must be non-negative")
        for name in ("normal_force_n", "impulse_n_s", "penetration_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"contact {name} must be finite and non-negative")
        if not np.isfinite(self.signed_distance_m):
            raise ValueError("contact signed_distance_m must be finite")


@dataclass(frozen=True, slots=True, eq=False)
class ContactFrame:
    """Contacts aggregated across the most recent physical substeps.

    Each sample retains its substep index.  The 12 hand force norms are the
    maximum norm of the net force on each configured link over those substeps;
    this preserves brief contacts without summing the same sustained force four
    times.  Illegal impulse is the maximum individual contact impulse.
    """

    contacts: tuple[ContactSample, ...]
    substeps_collected: int
    hand_link_names: tuple[str, ...]
    hand_contact_force_norms_n: FloatArray
    maximum_contact_force_n: float
    maximum_contact_impulse_n_s: float
    maximum_illegal_contact_impulse_n_s: float
    contact_capacity: int
    maximum_contacts_seen: int
    physics_step: int
    simulation_time_s: float

    def __post_init__(self) -> None:
        if self.substeps_collected < 0:
            raise ValueError("substeps_collected must be non-negative")
        if self.physics_step < 0:
            raise ValueError("contact frame physics_step must be non-negative")
        if self.contact_capacity < 1 or self.maximum_contacts_seen < 0:
            raise ValueError("contact capacities must be valid")
        object.__setattr__(
            self,
            "hand_contact_force_norms_n",
            readonly_float64(
                self.hand_contact_force_norms_n,
                (len(self.hand_link_names),),
                "hand_contact_force_norms_n",
            ),
        )
        for name in (
            "maximum_contact_force_n",
            "maximum_contact_impulse_n_s",
            "maximum_illegal_contact_impulse_n_s",
            "simulation_time_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"ContactFrame {name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class SimulationWarning:
    """One newly observed MuJoCo warning occurrence."""

    code: int
    name: str
    message: str
    occurrence_count: int
    last_info: int
    physics_step: int
    config_fingerprint: str

    def __post_init__(self) -> None:
        if self.code < 0 or self.occurrence_count < 1 or self.physics_step < 0:
            raise ValueError("warning code/count/physics step must be valid")
        if not self.name or not self.message or not self.config_fingerprint:
            raise ValueError("warning name, message and config fingerprint must not be empty")


@dataclass(frozen=True, slots=True)
class SimulationDiagnostics:
    """Cumulative episode diagnostics; reading does not consume events."""

    warnings: tuple[SimulationWarning, ...]
    contact_capacity_exceeded: bool
    solver_anomaly: bool
    nonfinite_state: bool
    physics_step: int
    simulation_time_s: float
    config_fingerprint: str

    def __post_init__(self) -> None:
        if self.physics_step < 0:
            raise ValueError("diagnostics physics_step must be non-negative")
        if not np.isfinite(self.simulation_time_s) or self.simulation_time_s < 0.0:
            raise ValueError("diagnostics simulation_time_s must be finite and non-negative")
        if not self.config_fingerprint:
            raise ValueError("diagnostics config_fingerprint must not be empty")

    @property
    def healthy(self) -> bool:
        return not (self.contact_capacity_exceeded or self.solver_anomaly or self.nonfinite_state)


@dataclass(frozen=True, slots=True, eq=False)
class PhysicsSubstepTrace:
    """Exactly four immutable cylinder snapshots from one policy step."""

    cylinder_positions_m: FloatArray
    cylinder_quaternions_wxyz: FloatArray
    cylinder_linear_velocities_m_s: FloatArray
    cylinder_angular_velocities_rad_s: FloatArray
    finite: NDArray[np.bool_]
    simulation_times_s: FloatArray

    def __post_init__(self) -> None:
        for name, shape in (
            ("cylinder_positions_m", (4, 3)),
            ("cylinder_linear_velocities_m_s", (4, 3)),
            ("cylinder_angular_velocities_rad_s", (4, 3)),
            ("simulation_times_s", (4,)),
        ):
            object.__setattr__(self, name, readonly_float64(getattr(self, name), shape, name))
        object.__setattr__(
            self,
            "cylinder_quaternions_wxyz",
            _quaternion_rows(self.cylinder_quaternions_wxyz, 4, "cylinder_quaternions_wxyz"),
        )
        finite = np.array(self.finite, dtype=np.bool_, order="C", copy=True)
        if finite.shape != (4,):
            raise ValueError("substep finite flags must have shape (4,)")
        finite.setflags(write=False)
        object.__setattr__(self, "finite", finite)
        time_deltas = np.diff(self.simulation_times_s)
        if np.any(self.simulation_times_s < 0.0) or np.any(time_deltas < 0.0):
            raise ValueError("substep simulation times must be non-negative and non-decreasing")
        if bool(np.all(self.finite)) and np.any(time_deltas <= 0.0):
            raise ValueError("healthy substep simulation times must be strictly increasing")


@dataclass(frozen=True, slots=True)
class PhysicsStepResult:
    """Atomic result of a requested group of physical substeps."""

    state: PhysicsState
    contacts: ContactFrame
    diagnostics: SimulationDiagnostics
    substeps_executed: int
    substep_trace: PhysicsSubstepTrace

    def __post_init__(self) -> None:
        if self.substeps_executed < 1:
            raise ValueError("PhysicsStepResult requires at least one substep")
        if self.substeps_executed != 4:
            raise ValueError("PhysicsStepResult trace requires exactly four substeps")
