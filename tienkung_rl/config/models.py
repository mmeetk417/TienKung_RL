"""Immutable, typed configuration contracts for ``reach_then_close``.

All physical quantities use SI units.  Unit-bearing fields carry the unit in
their name and in dataclass metadata so snapshots and schema tooling do not
need an out-of-band unit table.  The classes deliberately contain no generic
``dict`` fields: every value that changes task meaning has a named type.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, cast


def contract_field(*, unit: str = "1", minimum: float | None = None, maximum: float | None = None) -> Any:
    """Declare a required field with machine-readable unit/range metadata."""

    return field(metadata={"unit": unit, "minimum": minimum, "maximum": maximum})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: float, name: str) -> None:
    _require(math.isfinite(value), f"{name} must be finite")


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    _finite(value, name)
    if allow_zero:
        _require(value >= 0.0, f"{name} must be >= 0")
    else:
        _require(value > 0.0, f"{name} must be > 0")


def _probability(value: float, name: str) -> None:
    _finite(value, name)
    _require(0.0 <= value <= 1.0, f"{name} must lie in [0, 1]")


def _nonempty(value: str, name: str) -> None:
    _require(bool(value.strip()), f"{name} must not be empty")


def _validate_all_nested(instance: object) -> None:
    """Run nested validators and reject non-finite floats everywhere."""

    def visit(value: Any, path: str) -> None:
        if isinstance(value, float):
            _finite(value, path)
        elif is_dataclass(value) and not isinstance(value, type):
            validator = getattr(value, "validate", None)
            if callable(validator):
                validator()
            for item in fields(value):
                visit(getattr(value, item.name), f"{path}.{item.name}")
        elif isinstance(value, tuple):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")

    for config_field in fields(cast(Any, instance)):
        visit(getattr(instance, config_field.name), config_field.name)


@dataclass(frozen=True, kw_only=True)
class PlatformConfig:
    """Supported CPU platform; counts are integers in the shown ranges."""

    operating_system: str
    architecture: str
    python_version: str
    mujoco_version: str
    pytorch_version: str
    backend: str
    device: str
    num_envs: int = contract_field(unit="environment", minimum=1)
    worker_threads: int = contract_field(unit="thread", minimum=1)

    def validate(self) -> None:
        for name in (
            "operating_system",
            "architecture",
            "python_version",
            "mujoco_version",
            "pytorch_version",
            "backend",
            "device",
        ):
            _nonempty(getattr(self, name), f"platform.{name}")
        _require(self.operating_system == "ubuntu-22.04", "only ubuntu-22.04 is supported")
        _require(self.architecture == "x86_64", "only x86_64 is supported")
        _require(self.python_version == "3.11", "python_version must be pinned to 3.11")
        _require(self.mujoco_version == "3.11.0", "mujoco_version must be pinned to 3.11.0")
        _require(
            self.pytorch_version == "2.13.0+cpu",
            "pytorch_version must be pinned to 2.13.0+cpu",
        )
        _require(self.backend == "mujoco", "backend must be mujoco")
        _require(self.device == "cpu", "device must be cpu")
        _require(self.num_envs >= 1, "platform.num_envs must be >= 1")
        _require(self.worker_threads >= 1, "platform.worker_threads must be >= 1")
        _require(
            self.worker_threads <= self.num_envs,
            "worker_threads cannot exceed num_envs",
        )


@dataclass(frozen=True, kw_only=True)
class AssetConfig:
    """Asset paths and model version; legacy digests are optional report metadata."""

    source_urdf_path: str
    runtime_mjcf_path: str
    viewer_mjcf_path: str
    visual_mesh_directory: str
    collision_manifest_path: str
    generation_manifest_path: str
    source_urdf_sha256: str = ""
    runtime_mjcf_sha256: str = ""
    viewer_mjcf_sha256: str = ""
    visual_meshes_sha256: str = ""
    collision_assets_sha256: str = ""
    generation_manifest_sha256: str = ""
    asset_version: str
    calibration_status: str

    def validate(self) -> None:
        for name in (
            "source_urdf_path",
            "runtime_mjcf_path",
            "viewer_mjcf_path",
            "visual_mesh_directory",
            "collision_manifest_path",
            "generation_manifest_path",
            "asset_version",
        ):
            _nonempty(getattr(self, name), f"asset.{name}")
        _require(
            self.calibration_status in {"starting_point_unvalidated", "validated"},
            "asset.calibration_status must be starting_point_unvalidated or validated",
        )


@dataclass(frozen=True, kw_only=True)
class SolverConfig:
    integrator: str
    solver: str
    position_iterations: int = contract_field(unit="iteration", minimum=1)
    no_slip_iterations: int = contract_field(unit="iteration", minimum=0)
    line_search_iterations: int = contract_field(unit="iteration", minimum=0)
    tolerance: float = contract_field(unit="1", minimum=0.0)

    def validate(self) -> None:
        _require(
            self.integrator in {"implicitfast", "implicit", "rk4", "euler"},
            "unsupported MuJoCo integrator",
        )
        _require(self.solver in {"newton", "cg", "pgs"}, "unsupported MuJoCo solver")
        _require(self.position_iterations >= 1, "solver.position_iterations must be >= 1")
        _require(self.no_slip_iterations >= 0, "solver.no_slip_iterations must be >= 0")
        _require(
            self.line_search_iterations >= 0,
            "solver.line_search_iterations must be >= 0",
        )
        _positive(self.tolerance, "solver.tolerance", allow_zero=True)


@dataclass(frozen=True, kw_only=True)
class MaterialConfig:
    """MuJoCo contact material coefficients, all dimensionless."""

    name: str
    sliding_friction: float = contract_field(minimum=0.0)
    torsional_friction: float = contract_field(minimum=0.0)
    rolling_friction: float = contract_field(minimum=0.0)
    restitution: float = contract_field(minimum=0.0, maximum=1.0)

    def validate(self) -> None:
        _nonempty(self.name, "material.name")
        for name in ("sliding_friction", "torsional_friction", "rolling_friction"):
            _positive(getattr(self, name), f"material.{name}", allow_zero=True)
        _probability(self.restitution, "material.restitution")


@dataclass(frozen=True, kw_only=True)
class ContactConfig:
    margin_m: float = contract_field(unit="m", minimum=0.0)
    gap_m: float = contract_field(unit="m", minimum=0.0)
    impedance: tuple[float, float, float]
    reference_time_s: float = contract_field(unit="s", minimum=0.0)
    reference_damping: float = contract_field(unit="1", minimum=0.0)
    maximum_contacts: int = contract_field(unit="contact", minimum=1)
    illegal_impulse_threshold_n_s: float = contract_field(unit="N*s", minimum=0.0)

    def validate(self) -> None:
        _positive(self.margin_m, "contact.margin_m", allow_zero=True)
        _positive(self.gap_m, "contact.gap_m", allow_zero=True)
        _require(len(self.impedance) == 3, "contact.impedance must have three values")
        for index, value in enumerate(self.impedance):
            _probability(value, f"contact.impedance[{index}]")
        _positive(self.reference_time_s, "contact.reference_time_s")
        _positive(self.reference_damping, "contact.reference_damping")
        _require(self.maximum_contacts >= 1, "contact.maximum_contacts must be >= 1")
        _positive(self.illegal_impulse_threshold_n_s, "contact.illegal_impulse_threshold_n_s")


@dataclass(frozen=True, kw_only=True)
class ActuatorGroupConfig:
    joint_names: tuple[str, ...]
    stiffness_n_m_per_rad: tuple[float, ...]
    damping_n_m_s_per_rad: tuple[float, ...]
    effort_limit_n_m: tuple[float, ...]
    velocity_limit_rad_s: tuple[float, ...]

    def validate(self) -> None:
        count = len(self.joint_names)
        _require(count > 0, "actuator group must contain joints")
        _require(len(set(self.joint_names)) == count, "actuator joint names must be unique")
        for name in self.joint_names:
            _nonempty(name, "actuator.joint_name")
        for name in (
            "stiffness_n_m_per_rad",
            "damping_n_m_s_per_rad",
            "effort_limit_n_m",
            "velocity_limit_rad_s",
        ):
            values = getattr(self, name)
            _require(len(values) == count, f"actuator.{name} must match joint_names")
            for index, value in enumerate(values):
                _positive(value, f"actuator.{name}[{index}]")


@dataclass(frozen=True, kw_only=True)
class ActuatorCalibrationConfig:
    version: str
    status: str
    control_mode: str
    arm: ActuatorGroupConfig
    hand: ActuatorGroupConfig

    def validate(self) -> None:
        _nonempty(self.version, "actuator_calibration.version")
        _require(
            self.status in {"starting_point_unvalidated", "validated"},
            "invalid actuator calibration status",
        )
        _require(self.control_mode == "position", "right arm control_mode must be position")
        self.arm.validate()
        self.hand.validate()
        _require(len(self.arm.joint_names) == 7, "arm actuator group must have seven joints")
        _require(
            len(self.hand.joint_names) == 6,
            "hand actuator group must have six active joints",
        )


@dataclass(frozen=True, kw_only=True)
class PhysicsConfig:
    """MuJoCo CPU timing, solver, material, contact and actuator contract."""

    version: str
    physics_dt_s: float = contract_field(unit="s", minimum=0.0)
    decimation: int = contract_field(unit="physics_step/policy_step", minimum=1)
    policy_dt_s: float = contract_field(unit="s", minimum=0.0)
    gravity_m_s2: tuple[float, float, float]
    solver: SolverConfig
    table_material: MaterialConfig
    robot_material: MaterialConfig
    cylinder_material: MaterialConfig
    contact: ContactConfig
    actuator_calibration: ActuatorCalibrationConfig
    calibration_status: str
    cylinder_contact: ContactConfig | None = None

    def validate(self) -> None:
        _nonempty(self.version, "physics.version")
        _positive(self.physics_dt_s, "physics.physics_dt_s")
        _require(self.decimation >= 1, "physics.decimation must be >= 1")
        _positive(self.policy_dt_s, "physics.policy_dt_s")
        _require(
            math.isclose(
                self.policy_dt_s,
                self.physics_dt_s * self.decimation,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "physics.policy_dt_s must equal physics_dt_s * decimation",
        )
        _require(len(self.gravity_m_s2) == 3, "gravity_m_s2 must have three components")
        for index, value in enumerate(self.gravity_m_s2):
            _finite(value, f"physics.gravity_m_s2[{index}]")
        _require(self.gravity_m_s2[2] < 0.0, "gravity z component must be negative")
        self.solver.validate()
        self.table_material.validate()
        self.robot_material.validate()
        self.cylinder_material.validate()
        self.contact.validate()
        if self.cylinder_contact is not None:
            self.cylinder_contact.validate()
        self.actuator_calibration.validate()
        _require(
            self.calibration_status in {"starting_point_unvalidated", "validated"},
            "invalid physics calibration status",
        )


@dataclass(frozen=True, kw_only=True)
class SceneConfig:
    table_center_m: tuple[float, float, float]
    table_size_m: tuple[float, float, float]
    cylinder_origin_xy_m: tuple[float, float]
    cylinder_table_clearance_m: float = contract_field(unit="m", minimum=0.0)
    maximum_episode_steps: int = contract_field(unit="policy_step", minimum=1)

    def validate(self) -> None:
        for name, values, length in (
            ("table_center_m", self.table_center_m, 3),
            ("table_size_m", self.table_size_m, 3),
            ("cylinder_origin_xy_m", self.cylinder_origin_xy_m, 2),
        ):
            _require(len(values) == length, f"scene.{name} must have {length} values")
            for index, value in enumerate(values):
                _finite(value, f"scene.{name}[{index}]")
        for index, value in enumerate(self.table_size_m):
            _positive(value, f"scene.table_size_m[{index}]")
        _positive(
            self.cylinder_table_clearance_m,
            "scene.cylinder_table_clearance_m",
            allow_zero=True,
        )
        _require(self.maximum_episode_steps >= 1, "scene.maximum_episode_steps must be >= 1")


@dataclass(frozen=True, kw_only=True)
class ObservationFieldConfig:
    name: str
    dimension: int = contract_field(unit="scalar", minimum=1)
    scale: float = contract_field(unit="1")
    frame: str
    quaternion_order: str
    canonicalize_scalar_nonnegative: bool

    def validate(self) -> None:
        _nonempty(self.name, "observation field name")
        _require(self.dimension >= 1, f"observation field {self.name} dimension must be >= 1")
        _finite(self.scale, f"observation field {self.name} scale")
        _nonempty(self.frame, f"observation field {self.name} frame")
        _require(
            self.quaternion_order in {"none", "wxyz"},
            f"invalid quaternion order for {self.name}",
        )
        if self.canonicalize_scalar_nonnegative:
            _require(
                self.quaternion_order == "wxyz",
                f"only quaternions may be sign-canonicalized ({self.name})",
            )


@dataclass(frozen=True, kw_only=True)
class ObservationConfig:
    version: str
    fields: tuple[ObservationFieldConfig, ...]
    observed_joint_names: tuple[str, ...]
    contact_link_names: tuple[str, ...]
    dimension: int = contract_field(unit="scalar", minimum=1)
    history_length: int = contract_field(unit="frame", minimum=1)
    clip_min: float = contract_field(unit="normalized_scalar")
    clip_max: float = contract_field(unit="normalized_scalar")
    world_positions_relative_to_environment_origin: bool
    nonfinite_behavior: str

    def validate(self) -> None:
        _nonempty(self.version, "observation.version")
        _require(
            len(self.fields) == 17,
            "observation contract must contain exactly 17 ordered fields",
        )
        for item in self.fields:
            item.validate()
        _require(
            len({item.name for item in self.fields}) == len(self.fields),
            "observation field names must be unique",
        )
        expected_joints = (
            "shoulder_pitch_r_joint",
            "shoulder_roll_r_joint",
            "shoulder_yaw_r_joint",
            "elbow_pitch_r_joint",
            "elbow_yaw_r_joint",
            "wrist_pitch_r_joint",
            "wrist_roll_r_joint",
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_thumb_3_joint",
            "right_thumb_4_joint",
            "right_index_1_joint",
            "right_index_2_joint",
            "right_middle_1_joint",
            "right_middle_2_joint",
            "right_ring_1_joint",
            "right_ring_2_joint",
            "right_little_1_joint",
            "right_little_2_joint",
        )
        _require(
            self.observed_joint_names == expected_joints,
            "observation joint order must match the 19-joint contract",
        )
        _require(
            self.contact_link_names == expected_joints[7:],
            "observation contact links must match the ordered 12 right-hand links",
        )
        _require(
            sum(item.dimension for item in self.fields) == self.dimension == 106,
            "observation dimension must be 106",
        )
        _require(self.history_length == 1, "observation history_length must be 1")
        _finite(self.clip_min, "observation.clip_min")
        _finite(self.clip_max, "observation.clip_max")
        _require(
            self.clip_min == -10.0 and self.clip_max == 10.0,
            "observation clipping must be [-10, 10]",
        )
        _require(
            self.world_positions_relative_to_environment_origin,
            "world positions must be environment-origin relative",
        )
        _require(
            self.nonfinite_behavior == "safe_terminate_with_reason",
            "non-finite observations must safely terminate with a reason",
        )


@dataclass(frozen=True, kw_only=True)
class ActionConfig:
    version: str
    dimension: int = contract_field(unit="scalar", minimum=1)
    arm_joint_names: tuple[str, ...]
    normalized_min: float
    normalized_max: float
    target_formula: str
    center_joint_positions_rad: tuple[float, ...]
    half_range_rad: tuple[float, ...]
    maximum_joint_target_step_rad: float = contract_field(unit="rad/policy_step", minimum=0.0)
    distribution: str

    def validate(self) -> None:
        _nonempty(self.version, "action.version")
        _require(
            self.dimension == 7,
            "action dimension must be seven arm joints",
        )
        _require(
            len(self.arm_joint_names) == 7 and len(set(self.arm_joint_names)) == 7,
            "action arm_joint_names must contain seven unique joints",
        )
        _require(
            self.normalized_min == -1.0 and self.normalized_max == 1.0,
            "actions must lie in [-1, 1]",
        )
        _require(
            self.target_formula == "center_plus_action_times_half_range",
            "unsupported arm target formula",
        )
        for name, values in (
            ("center_joint_positions_rad", self.center_joint_positions_rad),
            ("half_range_rad", self.half_range_rad),
        ):
            _require(len(values) == 7, f"action.{name} must have seven values")
            for index, value in enumerate(values):
                _finite(value, f"action.{name}[{index}]")
        for index, value in enumerate(self.half_range_rad):
            _positive(value, f"action.half_range_rad[{index}]")
        _require(
            self.maximum_joint_target_step_rad == 0.01,
            "maximum joint target step must be 0.01 rad",
        )
        _require(
            self.distribution == "bounded_tanh_normal",
            "action distribution must be bounded_tanh_normal",
        )


@dataclass(frozen=True, kw_only=True)
class ReadyConfig:
    position_error_m: float = contract_field(unit="m", minimum=0.0)
    orientation_error_rad: float = contract_field(unit="rad", minimum=0.0)
    dwell_time_s: float = contract_field(unit="s", minimum=0.0)
    grasp_center_speed_m_s: float = contract_field(unit="m/s", minimum=0.0)
    arm_speed_rad_s: float = contract_field(unit="rad/s", minimum=0.0)
    object_displacement_m: float = contract_field(unit="m", minimum=0.0)
    object_tilt_rad: float = contract_field(unit="rad", minimum=0.0)
    object_linear_speed_m_s: float = contract_field(unit="m/s", minimum=0.0)
    object_angular_speed_rad_s: float = contract_field(unit="rad/s", minimum=0.0)
    contact_impulse_n_s: float = contract_field(unit="N*s", minimum=0.0)
    diagnostic_only: bool
    participates_in_curriculum: bool
    grants_reward: bool

    def validate(self) -> None:
        expected = (
            0.005,
            0.10,
            0.30,
            0.05,
            0.20,
            0.015,
            math.radians(45),
            0.05,
            0.50,
            0.05,
        )
        actual = (
            self.position_error_m,
            self.orientation_error_rad,
            self.dwell_time_s,
            self.grasp_center_speed_m_s,
            self.arm_speed_rad_s,
            self.object_displacement_m,
            self.object_tilt_rad,
            self.object_linear_speed_m_s,
            self.object_angular_speed_rad_s,
            self.contact_impulse_n_s,
        )
        for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
            _positive(value, f"ready threshold {index}")
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"ready threshold {index} violates FR-11",
            )
        _require(
            self.diagnostic_only and not self.participates_in_curriculum and not self.grants_reward,
            "strict Ready must be diagnostic-only",
        )


@dataclass(frozen=True, kw_only=True)
class AdmissionConfig:
    position_error_m: float = contract_field(unit="m", minimum=0.0)
    orientation_error_rad: float = contract_field(unit="rad", minimum=0.0)
    dwell_time_s: float = contract_field(unit="s", minimum=0.0)
    grasp_center_speed_m_s: float = contract_field(unit="m/s", minimum=0.0)
    object_tilt_rad: float = contract_field(unit="rad", minimum=0.0)
    illegal_contact_impulse_n_s: float = contract_field(unit="N*s", minimum=0.0)
    requires_finite_state: bool
    requires_valid_target: bool
    requires_reach_phase: bool
    dwell_required: bool

    def validate(self) -> None:
        expected = (
            0.010,
            0.10,
            0.05,
            math.radians(45),
            0.05,
        )
        actual = (
            self.position_error_m,
            self.orientation_error_rad,
            self.grasp_center_speed_m_s,
            self.object_tilt_rad,
            self.illegal_contact_impulse_n_s,
        )
        for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
            _positive(value, f"admission threshold {index}")
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"admission threshold {index} violates FR-08",
            )
        _require(
            self.requires_finite_state and self.requires_valid_target and self.requires_reach_phase,
            "admission must require finite state, valid target and Reach phase",
        )
        _require(
            math.isclose(self.dwell_time_s, 0.30, rel_tol=0.0, abs_tol=1e-12),
            "Option admission dwell must be exactly 0.30 s",
        )
        _require(self.dwell_required, "automatic Option admission must require dwell")


@dataclass(frozen=True, kw_only=True)
class OptionConfig:
    version: str
    state_machine_version: str
    settle_time_s: float = contract_field(unit="s", minimum=0.0)
    closing_time_s: float = contract_field(unit="s", minimum=0.0)
    hold_time_s: float = contract_field(unit="s", minimum=0.0)
    lift_time_s: float = contract_field(unit="s", minimum=0.0)
    lift_hold_time_s: float = contract_field(unit="s", minimum=0.0)
    lift_height_m: float = contract_field(unit="m", minimum=0.0)
    active_hand_joint_names: tuple[str, ...]
    latch_arm_target_on_accept: bool
    script_starts_next_policy_step: bool
    ignore_policy_while_active: bool
    actor_update_mask_while_active: bool
    automatic_trigger_on_admission_dwell: bool
    maximum_acceptances_per_episode: int

    def validate(self) -> None:
        _nonempty(self.version, "option.version")
        _nonempty(self.state_machine_version, "option.state_machine_version")
        expected = (0.5, 3.5, 1.0, 2.0, 5.0)
        actual = (
            self.settle_time_s,
            self.closing_time_s,
            self.hold_time_s,
            self.lift_time_s,
            self.lift_hold_time_s,
        )
        for name, value, target in zip(
            ("settle", "closing", "hold", "lift", "lift_hold"),
            actual,
            expected,
            strict=True,
        ):
            _positive(value, f"option.{name}_time_s")
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"option {name} duration violates FR-10",
            )
        _require(self.lift_height_m == 0.06, "Option lift height must be 0.06 m")
        _require(
            len(self.active_hand_joint_names) == 6 and len(set(self.active_hand_joint_names)) == 6,
            "Option must control six unique active hand joints",
        )
        _require(
            self.latch_arm_target_on_accept and self.script_starts_next_policy_step and self.ignore_policy_while_active,
            "Option latch/timing/override semantics must remain enabled",
        )
        _require(
            not self.actor_update_mask_while_active,
            "actor_update_mask must be false while Option is active",
        )
        _require(
            self.automatic_trigger_on_admission_dwell,
            "Option trigger must be automatic after admission dwell",
        )
        _require(
            self.maximum_acceptances_per_episode == 1,
            "only one Option acceptance is allowed per episode",
        )


@dataclass(frozen=True, kw_only=True)
class RewardConfig:
    version: str
    potential_discount: float
    position_potential_scale_m: float = contract_field(unit="m", minimum=0.0)
    orientation_potential_scale_rad: float = contract_field(unit="rad", minimum=0.0)
    object_displacement_scale_m: float = contract_field(unit="m", minimum=0.0)
    position_potential_weight: float
    orientation_potential_weight: float
    step_penalty: float
    joint_velocity_penalty_weight: float
    action_rate_penalty_weight: float
    object_displacement_penalty_weight: float
    unsafe_object_displacement_penalty: float
    accepted_trigger_reward: float
    rejected_trigger_penalty: float
    admissible_wait_penalty: float
    lift_5cm_reward: float
    complete_success_reward: float
    script_failure_penalty: float
    ready_reward: float
    potential_progress_semantics: str

    def validate(self) -> None:
        _nonempty(self.version, "reward.version")
        for name, target in (
            ("position_potential_scale_m", 0.08),
            ("orientation_potential_scale_rad", 0.35),
            ("object_displacement_scale_m", 0.020),
        ):
            value = getattr(self, name)
            _positive(value, f"reward.{name}")
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"reward.{name} violates inherited versioned semantics",
            )
        expected = (
            0.99,
            6.0,
            1.0,
            0.01,
            0.002,
            0.005,
            0.5,
            10.0,
            3.0,
            0.0,
            0.0,
            10.0,
            50.0,
            25.0,
            0.0,
        )
        actual = (
            self.potential_discount,
            self.position_potential_weight,
            self.orientation_potential_weight,
            self.step_penalty,
            self.joint_velocity_penalty_weight,
            self.action_rate_penalty_weight,
            self.object_displacement_penalty_weight,
            self.unsafe_object_displacement_penalty,
            self.accepted_trigger_reward,
            self.rejected_trigger_penalty,
            self.admissible_wait_penalty,
            self.lift_5cm_reward,
            self.complete_success_reward,
            self.script_failure_penalty,
            self.ready_reward,
        )
        for index, (value, target) in enumerate(zip(actual, expected, strict=True)):
            _positive(value, f"reward value {index}", allow_zero=True)
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"reward value {index} violates FR-12",
            )
        _require(
            self.potential_progress_semantics == "gamma_phi_next_minus_phi_current",
            "reward must use discount-consistent potential progress",
        )


@dataclass(frozen=True, kw_only=True)
class TerminationConfig:
    version: str
    lift_event_height_m: float = contract_field(unit="m", minimum=0.0)
    maximum_safe_displacement_m: float = contract_field(unit="m", minimum=0.0)
    topple_angle_rad: float = contract_field(unit="rad", minimum=0.0)
    drop_below_table_top_m: float = contract_field(unit="m", minimum=0.0)
    far_xy_distance_m: float = contract_field(unit="m", minimum=0.0)
    bilateral_contact_force_n: float = contract_field(unit="N", minimum=0.0)
    right_arm_table_contact_force_limit_n: float = contract_field(unit="N", minimum=0.0)
    bilateral_contact_dwell_steps: int = contract_field(unit="policy_step", minimum=1)
    lift_support_dwell_steps: int = contract_field(unit="policy_step", minimum=1)
    thumb_contact_group: str
    opposing_contact_group: str
    lift_support_contact_group: str
    drop_frame_semantics: str
    far_frame_semantics: str
    terminate_on_drop: bool
    terminate_on_illegal_contact: bool
    terminate_on_nonfinite: bool
    success_requires_option_accepted: bool
    success_requires_completed_lift_hold: bool
    success_requires_bilateral_persistent_contact: bool
    success_and_danger_mutually_exclusive: bool
    unique_primary_reason: bool
    record_all_auxiliary_reasons: bool

    def validate(self) -> None:
        _nonempty(self.version, "termination.version")
        _require(self.lift_event_height_m == 0.05, "Lift5cm threshold must be 0.05 m")
        _positive(self.maximum_safe_displacement_m, "termination.maximum_safe_displacement_m")
        _require(
            math.isclose(self.topple_angle_rad, math.radians(45), rel_tol=0.0, abs_tol=1e-12),
            "topple angle must be 45 degrees",
        )
        expected_scalars = (
            (self.drop_below_table_top_m, 0.10, "drop_below_table_top_m"),
            (self.far_xy_distance_m, 0.90, "far_xy_distance_m"),
            (self.bilateral_contact_force_n, 0.20, "bilateral_contact_force_n"),
            (self.right_arm_table_contact_force_limit_n, 5.0, "right_arm_table_contact_force_limit_n"),
        )
        for value, target, name in expected_scalars:
            _positive(value, f"termination.{name}")
            _require(
                math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12),
                f"termination.{name} violates inherited versioned semantics",
            )
        _require(self.bilateral_contact_dwell_steps == 3, "bilateral contact dwell must be 3 policy steps")
        _require(self.lift_support_dwell_steps == 5, "lift support dwell must be 5 policy steps")
        _require(self.thumb_contact_group == "thumb_side", "unexpected thumb contact group")
        _require(self.opposing_contact_group == "four_finger_opposing", "unexpected opposing contact group")
        _require(
            self.lift_support_contact_group == "right_hand_observation_links",
            "unexpected lift support contact group",
        )
        _require(
            self.drop_frame_semantics == "object_center_z_below_table_top",
            "unexpected drop frame semantics",
        )
        _require(
            self.far_frame_semantics == "environment_relative_xy_norm",
            "unexpected far distance frame semantics",
        )
        boolean_start = next(index for index, item in enumerate(fields(self)) if item.name == "terminate_on_drop")
        for item in fields(self)[boolean_start:]:
            _require(
                getattr(self, item.name) is True,
                f"termination.{item.name} must be true",
            )


@dataclass(frozen=True, kw_only=True)
class TaskConfig:
    version: str
    scene: SceneConfig
    observation: ObservationConfig
    action: ActionConfig
    ready: ReadyConfig
    admission: AdmissionConfig
    option: OptionConfig
    reward: RewardConfig
    termination: TerminationConfig

    def validate(self) -> None:
        _nonempty(self.version, "task.version")
        for item in fields(self)[1:]:
            getattr(self, item.name).validate()


@dataclass(frozen=True, kw_only=True)
class UniformDistributionConfig:
    distribution: str
    minimum: float
    maximum: float
    unit: str

    def validate(self) -> None:
        _require(
            self.distribution == "uniform",
            "randomization distributions must be uniform",
        )
        _finite(self.minimum, "distribution.minimum")
        _finite(self.maximum, "distribution.maximum")
        _require(self.minimum < self.maximum, "distribution minimum must be below maximum")
        _nonempty(self.unit, "distribution.unit")


@dataclass(frozen=True, kw_only=True)
class RandomizationConfig:
    version: str
    radius: UniformDistributionConfig
    height: UniformDistributionConfig
    mass: UniformDistributionConfig
    hand_sliding_friction: UniformDistributionConfig
    table_sliding_friction: UniformDistributionConfig
    fixed_torsional_friction: float
    fixed_rolling_friction: float
    independent_variables: bool
    enabled_from_stage: str
    inertia_model: str
    policy_observes_parameters: bool
    target_uses_actual_geometric_center: bool

    def validate(self) -> None:
        _nonempty(self.version, "randomization.version")
        expected = (
            (self.radius, 0.010, 0.015, "m"),
            (self.height, 0.080, 0.120, "m"),
            (self.mass, 0.100, 0.130, "kg"),
            (self.hand_sliding_friction, 0.7, 1.1, "1"),
            (self.table_sliding_friction, 0.7, 1.1, "1"),
        )
        for distribution, minimum, maximum, unit in expected:
            distribution.validate()
            _require(
                math.isclose(distribution.minimum, minimum, rel_tol=0.0, abs_tol=1e-12),
                "randomization lower bound violates FR-04",
            )
            _require(
                math.isclose(distribution.maximum, maximum, rel_tol=0.0, abs_tol=1e-12),
                "randomization upper bound violates FR-04",
            )
            _require(distribution.unit == unit, f"randomization unit must be {unit}")
        _positive(
            self.fixed_torsional_friction,
            "randomization.fixed_torsional_friction",
            allow_zero=True,
        )
        _positive(
            self.fixed_rolling_friction,
            "randomization.fixed_rolling_friction",
            allow_zero=True,
        )
        _require(self.independent_variables, "the five random variables must be independent")
        _require(self.enabled_from_stage == "S0", "randomization must be enabled from S0")
        _require(
            self.inertia_model == "solid_square_prism_from_sampled_geometry_and_mass",
            "mass randomization must update physical square-prism inertia",
        )
        _require(
            not self.policy_observes_parameters,
            "policy must not observe randomization parameters",
        )
        _require(
            self.target_uses_actual_geometric_center,
            "grasp target must use actual cylinder center",
        )


@dataclass(frozen=True, kw_only=True)
class CurriculumStageConfig:
    name: str
    x_offset_range_m: tuple[float, float]
    y_offset_range_m: tuple[float, float]
    current_fraction: float
    replay_fraction: float
    boundary_fraction: float
    replay_source: str
    corner_jitter_half_width_m: tuple[float, float]

    def validate(self) -> None:
        _require(self.name in {"S0", "S1", "S2", "S3"}, "invalid curriculum stage name")
        for axis, values in (
            ("x", self.x_offset_range_m),
            ("y", self.y_offset_range_m),
        ):
            _require(len(values) == 2, f"{self.name} {axis} range must contain two bounds")
            _finite(values[0], f"{self.name}.{axis}.minimum")
            _finite(values[1], f"{self.name}.{axis}.maximum")
            _require(0.0 <= values[0] <= values[1], f"{self.name} {axis} range is invalid")
        for name in ("current_fraction", "replay_fraction", "boundary_fraction"):
            _probability(getattr(self, name), f"{self.name}.{name}")
        _require(
            math.isclose(
                self.current_fraction + self.replay_fraction + self.boundary_fraction,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{self.name} mixture must sum to one",
        )
        _require(
            len(self.corner_jitter_half_width_m) == 2,
            "corner jitter needs x/y half widths",
        )
        for value in self.corner_jitter_half_width_m:
            _positive(value, "corner jitter", allow_zero=True)


@dataclass(frozen=True, kw_only=True)
class PromotionGateConfig:
    from_stage: str
    to_stage: str
    window_episodes: int
    option_accepted_rate_min: float
    lift_5cm_rate_min: float
    complete_success_rate_min: float
    dangerous_rate_max: float

    def validate(self) -> None:
        _require(self.window_episodes >= 1, "promotion window must be positive")
        for name in (
            "option_accepted_rate_min",
            "lift_5cm_rate_min",
            "complete_success_rate_min",
            "dangerous_rate_max",
        ):
            _probability(getattr(self, name), f"promotion.{name}")


@dataclass(frozen=True, kw_only=True)
class CurriculumConfig:
    version: str
    sampler_version: str
    initial_stage: str
    monotonic: bool
    cylinder_origin_xy_m: tuple[float, float]
    stages: tuple[CurriculumStageConfig, ...]
    boundary_corners_xy_offset_m: tuple[tuple[float, float], ...]
    promotion_gates: tuple[PromotionGateConfig, ...]
    consecutive_passing_windows: int
    count_episode_by_stage_at_reset: bool
    ready_participates_in_promotion: bool

    def validate(self) -> None:
        _nonempty(self.version, "curriculum.version")
        _nonempty(self.sampler_version, "curriculum.sampler_version")
        _require(self.initial_stage == "S0", "curriculum must start at S0")
        _require(self.monotonic, "curriculum promotion must be monotonic")
        _require(len(self.cylinder_origin_xy_m) == 2, "curriculum cylinder origin must be XY")
        for value in self.cylinder_origin_xy_m:
            _finite(value, "curriculum cylinder origin")
        _require(
            tuple(stage.name for stage in self.stages) == ("S0", "S1", "S2", "S3"),
            "curriculum must define ordered S0-S3",
        )
        for stage in self.stages:
            stage.validate()
        expected_ranges = (
            ((0.005, 0.008), (0.005, 0.008)),
            ((0.0, 0.035), (0.0, 0.050)),
            ((0.0, 0.045), (0.0, 0.070)),
            ((0.0, 0.045), (0.0, 0.070)),
        )
        expected_mix = (
            (1.0, 0.0, 0.0),
            (0.8, 0.2, 0.0),
            (0.8, 0.2, 0.0),
            (0.7, 0.0, 0.3),
        )
        for stage, ranges, mix in zip(self.stages, expected_ranges, expected_mix, strict=True):
            _require(
                stage.x_offset_range_m == ranges[0] and stage.y_offset_range_m == ranges[1],
                f"{stage.name} ranges violate FR-14",
            )
            _require(
                (stage.current_fraction, stage.replay_fraction, stage.boundary_fraction) == mix,
                f"{stage.name} mixture violates FR-14",
            )
        _require(
            len(self.boundary_corners_xy_offset_m) == 3,
            "S3 must define three non-trivial boundary corners",
        )
        _require(
            tuple((gate.from_stage, gate.to_stage) for gate in self.promotion_gates)
            == (("S0", "S1"), ("S1", "S2"), ("S2", "S3")),
            "promotion gates must be S0->S1->S2->S3",
        )
        expected_gates = (
            (2048, 0.70, 0.60, 0.50, 0.10),
            (4096, 0.80, 0.70, 0.60, 0.10),
            (8192, 0.85, 0.80, 0.70, 0.10),
        )
        for gate, expected in zip(self.promotion_gates, expected_gates, strict=True):
            gate.validate()
            _require(
                (
                    gate.window_episodes,
                    gate.option_accepted_rate_min,
                    gate.lift_5cm_rate_min,
                    gate.complete_success_rate_min,
                    gate.dangerous_rate_max,
                )
                == expected,
                f"promotion gate {gate.from_stage} violates FR-14",
            )
        _require(
            self.consecutive_passing_windows == 2,
            "promotion requires two consecutive complete windows",
        )
        _require(
            self.count_episode_by_stage_at_reset,
            "episodes must count against stage_at_reset",
        )
        _require(
            not self.ready_participates_in_promotion,
            "strict Ready must not participate in promotion",
        )


@dataclass(frozen=True, kw_only=True)
class NetworkConfig:
    actor_hidden_dimensions: tuple[int, ...]
    critic_hidden_dimensions: tuple[int, ...]
    activation: str

    def validate(self) -> None:
        for name, values in (
            ("actor", self.actor_hidden_dimensions),
            ("critic", self.critic_hidden_dimensions),
        ):
            _require(bool(values), f"{name} network must have hidden layers")
            _require(
                all(value >= 1 for value in values),
                f"{name} hidden dimensions must be positive",
            )
        _require(self.activation in {"elu", "relu", "tanh"}, "unsupported network activation")


@dataclass(frozen=True, kw_only=True)
class PPOConfig:
    version: str
    network: NetworkConfig
    rollout_steps_per_env: int
    learning_epochs: int
    mini_batches: int
    actor_learning_rate: float
    critic_learning_rate: float
    clip_ratio: float
    value_loss_coefficient: float
    entropy_coefficient: float
    gamma: float
    gae_lambda: float
    desired_kl: float
    maximum_gradient_norm: float
    initial_continuous_std: float
    minimum_continuous_std: float
    maximum_continuous_std: float
    option_return_semantics: str
    empty_actor_mask_behavior: str

    def validate(self) -> None:
        _nonempty(self.version, "ppo.version")
        self.network.validate()
        for name in ("rollout_steps_per_env", "learning_epochs", "mini_batches"):
            _require(getattr(self, name) >= 1, f"ppo.{name} must be >= 1")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "clip_ratio",
            "value_loss_coefficient",
            "desired_kl",
            "maximum_gradient_norm",
            "initial_continuous_std",
            "minimum_continuous_std",
            "maximum_continuous_std",
        ):
            _positive(getattr(self, name), f"ppo.{name}")
        _positive(self.entropy_coefficient, "ppo.entropy_coefficient", allow_zero=True)
        _probability(self.gamma, "ppo.gamma")
        _probability(self.gae_lambda, "ppo.gae_lambda")
        _require(
            self.minimum_continuous_std <= self.initial_continuous_std <= self.maximum_continuous_std,
            "initial continuous std must lie within min/max",
        )
        _require(
            self.option_return_semantics == "unit_discount_lambda_across_script",
            "Option return must use unit discount/lambda across script",
        )
        _require(
            self.empty_actor_mask_behavior == "skip_actor_update_without_nan",
            "empty actor masks must safely skip actor updates",
        )


@dataclass(frozen=True, kw_only=True)
class RunConfig:
    seed: int
    output_dir: str
    save_interval_iterations: int
    maximum_iterations: int
    command: str
    mode: str
    code_version: str
    checkpoint_source: str

    def validate(self) -> None:
        _require(0 <= self.seed <= 2**63 - 1, "run.seed must lie in [0, 2^63-1]")
        _nonempty(self.output_dir, "run.output_dir")
        _require(
            self.save_interval_iterations >= 1,
            "run.save_interval_iterations must be >= 1",
        )
        _require(self.maximum_iterations >= 1, "run.maximum_iterations must be >= 1")
        _nonempty(self.command, "run.command")
        _require(
            self.mode in {"train", "evaluate", "viewer", "smoke", "validate_asset"},
            "invalid run.mode",
        )
        _nonempty(self.code_version, "run.code_version")
        _require(
            self.checkpoint_source in {"none", "mujoco"},
            "Isaac or unknown checkpoint sources are forbidden",
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationThresholdConfig:
    option_accepted_rate_min: float
    lift_5cm_rate_min: float
    complete_success_rate_min: float
    dangerous_rate_max: float

    def validate(self) -> None:
        for item in fields(self):
            _probability(getattr(self, item.name), f"evaluation threshold {item.name}")


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
    validation_manifest_path: str
    final_manifest_path: str
    checkpoint_path: str
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    standard_case_count: int
    randomized_case_count: int
    deterministic_actions: bool
    preliminary_thresholds: EvaluationThresholdConfig
    final_thresholds: EvaluationThresholdConfig

    def validate(self) -> None:
        _nonempty(self.validation_manifest_path, "evaluation.validation_manifest_path")
        _nonempty(self.final_manifest_path, "evaluation.final_manifest_path")
        _require(
            len(self.training_seeds) == 3 and len(set(self.training_seeds)) == 3,
            "evaluation must declare three unique training seeds",
        )
        _require(
            bool(self.evaluation_seeds) and len(set(self.evaluation_seeds)) == len(self.evaluation_seeds),
            "evaluation seeds must be non-empty and unique",
        )
        _require(
            self.standard_case_count == 256 and self.randomized_case_count == 256,
            "formal manifests must contain 256 standard and 256 randomized cases",
        )
        _require(
            self.deterministic_actions,
            "formal evaluation must be deterministic",
        )
        self.preliminary_thresholds.validate()
        self.final_thresholds.validate()


@dataclass(frozen=True, kw_only=True)
class ResolvedRunConfig:
    """Fully resolved immutable configuration returned by the loader."""

    platform: PlatformConfig
    asset: AssetConfig
    physics: PhysicsConfig
    task: TaskConfig
    randomization: RandomizationConfig
    curriculum: CurriculumConfig
    ppo: PPOConfig
    run: RunConfig
    evaluation: EvaluationConfig

    def validate(self) -> None:
        # Keep structural and scalar validation local to each configuration
        # section. Cross-section equality was part of the removed contract
        # gate and must not reject otherwise usable configurations.
        _validate_all_nested(self)


__all__ = [
    "ActionConfig",
    "ActuatorCalibrationConfig",
    "ActuatorGroupConfig",
    "AdmissionConfig",
    "AssetConfig",
    "ContactConfig",
    "CurriculumConfig",
    "CurriculumStageConfig",
    "EvaluationConfig",
    "EvaluationThresholdConfig",
    "MaterialConfig",
    "NetworkConfig",
    "ObservationConfig",
    "ObservationFieldConfig",
    "OptionConfig",
    "PPOConfig",
    "PhysicsConfig",
    "PlatformConfig",
    "PromotionGateConfig",
    "RandomizationConfig",
    "ReadyConfig",
    "ResolvedRunConfig",
    "RewardConfig",
    "RunConfig",
    "SceneConfig",
    "SolverConfig",
    "TaskConfig",
    "TerminationConfig",
    "UniformDistributionConfig",
    "contract_field",
]
