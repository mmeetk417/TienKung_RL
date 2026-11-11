"""The single per-environment ``reach_then_close`` task runtime.

This module owns ordering only.  Action conversion, observations, admission,
Ready, Option progression, reward, termination, randomization, and curriculum
remain delegated to their versioned M06--M11 implementations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import torch
from numpy.typing import NDArray

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import (
    ActuatorControlRanges,
    ContactFrame,
    EnvironmentBackend,
    MujocoEnvironmentBackend,
    ParallelStepResult,
    PhysicsState,
    ResetSpec,
    SimulationDiagnostics,
    is_illegal_contact,
)

from .actions import ArmTargetController, policy_arm_action
from .admission import (
    AdmissionEpisodeState,
    AdmissionInput,
    AdmissionResult,
    ReadyEpisodeState,
    ReadyInput,
    ReadyResult,
    step_admission,
    step_ready,
)
from .contracts import (
    HAND_CONTACT_LINK_NAMES,
    PHYSICAL_ACTION_NAMES,
    Observation106,
    PhysicalAction,
)
from .curriculum import CurriculumManager, EpisodeCurriculumResult, ForcedTargetAuthority, PromotionDiagnostics
from .observations import ObservationInputs, build_observation
from .option import (
    LiftPreparation,
    OptionController,
    OptionEpisodeState,
    OptionPhase,
    OptionSubstepObservation,
)
from .randomization import (
    CylinderRandomizer,
    EpisodeRandomizationParameters,
    ForcedCylinderParameters,
    ForcedParameterAuthority,
)
from .reward import RewardBreakdown, RewardEpisodeState, RewardEvents, RewardState, compute_reward
from .termination import (
    InheritedTerminationInput,
    OutcomeClassifier,
    OutcomeInput,
    OutcomeResult,
    TerminationReason,
    evaluate_inherited_termination_signals,
)

FloatArray = NDArray[np.float64]

TASK_SNAPSHOT_SCHEMA_VERSION = "reach_then_close_task_snapshot_v3"
TASK_EXTRAS_SCHEMA_VERSION = "reach_then_close_task_extras_v3"
_TARGET_TCP_SITE = "target_tcp_site"
_TCP_SITE = "right_tcp_site"
_GRASP_CENTER_SITE = "grasp_center_site"


def _deep_freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _tuple3(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple4(values: FloatArray) -> tuple[float, float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]), float(values[3]))


def _quat_conjugate(quaternion: FloatArray) -> FloatArray:
    return np.asarray(
        (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]),
        dtype=np.float64,
    )


def _quat_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    result = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion product is non-finite or degenerate")
    result /= norm
    if result[0] < 0.0:
        result *= -1.0
    return np.ascontiguousarray(result, dtype=np.float64)


def _quat_rotate(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return np.asarray(
        2.0 * np.dot(imaginary, vector) * imaginary
        + (scalar * scalar - np.dot(imaginary, imaginary)) * vector
        + 2.0 * scalar * np.cross(imaginary, vector),
        dtype=np.float64,
    )


def _orientation_error_rad(target: FloatArray, current: FloatArray) -> float:
    relative = _quat_multiply(_quat_conjugate(target), current)
    return 2.0 * math.acos(min(1.0, abs(float(relative[0]))))


def _object_tilt_rad(quaternion: FloatArray) -> float:
    axis = _quat_rotate(quaternion, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    return math.acos(min(1.0, max(-1.0, float(axis[2]))))


def _finite_physics(state: PhysicsState, diagnostics: SimulationDiagnostics) -> bool:
    arrays = (
        state.joint_positions_rad,
        state.joint_velocities_rad_s,
        state.body_positions_m,
        state.body_quaternions_wxyz,
        state.site_positions_m,
        state.site_quaternions_wxyz,
        state.site_linear_velocities_m_s,
        state.site_angular_velocities_rad_s,
        state.actuator_targets_rad,
        state.actuator_forces_n_m,
    )
    return not diagnostics.nonfinite_state and all(bool(np.isfinite(value).all()) for value in arrays)


def _empty_contacts(config: ResolvedRunConfig, state: PhysicsState) -> ContactFrame:
    return ContactFrame(
        contacts=(),
        substeps_collected=0,
        hand_link_names=HAND_CONTACT_LINK_NAMES,
        hand_contact_force_norms_n=np.zeros(len(HAND_CONTACT_LINK_NAMES), dtype=np.float64),
        maximum_contact_force_n=0.0,
        maximum_contact_impulse_n_s=0.0,
        maximum_illegal_contact_impulse_n_s=0.0,
        contact_capacity=config.physics.contact.maximum_contacts,
        maximum_contacts_seen=0,
        physics_step=state.total_physics_steps,
        simulation_time_s=state.simulation_time_s,
    )


@dataclass(frozen=True, slots=True)
class _ContactReduction:
    thumb_force_n: float
    opposing_force_n: float
    net_hand_force_n: float


def _reduce_contacts(contacts: ContactFrame, substep_index: int | None = None) -> _ContactReduction:
    thumb = 0.0
    opposing = 0.0
    net_by_link = {name: np.zeros(3, dtype=np.float64) for name in HAND_CONTACT_LINK_NAMES}
    for sample in contacts.contacts:
        if substep_index is not None and sample.substep_index != substep_index:
            continue
        first = set(sample.geom1_groups)
        second = set(sample.geom2_groups)
        other = second if "cylinder" in first else first if "cylinder" in second else set()
        if "thumb_side" in other:
            thumb = max(thumb, sample.normal_force_n)
        if "four_finger_opposing" in other:
            opposing = max(opposing, sample.normal_force_n)
        if sample.body1 in net_by_link:
            net_by_link[sample.body1] -= sample.force_on_geom2_world_n
        if sample.body2 in net_by_link:
            net_by_link[sample.body2] += sample.force_on_geom2_world_n
    return _ContactReduction(
        thumb_force_n=thumb,
        opposing_force_n=opposing,
        net_hand_force_n=float(sum(np.linalg.norm(value) for value in net_by_link.values())),
    )


@dataclass(frozen=True, slots=True)
class _ReducedTaskState:
    position_error_m: float
    orientation_error_rad: float
    orientation_error_quaternion_wxyz: tuple[float, float, float, float]
    grasp_center_speed_m_s: float
    arm_speed_max_rad_s: float
    object_displacement_m: float
    object_tilt_rad: float
    object_linear_speed_m_s: float
    object_angular_speed_rad_s: float
    illegal_contact_impulse_n_s: float
    palm_clearance_m: float
    object_lift_height_m: float
    thumb_force_n: float
    opposing_force_n: float
    net_hand_force_n: float
    dropped: bool
    object_far: bool
    toppled: bool
    excessive_displacement: bool
    illegal_contact: bool
    finite: bool
    simulation_error: bool


@dataclass(frozen=True, slots=True)
class PreparedPolicyStep:
    """The immutable output of policy-step stages 1--3 before physics."""

    physical_action: PhysicalAction
    reach_transition: bool
    actor_update_mask: bool
    critic_update_mask: bool
    discount_factor: float
    gae_lambda_factor: float
    policy_arm_action: tuple[float, ...]
    before_reward_state: RewardState


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Read-only task diagnostics for viewer, evaluation, and telemetry."""

    schema_version: str
    env_id: int
    episode_step: int
    stage_at_reset: str
    target_tcp_position_m: tuple[float, float, float]
    target_tcp_quaternion_wxyz: tuple[float, float, float, float]
    cylinder_reset_position_m: tuple[float, float, float]
    option_phase: str
    option_phase_step: int
    option_script_steps: int
    option_script_progress: float
    option_support_progress: float
    trigger_requested: bool
    trigger_accepted: bool
    trigger_rejected: bool
    option_accepted_ever: bool
    strict_ready: bool
    strict_ready_ever: bool
    strict_ready_consecutive_steps: int
    option_admissible: bool
    admission_instantaneous: bool
    admission_consecutive_steps: int
    admission_dwell_steps_required: int
    position_error_m: float
    orientation_error_rad: float
    object_displacement_m: float
    object_tilt_rad: float
    illegal_contact_impulse_n_s: float
    step_reward: float
    episode_reward_total: float
    lift_5cm: bool
    success: bool
    unsafe: bool
    done: bool
    primary_termination_reason: str | None
    secondary_termination_reasons: tuple[str, ...]
    admission_failures: tuple[str, ...]
    reward_terms: Mapping[str, float]
    randomization: Mapping[str, object]
    simulation_warning_names: tuple[str, ...]
    contact_count: int
    maximum_contact_force_n: float
    maximum_contact_impulse_n_s: float
    maximum_contacts_seen: int
    contact_capacity: int
    contact_capacity_exceeded: bool
    solver_anomaly: bool
    nonfinite_state: bool
    physics_step: int
    simulation_time_s: float


@dataclass(frozen=True, slots=True)
class TaskStepTransition:
    """Complete pre-reset transition emitted by policy-step stage 10."""

    env_id: int
    terminal_observation: Observation106
    reward: float
    done: bool
    timed_out: bool
    actor_update_mask: bool
    critic_update_mask: bool
    discount_factor: float
    gae_lambda_factor: float
    admission: AdmissionResult
    ready: ReadyResult
    outcome: OutcomeResult
    reward_breakdown: RewardBreakdown
    snapshot: TaskSnapshot
    curriculum_update: PromotionDiagnostics | None


@dataclass(slots=True)
class EpisodeState:
    """All episode-local state required by design section 11.1."""

    episode_step: int
    current_arm_targets_rad: FloatArray
    previous_arm_targets_rad: FloatArray
    physical_action: PhysicalAction
    previous_policy_arm_action: tuple[float, ...]
    option_state: OptionEpisodeState
    admission_state: AdmissionEpisodeState
    ready_state: ReadyEpisodeState
    reward_state: RewardEpisodeState
    trigger_requested: bool
    trigger_accepted: bool
    trigger_rejected: bool
    stage_at_reset: str
    randomization: EpisodeRandomizationParameters
    cylinder_reset_position_m: FloatArray
    target_tcp_position_m: FloatArray
    target_tcp_quaternion_wxyz: FloatArray
    reward_totals: dict[str, float] = field(default_factory=dict)
    last_step_reward: float = 0.0
    last_admission: AdmissionResult | None = None
    last_ready: ReadyResult | None = None
    last_outcome: OutcomeResult | None = None
    primary_termination_reason: str | None = None
    secondary_termination_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingReset:
    reset_spec: ResetSpec
    parameters: EpisodeRandomizationParameters


@dataclass(frozen=True, slots=True)
class TaskResetOverrides:
    """Authorized, persistent reset overrides for viewer/manifest workflows.

    Sampling still happens before replacement, so using a frozen case does not
    silently change any RNG stream position.  The authority-bearing M06/M11
    validators remain the only code allowed to accept the replacement values.
    """

    target_xy_m: tuple[float, float] | None = None
    cylinder: ForcedCylinderParameters | None = None
    target_authority: ForcedTargetAuthority | None = None
    cylinder_authority: ForcedParameterAuthority | None = None

    def __post_init__(self) -> None:
        if (self.target_xy_m is None) != (self.target_authority is None):
            raise ValueError("forced target XY and its authority must be supplied together")
        if (self.cylinder is None) != (self.cylinder_authority is None):
            raise ValueError("forced cylinder parameters and their authority must be supplied together")
        if (
            self.target_authority is not None
            and self.cylinder_authority is not None
            and self.target_authority.value != self.cylinder_authority.value
        ):
            raise ValueError("target and cylinder reset overrides must use the same workflow authority")

    @property
    def authority_label(self) -> str | None:
        if self.target_authority is not None:
            return self.target_authority.value
        if self.cylinder_authority is not None:
            return self.cylinder_authority.value
        return None


class ReachThenCloseEnvironment:
    """Simulator-independent state machine for one environment ID."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        env_id: int,
        curriculum: CurriculumManager,
        randomization_seed: int,
        actuator_ranges: ActuatorControlRanges,
        curriculum_updates_enabled: bool = True,
    ) -> None:
        config.validate()
        if env_id < 0:
            raise ValueError("env_id must be non-negative")
        if type(curriculum_updates_enabled) is not bool:
            raise TypeError("curriculum_updates_enabled must be bool")
        expected_actuators = tuple(f"{name}_position" for name in PHYSICAL_ACTION_NAMES)
        if actuator_ranges.actuator_names != expected_actuators or actuator_ranges.ranges_rad.shape != (13, 2):
            raise ValueError("actuator ranges must match the M07 7-arm then 6-hand physical action order")
        self.config = config
        self.env_id = env_id
        self.curriculum = curriculum
        self.randomizer = CylinderRandomizer(config.randomization, randomization_seed)
        self.actuator_ranges = actuator_ranges
        self.curriculum_updates_enabled = curriculum_updates_enabled
        self.arm_controller = ArmTargetController(config.task.action)
        self.option_controller: OptionController | None = None
        self.outcome_classifier = OutcomeClassifier(config.task.termination)
        self.episode: EpisodeState | None = None
        self.observation: Observation106 | None = None
        self._pending_reset: PendingReset | None = None
        self._prepared: PreparedPolicyStep | None = None
        self._object_to_target_tcp_position_m: FloatArray | None = None
        self._object_to_target_tcp_quaternion_wxyz: FloatArray | None = None
        self._grasp_center_offset_tcp_m: FloatArray | None = None

    @property
    def needs_lift_kinematics(self) -> bool:
        controller = self._require_option()
        return controller.state.phase is OptionPhase.Lift

    def make_reset(self, overrides: TaskResetOverrides | None = None) -> PendingReset:
        if self._pending_reset is not None:
            raise RuntimeError("the previous reset request has not been completed")
        selected = TaskResetOverrides() if overrides is None else overrides
        if not isinstance(selected, TaskResetOverrides):
            raise TypeError("reset overrides must be TaskResetOverrides or None")
        placement = self.curriculum.sample_target(
            forced_xy_m=selected.target_xy_m,
            authority=selected.target_authority,
        )
        sample = self.randomizer.sample(
            forced=selected.cylinder,
            authority=selected.cylinder_authority,
        )
        scene = self.config.task.scene
        cylinder_z = (
            scene.table_center_m[2]
            + 0.5 * scene.table_size_m[2]
            + scene.cylinder_table_clearance_m
            + 0.5 * sample.height_m
        )
        cylinder_position = (placement.target_xy_m[0], placement.target_xy_m[1], cylinder_z)
        parameters = EpisodeRandomizationParameters(
            stage_at_reset=placement.stage_at_reset,
            curriculum_component=placement.component_label,
            cylinder_position_m=cylinder_position,
            grasp_target_position_m=cylinder_position,
            cylinder=sample,
            rng_state_after=self.randomizer.state(),
            forced_authority=selected.authority_label,
            reset_sequence=(
                "curriculum_xy",
                "five_parameter_sample",
                "inertia_and_friction_application",
                "physics_reset",
                "transient_clear",
                "first_observation",
            ),
        )
        pending = PendingReset(
            ResetSpec(cylinder_position_m=cylinder_position, cylinder=sample.to_cylinder_parameters()),
            parameters,
        )
        self._pending_reset = pending
        return pending

    def complete_reset(
        self,
        state: PhysicsState,
        diagnostics: SimulationDiagnostics,
        contacts: ContactFrame | None = None,
    ) -> Observation106:
        pending = self._pending_reset
        if pending is None:
            raise RuntimeError("make_reset must precede complete_reset")
        contact_frame = _empty_contacts(self.config, state) if contacts is None else contacts
        if not _finite_physics(state, diagnostics):
            raise FloatingPointError("reset produced a non-finite physics state")
        if not np.allclose(
            state.cylinder_position_m,
            np.asarray(pending.parameters.cylinder_position_m, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError("physics reset did not apply the sampled cylinder position")
        if state.actuator_names != self.actuator_ranges.actuator_names:
            raise ValueError("PhysicsState actuator order differs from the M07 control-range contract")

        arm_positions = np.ascontiguousarray(state.joint_positions_rad[:7], dtype=np.float64)
        self.arm_controller.reset(arm_positions)
        pregrasp = np.ascontiguousarray(state.actuator_targets_rad[7:], dtype=np.float64)
        if self.option_controller is None:
            self.option_controller = OptionController(
                self.config.task.option,
                pregrasp,
                np.ascontiguousarray(self.actuator_ranges.lower_rad, dtype=np.float64),
                np.ascontiguousarray(self.actuator_ranges.upper_rad, dtype=np.float64),
            )
        physical_action = PhysicalAction(
            torch.from_numpy(np.array(state.actuator_targets_rad, dtype=np.float64, order="C", copy=True))
        )
        option_state = self.option_controller.reset(np.ascontiguousarray(state.actuator_targets_rad))
        self.outcome_classifier.reset()

        if self._object_to_target_tcp_position_m is None:
            nominal_height = 0.5 * (self.config.randomization.height.minimum + self.config.randomization.height.maximum)
            scene = self.config.task.scene
            nominal_cylinder = np.asarray(
                (
                    *scene.cylinder_origin_xy_m,
                    scene.table_center_m[2]
                    + 0.5 * scene.table_size_m[2]
                    + scene.cylinder_table_clearance_m
                    + 0.5 * nominal_height,
                ),
                dtype=np.float64,
            )
            target_site = state.site_index(_TARGET_TCP_SITE)
            target_world = state.site_positions_m[target_site]
            self._object_to_target_tcp_position_m = np.ascontiguousarray(
                target_world - nominal_cylinder,
                dtype=np.float64,
            )
            self._object_to_target_tcp_quaternion_wxyz = np.ascontiguousarray(
                state.site_quaternions_wxyz[target_site],
                dtype=np.float64,
            )
        if self._grasp_center_offset_tcp_m is None:
            self._grasp_center_offset_tcp_m = np.ascontiguousarray(
                _quat_rotate(
                    _quat_conjugate(state.tcp_quaternion_wxyz),
                    state.grasp_center_position_m - state.tcp_position_m,
                ),
                dtype=np.float64,
            )
        if self._object_to_target_tcp_quaternion_wxyz is None:
            raise RuntimeError("target TCP calibration is incomplete")
        target_tcp_position = np.ascontiguousarray(
            state.cylinder_position_m
            + _quat_rotate(
                state.cylinder_quaternion_wxyz,
                self._object_to_target_tcp_position_m,
            ),
            dtype=np.float64,
        )
        target_tcp_quaternion = _quat_multiply(
            state.cylinder_quaternion_wxyz,
            self._object_to_target_tcp_quaternion_wxyz,
        )
        self.episode = EpisodeState(
            episode_step=0,
            current_arm_targets_rad=arm_positions.copy(),
            previous_arm_targets_rad=arm_positions.copy(),
            physical_action=physical_action,
            previous_policy_arm_action=(0.0,) * 7,
            option_state=option_state,
            admission_state=AdmissionEpisodeState(),
            ready_state=ReadyEpisodeState(),
            reward_state=RewardEpisodeState(),
            trigger_requested=False,
            trigger_accepted=False,
            trigger_rejected=False,
            stage_at_reset=pending.parameters.stage_at_reset,
            randomization=pending.parameters,
            cylinder_reset_position_m=np.ascontiguousarray(state.cylinder_position_m, dtype=np.float64),
            target_tcp_position_m=target_tcp_position,
            target_tcp_quaternion_wxyz=target_tcp_quaternion,
            reward_totals={},
        )
        self._pending_reset = None
        self._prepared = None
        self.install_last_physics(state, contact_frame, diagnostics)
        self.observation = self._build_observation(state, contact_frame)
        return self.observation

    def _require_episode(self) -> EpisodeState:
        if self.episode is None:
            raise RuntimeError("environment must be reset before use")
        return self.episode

    def _require_option(self) -> OptionController:
        if self.option_controller is None:
            raise RuntimeError("environment must be reset before Option use")
        return self.option_controller

    def _observation_inputs(self, state: PhysicsState) -> ObservationInputs:
        episode = self._require_episode()
        option = self._require_option().state
        if self._grasp_center_offset_tcp_m is None:
            raise RuntimeError("grasp-center calibration is unavailable")
        grasp_index = state.site_index(_GRASP_CENTER_SITE)
        return ObservationInputs(
            environment_origin_m=np.zeros(3, dtype=np.float64),
            grasp_center_linear_velocity_m_s=np.ascontiguousarray(
                state.site_linear_velocities_m_s[grasp_index], dtype=np.float64
            ),
            target_tcp_position_m=episode.target_tcp_position_m,
            target_tcp_quaternion_wxyz=episode.target_tcp_quaternion_wxyz,
            grasp_center_offset_tcp_m=self._grasp_center_offset_tcp_m,
            option_active=float(option.script_active),
            option_progress=option.script_progress,
            option_support_progress=option.support_progress,
        )

    def _build_observation(self, state: PhysicsState, contacts: ContactFrame) -> Observation106:
        episode = self._require_episode()
        return build_observation(
            state=state,
            contacts=contacts,
            physical_action=episode.physical_action,
            inputs=self._observation_inputs(state),
            config=self.config.task.observation,
        )

    def _reduce_state(
        self,
        state: PhysicsState,
        contacts: ContactFrame,
        diagnostics: SimulationDiagnostics,
    ) -> _ReducedTaskState:
        episode = self._require_episode()
        if self._grasp_center_offset_tcp_m is None:
            raise RuntimeError("grasp-center calibration is unavailable")
        target_grasp = episode.target_tcp_position_m + _quat_rotate(
            episode.target_tcp_quaternion_wxyz,
            self._grasp_center_offset_tcp_m,
        )
        orientation_quaternion = _quat_multiply(
            _quat_conjugate(episode.target_tcp_quaternion_wxyz),
            state.tcp_quaternion_wxyz,
        )
        displacement = float(np.linalg.norm(state.cylinder_position_m - episode.cylinder_reset_position_m))
        tilt = _object_tilt_rad(state.cylinder_quaternion_wxyz)
        contact = _reduce_contacts(contacts)
        scene = self.config.task.scene
        table_top = scene.table_center_m[2] + 0.5 * scene.table_size_m[2]
        inherited = evaluate_inherited_termination_signals(
            InheritedTerminationInput(
                object_center_z_m=float(state.cylinder_position_m[2]),
                table_top_z_m=table_top,
                object_position_xy_in_environment_frame_m=(
                    float(state.cylinder_position_m[0]),
                    float(state.cylinder_position_m[1]),
                ),
                contact_force_by_group_n={
                    "thumb_side": contact.thumb_force_n,
                    "four_finger_opposing": contact.opposing_force_n,
                    "right_hand_observation_links": contact.net_hand_force_n,
                },
            ),
            self.config.task.termination,
        )
        grasp_index = state.site_index(_GRASP_CENTER_SITE)
        finite = _finite_physics(state, diagnostics)
        simulation_error = diagnostics.solver_anomaly or diagnostics.contact_capacity_exceeded
        position_error = float(np.linalg.norm(target_grasp - state.grasp_center_position_m))
        # This conservative finite proxy is zero only once the grasp center has
        # penetrated the randomized cylinder radius; it does not own contact classification.
        palm_clearance = max(
            0.0,
            float(np.linalg.norm(state.grasp_center_position_m - state.cylinder_position_m))
            - state.cylinder_parameters.radius_m,
        )
        return _ReducedTaskState(
            position_error_m=position_error,
            orientation_error_rad=_orientation_error_rad(episode.target_tcp_quaternion_wxyz, state.tcp_quaternion_wxyz),
            orientation_error_quaternion_wxyz=_tuple4(orientation_quaternion),
            grasp_center_speed_m_s=float(np.linalg.norm(state.site_linear_velocities_m_s[grasp_index])),
            arm_speed_max_rad_s=float(np.max(np.abs(state.joint_velocities_rad_s[:7]))),
            object_displacement_m=displacement,
            object_tilt_rad=tilt,
            object_linear_speed_m_s=float(np.linalg.norm(state.cylinder_linear_velocity_m_s)),
            object_angular_speed_rad_s=float(np.linalg.norm(state.cylinder_angular_velocity_rad_s)),
            illegal_contact_impulse_n_s=contacts.maximum_illegal_contact_impulse_n_s,
            palm_clearance_m=palm_clearance,
            object_lift_height_m=max(0.0, float(state.cylinder_position_m[2] - episode.cylinder_reset_position_m[2])),
            thumb_force_n=contact.thumb_force_n,
            opposing_force_n=contact.opposing_force_n,
            net_hand_force_n=contact.net_hand_force_n,
            dropped=inherited.dropped,
            object_far=inherited.object_far,
            toppled=tilt > self.config.task.termination.topple_angle_rad,
            excessive_displacement=displacement > self.config.task.termination.maximum_safe_displacement_m,
            illegal_contact=contacts.maximum_illegal_contact_impulse_n_s > 0.0,
            finite=finite,
            simulation_error=simulation_error,
        )

    def _reward_state(self, reduced: _ReducedTaskState, arm_action: tuple[float, ...]) -> RewardState:
        return RewardState(
            position_error_m=reduced.position_error_m,
            orientation_error_quaternion_wxyz=reduced.orientation_error_quaternion_wxyz,
            arm_joint_velocities_rad_s=tuple(
                float(item) for item in self._last_physics_state.joint_velocities_rad_s[:7]
            ),
            arm_action=arm_action,
            object_displacement_m=reduced.object_displacement_m,
        )

    _last_physics_state: PhysicsState
    _last_contacts: ContactFrame
    _last_diagnostics: SimulationDiagnostics

    def prepare_policy_step(
        self,
        action: torch.Tensor,
        *,
        lift_kinematics: tuple[FloatArray, FloatArray] | None = None,
    ) -> PreparedPolicyStep:
        if self._prepared is not None:
            raise RuntimeError("finish_policy_step must complete the previous policy step")
        episode = self._require_episode()
        arm_action = policy_arm_action(action)
        option = self._require_option()
        reach_transition = option.state.phase is OptionPhase.Reach
        episode.previous_arm_targets_rad = episode.current_arm_targets_rad.copy()
        if reach_transition:
            arm_targets = self.arm_controller.step(arm_action)
            episode.current_arm_targets_rad = arm_targets.copy()
        else:
            arm_targets = episode.current_arm_targets_rad.copy()
        preparation = None
        if self.needs_lift_kinematics:
            if lift_kinematics is None:
                raise ValueError("every Lift step requires the arm TCP Jacobian")
            arm_positions, tcp_jacobian = lift_kinematics
            preparation = LiftPreparation(
                measured_arm_positions_rad=tuple(float(value) for value in arm_positions),
                tcp_world_position_m=tuple(float(value) for value in self._last_physics_state.tcp_position_m),
                tcp_world_quaternion_wxyz=tuple(float(value) for value in self._last_physics_state.tcp_quaternion_wxyz),
                tcp_jacobian_6x7=tuple(tuple(float(value) for value in row) for row in tcp_jacobian),
            )
        physical_action = option.begin_policy_step(arm_targets, lift_preparation=preparation)
        episode.physical_action = physical_action
        policy_arm = tuple(float(item) for item in arm_action.values.numpy())
        current_reduced = self._reduce_state(
            self._last_physics_state,
            self._last_contacts,
            self._last_diagnostics,
        )
        prepared = PreparedPolicyStep(
            physical_action=physical_action,
            reach_transition=reach_transition,
            actor_update_mask=reach_transition,
            critic_update_mask=True,
            discount_factor=self.config.ppo.gamma if reach_transition else 1.0,
            gae_lambda_factor=self.config.ppo.gae_lambda if reach_transition else 1.0,
            policy_arm_action=policy_arm,
            before_reward_state=self._reward_state(current_reduced, episode.previous_policy_arm_action),
        )
        self._prepared = prepared
        return prepared

    def finish_policy_step(self, result: ParallelStepResult) -> TaskStepTransition:
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("prepare_policy_step must precede finish_policy_step")
        if result.env_id != self.env_id:
            raise ValueError("parallel result env_id differs from task environment")
        if result.substeps_executed != self.config.physics.decimation or result.substeps_executed != 4:
            raise ValueError("one policy step must execute exactly four MuJoCo substeps")
        episode = self._require_episode()
        episode.episode_step += 1
        reduced = self._reduce_state(result.state, result.contacts, result.diagnostics)
        admission_input = AdmissionInput(
            state_finite=reduced.finite,
            target_valid=True,
            reach_phase=prepared.reach_transition,
            position_error_m=reduced.position_error_m,
            orientation_error_rad=reduced.orientation_error_rad,
            grasp_center_speed_m_s=reduced.grasp_center_speed_m_s,
            arm_speed_max_rad_s=reduced.arm_speed_max_rad_s,
            object_displacement_m=reduced.object_displacement_m,
            object_tilt_rad=reduced.object_tilt_rad,
            object_linear_speed_m_s=reduced.object_linear_speed_m_s,
            object_angular_speed_rad_s=reduced.object_angular_speed_rad_s,
            illegal_contact_impulse_n_s=reduced.illegal_contact_impulse_n_s,
        )
        admission, admission_state = step_admission(
            admission_input,
            episode.admission_state,
            self.config.task.admission,
        )
        episode.admission_state = admission_state
        ready, ready_state = step_ready(
            ReadyInput(
                state_finite=reduced.finite,
                target_valid=True,
                position_error_m=reduced.position_error_m,
                orientation_error_rad=reduced.orientation_error_rad,
                grasp_center_speed_m_s=reduced.grasp_center_speed_m_s,
                arm_speed_max_rad_s=reduced.arm_speed_max_rad_s,
                object_displacement_m=reduced.object_displacement_m,
                object_tilt_rad=reduced.object_tilt_rad,
                object_linear_speed_m_s=reduced.object_linear_speed_m_s,
                object_angular_speed_rad_s=reduced.object_angular_speed_rad_s,
                illegal_contact_impulse_n_s=reduced.illegal_contact_impulse_n_s,
            ),
            episode.ready_state,
            self.config.task.ready,
        )
        episode.ready_state = ready_state
        if ready.newly_ready:
            self.curriculum.record_strict_ready()

        option = self._require_option()
        scene = self.config.task.scene
        table_top = scene.table_center_m[2] + 0.5 * scene.table_size_m[2]
        trace = result.substep_trace
        transient_toppled = False
        transient_dropped = False
        transient_illegal_contact = False
        transient_nonfinite = False
        for substep_index in range(result.substeps_executed):
            position = trace.cylinder_positions_m[substep_index]
            quaternion = trace.cylinder_quaternions_wxyz[substep_index]
            contact = _reduce_contacts(result.contacts, substep_index)
            sample_illegal = any(
                sample.substep_index == substep_index
                and is_illegal_contact(
                    sample.geom1_groups,
                    sample.geom2_groups,
                    float(np.linalg.norm(sample.force_on_geom2_world_n)),
                    right_arm_table_force_limit_n=(self.config.task.termination.right_arm_table_contact_force_limit_n),
                )
                for sample in result.contacts.contacts
            )
            step_toppled = _object_tilt_rad(quaternion) > self.config.task.termination.topple_angle_rad
            step_dropped = float(position[2]) < table_top - self.config.task.termination.drop_below_table_top_m
            step_nonfinite = not bool(trace.finite[substep_index]) or reduced.simulation_error
            transient_toppled |= step_toppled
            transient_dropped |= step_dropped
            transient_illegal_contact |= sample_illegal
            transient_nonfinite |= step_nonfinite
            option.observe_substep(
                OptionSubstepObservation(
                    thumb_side_force_n=contact.thumb_force_n,
                    opposing_side_force_n=contact.opposing_force_n,
                    net_hand_force_n=contact.net_hand_force_n,
                    object_lift_height_m=max(
                        0.0,
                        float(position[2] - episode.cylinder_reset_position_m[2]),
                    ),
                    toppled=step_toppled,
                    dropped=step_dropped,
                    illegal_contact=sample_illegal,
                    state_finite=not step_nonfinite,
                )
            )
        option_state = option.finish_policy_step(
            trigger_requested=prepared.reach_transition and admission.newly_admissible,
            option_admissible=admission.option_admissible,
        )
        timed_out = episode.episode_step >= self.config.task.scene.maximum_episode_steps
        outcome = self.outcome_classifier.step(
            OutcomeInput(
                step_index=episode.episode_step - 1,
                option_accepted=option_state.acceptance_latched,
                object_lift_height_m=reduced.object_lift_height_m,
                completed_lift_hold=option_state.phase is OptionPhase.Success,
                thumb_contact_force_n=reduced.thumb_force_n,
                opposing_contact_force_n=reduced.opposing_force_n,
                lift_support_force_n=reduced.net_hand_force_n,
                toppled=reduced.toppled or transient_toppled,
                dropped=reduced.dropped or transient_dropped,
                object_far=reduced.object_far,
                excessive_displacement=False,
                illegal_contact=reduced.illegal_contact or transient_illegal_contact,
                state_finite=reduced.finite and not transient_nonfinite,
                simulation_error=reduced.simulation_error,
                script_failed=option_state.phase is OptionPhase.Failure,
                timed_out=timed_out,
            )
        )
        time_limit_bootstrap = outcome.primary_reason is TerminationReason.TIMEOUT
        reward = compute_reward(
            prepared.before_reward_state,
            self._reward_state(reduced, prepared.policy_arm_action),
            RewardEvents(
                reach_phase=prepared.reach_transition,
                option_admissible=admission.option_admissible,
                trigger_requested=option_state.trigger_requested,
                trigger_accepted=option_state.trigger_accepted,
                trigger_rejected=option_state.trigger_rejected,
                lift_5cm_event=outcome.lift_5cm_event,
                complete_success=outcome.success,
                script_failure=(
                    outcome.primary_reason is not None and option_state.acceptance_latched and not outcome.success
                ),
                strict_ready=ready.ready,
                newly_strict_ready=ready.newly_ready,
                strict_ready_ever=ready.ever_ready,
            ),
            episode.reward_state,
            self.config.task.reward,
        )
        episode.reward_state = reward.episode_state
        episode.option_state = option_state
        episode.trigger_requested = option_state.trigger_requested
        episode.trigger_accepted = option_state.trigger_accepted
        episode.trigger_rejected = option_state.trigger_rejected
        episode.previous_policy_arm_action = prepared.policy_arm_action
        episode.last_admission = admission
        episode.last_ready = ready
        episode.last_outcome = outcome
        episode.primary_termination_reason = outcome.primary_reason_name
        episode.secondary_termination_reasons = outcome.secondary_reason_names
        for name, value in reward.breakdown.terms.items():
            episode.reward_totals[name] = episode.reward_totals.get(name, 0.0) + value
        episode.last_step_reward = reward.breakdown.total
        self._last_physics_state = result.state
        self._last_contacts = result.contacts
        self._last_diagnostics = result.diagnostics
        terminal_observation = self._build_observation(result.state, result.contacts)
        self.observation = terminal_observation
        snapshot = self.snapshot()
        curriculum_update = None
        if outcome.done and self.curriculum_updates_enabled:
            curriculum_update = self.curriculum.record_completed_episode(
                EpisodeCurriculumResult(
                    stage_at_reset=episode.stage_at_reset,
                    option_accepted=option_state.acceptance_latched,
                    lifted_5cm=outcome.lift_5cm,
                    complete_success=outcome.success,
                    dangerous=outcome.unsafe,
                )
            )
        self._prepared = None
        return TaskStepTransition(
            env_id=self.env_id,
            terminal_observation=terminal_observation,
            reward=reward.breakdown.total,
            done=outcome.done,
            timed_out=time_limit_bootstrap,
            actor_update_mask=prepared.actor_update_mask,
            critic_update_mask=prepared.critic_update_mask,
            discount_factor=prepared.discount_factor,
            gae_lambda_factor=prepared.gae_lambda_factor,
            admission=admission,
            ready=ready,
            outcome=outcome,
            reward_breakdown=reward.breakdown,
            snapshot=snapshot,
            curriculum_update=curriculum_update,
        )

    def install_last_physics(
        self,
        state: PhysicsState,
        contacts: ContactFrame,
        diagnostics: SimulationDiagnostics,
    ) -> None:
        """Install the latest immutable simulator frame after reset."""

        self._last_physics_state = state
        self._last_contacts = contacts
        self._last_diagnostics = diagnostics

    def snapshot(self) -> TaskSnapshot:
        episode = self._require_episode()
        option = self._require_option().state
        outcome = episode.last_outcome
        ready = episode.last_ready
        admission = episode.last_admission
        reduced = self._reduce_state(
            self._last_physics_state,
            self._last_contacts,
            self._last_diagnostics,
        )
        randomization = _deep_freeze(episode.randomization.to_dict())
        if not isinstance(randomization, Mapping):  # pragma: no cover - constructed from a mapping
            raise RuntimeError("frozen randomization snapshot is not a mapping")
        return TaskSnapshot(
            schema_version=TASK_SNAPSHOT_SCHEMA_VERSION,
            env_id=self.env_id,
            episode_step=episode.episode_step,
            stage_at_reset=episode.stage_at_reset,
            target_tcp_position_m=_tuple3(episode.target_tcp_position_m),
            target_tcp_quaternion_wxyz=_tuple4(episode.target_tcp_quaternion_wxyz),
            cylinder_reset_position_m=_tuple3(episode.cylinder_reset_position_m),
            option_phase=option.phase.value,
            option_phase_step=option.phase_step,
            option_script_steps=option.script_steps_completed,
            option_script_progress=option.script_progress,
            option_support_progress=option.support_progress,
            trigger_requested=episode.trigger_requested,
            trigger_accepted=episode.trigger_accepted,
            trigger_rejected=episode.trigger_rejected,
            option_accepted_ever=option.acceptance_latched,
            strict_ready=False if ready is None else ready.ready,
            strict_ready_ever=episode.ready_state.ever_ready,
            strict_ready_consecutive_steps=episode.ready_state.consecutive_steps,
            option_admissible=False if admission is None else admission.option_admissible,
            admission_instantaneous=False if admission is None else admission.instantaneous_admissible,
            admission_consecutive_steps=0 if admission is None else admission.consecutive_steps,
            admission_dwell_steps_required=(
                round(self.config.task.admission.dwell_time_s / self.config.physics.policy_dt_s)
                if admission is None
                else admission.dwell_steps_required
            ),
            position_error_m=reduced.position_error_m,
            orientation_error_rad=reduced.orientation_error_rad,
            object_displacement_m=reduced.object_displacement_m,
            object_tilt_rad=reduced.object_tilt_rad,
            illegal_contact_impulse_n_s=reduced.illegal_contact_impulse_n_s,
            step_reward=episode.last_step_reward,
            episode_reward_total=math.fsum(episode.reward_totals.values()),
            lift_5cm=False if outcome is None else outcome.lift_5cm,
            success=False if outcome is None else outcome.success,
            unsafe=False if outcome is None else outcome.unsafe,
            done=False if outcome is None else outcome.done,
            primary_termination_reason=episode.primary_termination_reason,
            secondary_termination_reasons=episode.secondary_termination_reasons,
            admission_failures=() if admission is None else admission.failure_reasons,
            reward_terms=MappingProxyType(dict(episode.reward_totals)),
            randomization=randomization,
            simulation_warning_names=tuple(item.name for item in self._last_diagnostics.warnings),
            contact_count=len(self._last_contacts.contacts),
            maximum_contact_force_n=self._last_contacts.maximum_contact_force_n,
            maximum_contact_impulse_n_s=self._last_contacts.maximum_contact_impulse_n_s,
            maximum_contacts_seen=self._last_contacts.maximum_contacts_seen,
            contact_capacity=self._last_contacts.contact_capacity,
            contact_capacity_exceeded=self._last_diagnostics.contact_capacity_exceeded,
            solver_anomaly=self._last_diagnostics.solver_anomaly,
            nonfinite_state=self._last_diagnostics.nonfinite_state,
            physics_step=self._last_diagnostics.physics_step,
            simulation_time_s=self._last_diagnostics.simulation_time_s,
        )


@dataclass(frozen=True, slots=True)
class DirectTaskStepResult:
    observation: Observation106
    transition: TaskStepTransition


class ReachThenCloseTask:
    """Single-environment direct mode used by viewer and parity tests."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        backend: EnvironmentBackend | None = None,
        seed: int | None = None,
        reset_overrides: TaskResetOverrides | None = None,
        curriculum_updates_enabled: bool = True,
    ) -> None:
        self.config = config
        self.backend = MujocoEnvironmentBackend(config) if backend is None else backend
        root_seed = config.run.seed if seed is None else seed
        self.curriculum = CurriculumManager(config, root_seed)
        self.environment = ReachThenCloseEnvironment(
            config,
            env_id=0,
            curriculum=self.curriculum,
            randomization_seed=root_seed,
            actuator_ranges=self.backend.read_actuator_control_ranges(),
            curriculum_updates_enabled=curriculum_updates_enabled,
        )
        self._owns_backend = backend is None
        if reset_overrides is not None and not isinstance(reset_overrides, TaskResetOverrides):
            raise TypeError("reset_overrides must be TaskResetOverrides or None")
        self.reset_overrides = reset_overrides

    def reset(self) -> Observation106:
        pending = self.environment.make_reset(self.reset_overrides)
        state = self.backend.reset(pending.reset_spec)
        contacts = self.backend.read_contacts()
        diagnostics = self.backend.read_warnings()
        observation = self.environment.complete_reset(state, diagnostics, contacts)
        self.environment.install_last_physics(state, contacts, diagnostics)
        return observation

    def step(self, action: torch.Tensor, *, auto_reset: bool = True) -> DirectTaskStepResult:
        if type(auto_reset) is not bool:
            raise TypeError("auto_reset must be bool")
        lift = None
        if self.environment.needs_lift_kinematics:
            lift = self.backend.read_arm_site_jacobian(_TCP_SITE)
        prepared = self.environment.prepare_policy_step(action, lift_kinematics=lift)
        self.backend.apply_targets(np.ascontiguousarray(prepared.physical_action.values.numpy()))
        physics = self.backend.step_substeps(4)
        transition = self.environment.finish_policy_step(
            ParallelStepResult(
                env_id=0,
                state=physics.state,
                contacts=physics.contacts,
                diagnostics=physics.diagnostics,
                substeps_executed=physics.substeps_executed,
                physics_step=physics.diagnostics.physics_step,
                substep_trace=physics.substep_trace,
            )
        )
        observation = transition.terminal_observation
        if transition.done and auto_reset:
            observation = self.reset()
        return DirectTaskStepResult(observation=observation, transition=transition)

    def get_observation(self) -> Observation106:
        if self.environment.observation is None:
            return self.reset()
        return self.environment.observation

    def reseed_reset_streams(self, seed: int) -> None:
        """Install fresh curriculum/randomization streams between evaluation cases."""

        if type(seed) is not int or seed < 0:
            raise ValueError("reset stream seed must be a non-negative integer")
        environment = self.environment
        if environment._pending_reset is not None or environment._prepared is not None:
            raise RuntimeError("reset streams cannot be reseeded during a reset or policy step")
        episode = environment.episode
        if episode is not None and (episode.last_outcome is None or not episode.last_outcome.done):
            raise RuntimeError("reset streams can only be reseeded before the first case or after a terminal case")
        curriculum = CurriculumManager(self.config, seed)
        environment.curriculum = curriculum
        environment.randomizer = CylinderRandomizer(self.config.randomization, seed)
        self.curriculum = curriculum

    def snapshot(self) -> TaskSnapshot:
        return self.environment.snapshot()

    def close(self) -> None:
        if self._owns_backend:
            self.backend.close()

    def __enter__(self) -> ReachThenCloseTask:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def ordered_snapshots(
    environments: list[ReachThenCloseEnvironment] | tuple[ReachThenCloseEnvironment, ...],
) -> tuple[TaskSnapshot, ...]:
    """Return immutable snapshots in stable environment-ID order."""

    return tuple(item.snapshot() for item in sorted(environments, key=lambda item: item.env_id))


__all__ = [
    "TASK_EXTRAS_SCHEMA_VERSION",
    "TASK_SNAPSHOT_SCHEMA_VERSION",
    "DirectTaskStepResult",
    "EpisodeState",
    "PendingReset",
    "PreparedPolicyStep",
    "ReachThenCloseEnvironment",
    "ReachThenCloseTask",
    "TaskResetOverrides",
    "TaskSnapshot",
    "TaskStepTransition",
    "ordered_snapshots",
]
