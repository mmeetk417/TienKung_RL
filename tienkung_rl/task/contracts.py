"""Versioned action and observation contracts for ``reach_then_close``.

The task boundary uses CPU tensors deliberately: policy-facing values are
``float32`` while the 13 actuator targets are ``float64`` to match MuJoCo's
native control buffer without an implicit precision conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACTION_CONTRACT_VERSION = "reach_action_runtime"
OBSERVATION_CONTRACT_VERSION = "reach_observation_runtime"

ARM_POLICY_DIMENSION = 7
POLICY_ACTION_DIMENSION = 7
ACTIVE_HAND_DIMENSION = 6
PHYSICAL_ACTION_DIMENSION = 13
OBSERVATION_DIMENSION = 106
OBSERVATION_HISTORY_LENGTH = 1
MAXIMUM_ARM_TARGET_STEP_RAD = 0.01

ARM_JOINT_NAMES = (
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)

HAND_OBSERVATION_JOINT_NAMES = (
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

HAND_CONTACT_LINK_NAMES = tuple(name.removesuffix("_joint") for name in HAND_OBSERVATION_JOINT_NAMES)

OBSERVED_JOINT_NAMES = ARM_JOINT_NAMES + HAND_OBSERVATION_JOINT_NAMES

ACTIVE_HAND_JOINT_NAMES = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)

POLICY_ACTION_NAMES = ARM_JOINT_NAMES
PHYSICAL_ACTION_NAMES = ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES


@dataclass(frozen=True, slots=True)
class ObservationBlock:
    """One ordered field in the 106-dimensional observation vector."""

    name: str
    dimension: int
    scale: float
    frame: str
    quaternion_order: str = "none"
    canonicalize_scalar_nonnegative: bool = False


OBSERVATION_BLOCKS = (
    ObservationBlock("observed_joint_positions", 19, 1.0, "joint"),
    ObservationBlock("scaled_observed_joint_velocities", 19, 0.10, "joint"),
    ObservationBlock("grasp_center_world_position", 3, 1.0, "environment_world"),
    ObservationBlock("tcp_world_quaternion", 4, 1.0, "environment_world", "wxyz"),
    ObservationBlock("cylinder_world_position", 3, 1.0, "environment_world"),
    ObservationBlock("cylinder_world_quaternion", 4, 1.0, "environment_world", "wxyz"),
    ObservationBlock("cylinder_to_grasp_center_position", 3, 1.0, "environment_world"),
    ObservationBlock("scaled_cylinder_linear_velocity", 3, 0.25, "environment_world"),
    ObservationBlock("scaled_cylinder_angular_velocity", 3, 0.25, "environment_world"),
    ObservationBlock("right_hand_contact_force_norms", 12, 0.10, "contact"),
    ObservationBlock("internal_physical_action_state", 13, 1.0, "actuator_target"),
    ObservationBlock("cylinder_relative_quaternion_in_tcp", 4, 1.0, "tcp", "wxyz"),
    ObservationBlock("scaled_cylinder_relative_linear_velocity", 3, 0.25, "grasp_center"),
    ObservationBlock("option_active_progress_support", 3, 1.0, "option"),
    ObservationBlock("grasp_position_error_in_cylinder", 3, 1.0, "cylinder"),
    ObservationBlock(
        "current_tcp_relative_target_quaternion",
        4,
        1.0,
        "target_tcp",
        "wxyz",
        True,
    ),
    ObservationBlock("approach_direction_error_in_cylinder", 3, 1.0, "cylinder"),
)

OBSERVATION_BLOCK_SLICES: dict[str, slice] = {}
_block_offset = 0
for _block in OBSERVATION_BLOCKS:
    OBSERVATION_BLOCK_SLICES[_block.name] = slice(_block_offset, _block_offset + _block.dimension)
    _block_offset += _block.dimension
if _block_offset != OBSERVATION_DIMENSION:  # pragma: no cover - import-time invariant
    raise RuntimeError("observation block dimensions do not sum to 106")


def _checked_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU, got {value.device}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value.detach().clone().contiguous()


@dataclass(frozen=True, slots=True)
class ArmPolicyAction:
    values: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _checked_tensor(
                self.values,
                shape=(ARM_POLICY_DIMENSION,),
                dtype=torch.float32,
                name="ArmPolicyAction.values",
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    values: torch.Tensor

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            _checked_tensor(
                self.values,
                shape=(PHYSICAL_ACTION_DIMENSION,),
                dtype=torch.float64,
                name="PhysicalAction.values",
            ),
        )

    @property
    def arm_targets_rad(self) -> torch.Tensor:
        return self.values[:ARM_POLICY_DIMENSION].clone()

    @property
    def hand_targets_rad(self) -> torch.Tensor:
        return self.values[ARM_POLICY_DIMENSION:].clone()


@dataclass(frozen=True, slots=True)
class Observation106:
    values: torch.Tensor
    history_length: int = OBSERVATION_HISTORY_LENGTH

    def __post_init__(self) -> None:
        if self.history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError("Observation106 history_length must be 1")
        object.__setattr__(
            self,
            "values",
            _checked_tensor(
                self.values,
                shape=(OBSERVATION_DIMENSION,),
                dtype=torch.float32,
                name="Observation106.values",
            ),
        )


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTIVE_HAND_DIMENSION",
    "ACTIVE_HAND_JOINT_NAMES",
    "ARM_JOINT_NAMES",
    "ARM_POLICY_DIMENSION",
    "HAND_CONTACT_LINK_NAMES",
    "HAND_OBSERVATION_JOINT_NAMES",
    "MAXIMUM_ARM_TARGET_STEP_RAD",
    "OBSERVATION_BLOCKS",
    "OBSERVATION_BLOCK_SLICES",
    "OBSERVATION_CONTRACT_VERSION",
    "OBSERVATION_DIMENSION",
    "OBSERVATION_HISTORY_LENGTH",
    "OBSERVED_JOINT_NAMES",
    "PHYSICAL_ACTION_DIMENSION",
    "PHYSICAL_ACTION_NAMES",
    "POLICY_ACTION_DIMENSION",
    "POLICY_ACTION_NAMES",
    "ArmPolicyAction",
    "Observation106",
    "ObservationBlock",
    "PhysicalAction",
]
