"""Strict conversion from the 7-D arm policy action to MuJoCo targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from tienkung_rl.config.models import ActionConfig

from .contracts import (
    ACTIVE_HAND_DIMENSION,
    ARM_JOINT_NAMES,
    ARM_POLICY_DIMENSION,
    MAXIMUM_ARM_TARGET_STEP_RAD,
    POLICY_ACTION_DIMENSION,
    ArmPolicyAction,
    PhysicalAction,
)

FloatArray = NDArray[np.float64]


def policy_arm_action(action: torch.Tensor) -> ArmPolicyAction:
    """Validate and wrap exactly seven normalized arm actions."""

    if not isinstance(action, torch.Tensor):
        raise TypeError("policy action must be a torch.Tensor")
    if action.device.type != "cpu":
        raise ValueError(f"policy action must be on CPU, got {action.device}")
    if action.dtype != torch.float32:
        raise TypeError(f"policy action must have dtype torch.float32, got {action.dtype}")
    if tuple(action.shape) != (POLICY_ACTION_DIMENSION,):
        raise ValueError(f"policy action must have shape (7,), got {tuple(action.shape)}")
    if not bool(torch.isfinite(action).all()):
        raise ValueError("policy action contains a non-finite value")
    return ArmPolicyAction(action)


def normalized_arm_targets(action: ArmPolicyAction, config: ActionConfig) -> FloatArray:
    """Map normalized absolute actions with ``center + action * half_range``."""

    config.validate()
    if config.arm_joint_names != ARM_JOINT_NAMES:
        raise ValueError("action config arm joint order does not match the M07 contract")
    values = action.values.numpy().astype(np.float64, copy=False)
    below = np.flatnonzero(values < config.normalized_min)
    above = np.flatnonzero(values > config.normalized_max)
    if below.size or above.size:
        invalid = np.concatenate((below, above)).tolist()
        raise ValueError(f"normalized arm action lies outside [-1, 1] at indices {invalid}")
    center = np.asarray(config.center_joint_positions_rad, dtype=np.float64)
    half_range = np.asarray(config.half_range_rad, dtype=np.float64)
    return np.ascontiguousarray(center + values * half_range, dtype=np.float64)


def rate_limit_arm_targets(
    desired_targets_rad: FloatArray,
    previous_targets_rad: FloatArray,
    maximum_step_rad: float = MAXIMUM_ARM_TARGET_STEP_RAD,
) -> FloatArray:
    """Limit every signed target delta independently to 0.01 rad."""

    desired = _strict_float64_vector(desired_targets_rad, ARM_POLICY_DIMENSION, "desired_targets_rad")
    previous = _strict_float64_vector(previous_targets_rad, ARM_POLICY_DIMENSION, "previous_targets_rad")
    if maximum_step_rad != MAXIMUM_ARM_TARGET_STEP_RAD:
        raise ValueError("maximum arm target step must be exactly 0.01 rad")
    delta = np.clip(desired - previous, -maximum_step_rad, maximum_step_rad)
    return np.ascontiguousarray(previous + delta, dtype=np.float64)


def assemble_reach_physical_action(
    arm_targets_rad: FloatArray,
    pregrasp_hand_targets_rad: FloatArray,
) -> PhysicalAction:
    """Build the actuator-only Reach action; no trigger argument exists here."""

    arm = _strict_float64_vector(arm_targets_rad, ARM_POLICY_DIMENSION, "arm_targets_rad")
    hand = _strict_float64_vector(
        pregrasp_hand_targets_rad,
        ACTIVE_HAND_DIMENSION,
        "pregrasp_hand_targets_rad",
    )
    return PhysicalAction(torch.from_numpy(np.concatenate((arm, hand))))


def _strict_float64_vector(value: object, size: int, name: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {value.dtype}")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains a non-finite value")
    return value


@dataclass(slots=True)
class ArmTargetController:
    """Stateful 0.01-rad limiter with an explicit measured-state reset."""

    config: ActionConfig
    _previous_targets_rad: FloatArray | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        if self.config.arm_joint_names != ARM_JOINT_NAMES:
            raise ValueError("action config arm joint order does not match the M07 contract")

    def reset(self, measured_arm_positions_rad: FloatArray) -> None:
        measured = _strict_float64_vector(
            measured_arm_positions_rad,
            ARM_POLICY_DIMENSION,
            "measured_arm_positions_rad",
        )
        self._previous_targets_rad = np.array(measured, dtype=np.float64, order="C", copy=True)

    def step(self, action: ArmPolicyAction) -> FloatArray:
        if self._previous_targets_rad is None:
            raise RuntimeError("ArmTargetController must be reset from measured joint positions before step")
        desired = normalized_arm_targets(action, self.config)
        limited = rate_limit_arm_targets(
            desired,
            self._previous_targets_rad,
            self.config.maximum_joint_target_step_rad,
        )
        self._previous_targets_rad = limited.copy()
        return limited


__all__ = [
    "ArmTargetController",
    "assemble_reach_physical_action",
    "normalized_arm_targets",
    "policy_arm_action",
    "rate_limit_arm_targets",
]
