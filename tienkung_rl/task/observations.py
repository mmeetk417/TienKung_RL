"""Pure construction of the ordered, clipped 106-D policy observation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from tienkung_rl.config.models import ObservationConfig
from tienkung_rl.sim.state import ContactFrame, PhysicsState

from .contracts import (
    HAND_CONTACT_LINK_NAMES,
    OBSERVATION_BLOCKS,
    OBSERVATION_DIMENSION,
    OBSERVED_JOINT_NAMES,
    Observation106,
    PhysicalAction,
)

FloatArray = NDArray[np.float64]


class ObservationNonFiniteError(RuntimeError):
    """Fail-closed signal consumed as a safe task termination reason."""

    def __init__(self, field_name: str, phase: str, indices: tuple[int, ...]) -> None:
        self.field_name = field_name
        self.phase = phase
        self.indices = indices
        self.safe_termination_reason = f"nonfinite_observation:{field_name}"
        super().__init__(f"non-finite observation field {field_name!r} during {phase}; indices={indices}")


@dataclass(frozen=True, slots=True)
class ObservationInputs:
    """Task values not owned by ``PhysicsState`` or ``ContactFrame``.

    Target geometry is represented by the authoritative target TCP pose and
    the calibrated TCP-to-grasp-center offset, matching the extracted Isaac
    implementation.  All vectors are world-frame unless their name says TCP.
    """

    environment_origin_m: FloatArray
    grasp_center_linear_velocity_m_s: FloatArray
    target_tcp_position_m: FloatArray
    target_tcp_quaternion_wxyz: FloatArray
    grasp_center_offset_tcp_m: FloatArray
    option_active: float
    option_progress: float
    option_support_progress: float

    def __post_init__(self) -> None:
        for field_name in (
            "environment_origin_m",
            "grasp_center_linear_velocity_m_s",
            "target_tcp_position_m",
            "grasp_center_offset_tcp_m",
        ):
            object.__setattr__(
                self,
                field_name,
                _shape_checked_float64(getattr(self, field_name), (3,), field_name),
            )
        object.__setattr__(
            self,
            "target_tcp_quaternion_wxyz",
            _shape_checked_float64(
                self.target_tcp_quaternion_wxyz,
                (4,),
                "target_tcp_quaternion_wxyz",
            ),
        )


def _shape_checked_float64(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.dtype != np.float64:
        raise TypeError(f"{name} must have dtype float64, got {value.dtype}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    return np.array(value, dtype=np.float64, order="C", copy=True)


def _require_finite(name: str, value: FloatArray, phase: str = "precheck") -> None:
    invalid = np.flatnonzero(~np.isfinite(value))
    if invalid.size:
        raise ObservationNonFiniteError(name, phase, tuple(int(index) for index in invalid))


def _normalized_quaternion(value: FloatArray, name: str) -> FloatArray:
    _require_finite(name, value)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} has zero norm")
    return np.asarray(value / norm, dtype=np.float64)


def _quaternion_conjugate(quaternion: FloatArray) -> FloatArray:
    return np.asarray(
        (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]),
        dtype=np.float64,
    )


def _quaternion_multiply(left: FloatArray, right: FloatArray) -> FloatArray:
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
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion product has zero norm")
    return np.asarray(result / norm, dtype=np.float64)


def _quaternion_rotate(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    # Algebraically equivalent to q * [0, v] * conjugate(q), without
    # normalizing the pure-vector intermediate quaternion.
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return np.asarray(
        2.0 * np.dot(imaginary, vector) * imaginary
        + (scalar * scalar - np.dot(imaginary, imaginary)) * vector
        + 2.0 * scalar * np.cross(imaginary, vector),
        dtype=np.float64,
    )


def _canonicalize_scalar_nonnegative(quaternion: FloatArray) -> FloatArray:
    result = np.array(quaternion, dtype=np.float64, copy=True)
    if result[0] < 0.0:
        result *= -1.0
    # Make the zero boundary deterministic as well (avoid a persisted -0.0).
    result[result == 0.0] = 0.0
    return result


def _validate_config(config: ObservationConfig) -> None:
    config.validate()
    if config.observed_joint_names != OBSERVED_JOINT_NAMES:
        raise ValueError("observation config joint order does not match the M07 contract")
    configured = tuple(
        (
            field.name,
            field.dimension,
            field.scale,
            field.frame,
            field.quaternion_order,
            field.canonicalize_scalar_nonnegative,
        )
        for field in config.fields
    )
    frozen = tuple(
        (
            block.name,
            block.dimension,
            block.scale,
            block.frame,
            block.quaternion_order,
            block.canonicalize_scalar_nonnegative,
        )
        for block in OBSERVATION_BLOCKS
    )
    if configured != frozen:
        raise ValueError("observation config fields do not match the ordered M07 block contract")


def build_observation_blocks(
    *,
    state: PhysicsState,
    contacts: ContactFrame,
    physical_action: PhysicalAction,
    inputs: ObservationInputs,
    config: ObservationConfig,
) -> tuple[tuple[str, FloatArray], ...]:
    """Return all 17 un-clipped blocks in their only valid order."""

    _validate_config(config)
    for field_name in (
        "environment_origin_m",
        "grasp_center_linear_velocity_m_s",
        "target_tcp_position_m",
        "target_tcp_quaternion_wxyz",
        "grasp_center_offset_tcp_m",
    ):
        _require_finite(field_name, getattr(inputs, field_name))
    progress = np.asarray(
        (inputs.option_active, inputs.option_progress, inputs.option_support_progress),
        dtype=np.float64,
    )
    _require_finite("option_active_progress_support", progress)
    if inputs.option_active not in (0.0, 1.0):
        raise ValueError("option_active must be exactly 0 or 1")
    if np.any(progress[1:] < 0.0) or np.any(progress[1:] > 1.0):
        raise ValueError("option progress values must lie in [0, 1]")

    joint_indices = tuple(state.joint_index(name) for name in OBSERVED_JOINT_NAMES)
    joint_positions = np.asarray(state.joint_positions_rad[list(joint_indices)], dtype=np.float64)
    joint_velocities = np.asarray(state.joint_velocities_rad_s[list(joint_indices)], dtype=np.float64)
    if contacts.hand_link_names != HAND_CONTACT_LINK_NAMES:
        raise ValueError("ContactFrame link order does not match the 12-link M07 contract")

    tcp_quaternion = _normalized_quaternion(state.tcp_quaternion_wxyz, "tcp_world_quaternion")
    cylinder_quaternion = _normalized_quaternion(
        state.cylinder_quaternion_wxyz,
        "cylinder_world_quaternion",
    )
    target_tcp_quaternion = _normalized_quaternion(
        inputs.target_tcp_quaternion_wxyz,
        "target_tcp_quaternion_wxyz",
    )
    inverse_tcp = _quaternion_conjugate(tcp_quaternion)
    inverse_cylinder = _quaternion_conjugate(cylinder_quaternion)

    grasp_position = state.grasp_center_position_m
    tcp_position = state.tcp_position_m
    cylinder_position = state.cylinder_position_m
    offset_norm = float(np.linalg.norm(inputs.grasp_center_offset_tcp_m))
    if offset_norm <= np.finfo(np.float64).eps:
        raise ValueError("grasp_center_offset_tcp_m must be non-zero")
    target_grasp_position = inputs.target_tcp_position_m + _quaternion_rotate(
        target_tcp_quaternion,
        inputs.grasp_center_offset_tcp_m,
    )
    target_approach_world = _quaternion_rotate(
        target_tcp_quaternion,
        inputs.grasp_center_offset_tcp_m / offset_norm,
    )
    current_approach_delta = grasp_position - tcp_position
    current_approach_norm = float(np.linalg.norm(current_approach_delta))
    if current_approach_norm <= np.finfo(np.float64).eps:
        raise ValueError("current TCP-to-grasp-center direction must be non-zero")
    current_approach_world = current_approach_delta / current_approach_norm

    current_from_target = _canonicalize_scalar_nonnegative(
        _quaternion_multiply(_quaternion_conjugate(target_tcp_quaternion), tcp_quaternion)
    )
    object_quaternion_tcp = _quaternion_multiply(inverse_tcp, cylinder_quaternion)

    blocks = (
        ("observed_joint_positions", joint_positions),
        ("scaled_observed_joint_velocities", joint_velocities * 0.10),
        ("grasp_center_world_position", grasp_position - inputs.environment_origin_m),
        ("tcp_world_quaternion", tcp_quaternion),
        ("cylinder_world_position", cylinder_position - inputs.environment_origin_m),
        ("cylinder_world_quaternion", cylinder_quaternion),
        ("cylinder_to_grasp_center_position", cylinder_position - grasp_position),
        ("scaled_cylinder_linear_velocity", state.cylinder_linear_velocity_m_s * 0.25),
        ("scaled_cylinder_angular_velocity", state.cylinder_angular_velocity_rad_s * 0.25),
        ("right_hand_contact_force_norms", contacts.hand_contact_force_norms_n * 0.10),
        ("internal_physical_action_state", physical_action.values.numpy()),
        ("cylinder_relative_quaternion_in_tcp", object_quaternion_tcp),
        (
            "scaled_cylinder_relative_linear_velocity",
            (state.cylinder_linear_velocity_m_s - inputs.grasp_center_linear_velocity_m_s) * 0.25,
        ),
        ("option_active_progress_support", progress),
        (
            "grasp_position_error_in_cylinder",
            _quaternion_rotate(inverse_cylinder, target_grasp_position - grasp_position),
        ),
        ("current_tcp_relative_target_quaternion", current_from_target),
        (
            "approach_direction_error_in_cylinder",
            _quaternion_rotate(inverse_cylinder, target_approach_world)
            - _quaternion_rotate(inverse_cylinder, current_approach_world),
        ),
    )
    result: list[tuple[str, FloatArray]] = []
    for expected, (name, value) in zip(OBSERVATION_BLOCKS, blocks, strict=True):
        array = np.ascontiguousarray(value, dtype=np.float64)
        if name != expected.name or array.shape != (expected.dimension,):
            raise RuntimeError(
                f"observation block {name!r} violates frozen name/shape {expected.name!r}/{expected.dimension}"
            )
        _require_finite(name, array)
        result.append((name, array))
    return tuple(result)


def build_observation(
    *,
    state: PhysicsState,
    contacts: ContactFrame,
    physical_action: PhysicalAction,
    inputs: ObservationInputs,
    config: ObservationConfig,
) -> Observation106:
    """Build, post-check, clip, and return one CPU ``float32`` observation."""

    blocks = build_observation_blocks(
        state=state,
        contacts=contacts,
        physical_action=physical_action,
        inputs=inputs,
        config=config,
    )
    concatenated = np.concatenate(tuple(value for _, value in blocks))
    if concatenated.shape != (OBSERVATION_DIMENSION,):
        raise RuntimeError(f"observation must have shape (106,), got {concatenated.shape}")
    _require_finite("concatenated_observation", concatenated, "postcheck")
    clipped = np.clip(concatenated, config.clip_min, config.clip_max)
    _require_finite("clipped_observation", clipped, "postcheck")
    return Observation106(torch.from_numpy(clipped.astype(np.float32, copy=False)))


__all__ = [
    "ObservationInputs",
    "ObservationNonFiniteError",
    "build_observation",
    "build_observation_blocks",
]
