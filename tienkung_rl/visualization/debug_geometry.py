"""Read-only debug geometry plans and their native MuJoCo renderer adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from tienkung_rl.sim.state import ContactFrame, PhysicsState
from tienkung_rl.task.environment import TaskSnapshot

FloatArray = NDArray[np.float64]
PrimitiveKind = Literal["sphere", "arrow"]


@dataclass(frozen=True, slots=True)
class DebugPrimitive:
    """Simulator-independent drawable with stable semantic coloring."""

    label: str
    kind: PrimitiveKind
    start_m: tuple[float, float, float]
    end_m: tuple[float, float, float] | None
    radius_m: float
    rgba: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        values: tuple[float, ...] = (*self.start_m, self.radius_m, *self.rgba)
        if self.end_m is not None:
            values = (*values, *self.end_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("debug primitive contains a non-finite value")
        if self.radius_m <= 0.0 or any(not 0.0 <= value <= 1.0 for value in self.rgba):
            raise ValueError("debug primitive radius/color is invalid")
        if self.kind == "arrow" and self.end_m is None:
            raise ValueError("arrow debug primitives require an end point")


@dataclass(frozen=True, slots=True)
class DebugGeometryPlan:
    primitives: tuple[DebugPrimitive, ...]
    show_robot_collision_geometries: bool


class _UserScene(Protocol):
    ngeom: int
    maxgeom: int
    geoms: Any


class _ViewerOption(Protocol):
    geomgroup: Any


class NativeViewerScene(Protocol):
    user_scn: _UserScene
    opt: _ViewerOption


class NativeViewerModel(Protocol):
    geom_contype: Any
    geom_group: Any


_TASK_COLLISION_RENDER_GROUP = 3
_ROBOT_COLLISION_RENDER_GROUP = 5
_ROBOT_COLLISION_TYPE = 1


def separate_robot_collision_render_group(model: NativeViewerModel) -> int:
    """Move robot collision geoms to a viewer-only group without changing physics."""

    geom_groups = np.asarray(model.geom_group)
    geom_contype = np.asarray(model.geom_contype)
    robot_collision_mask = (geom_groups == _TASK_COLLISION_RENDER_GROUP) & (
        geom_contype == _ROBOT_COLLISION_TYPE
    )
    count = int(np.count_nonzero(robot_collision_mask))
    if count == 0:
        raise ValueError("viewer model has no robot collision geometry to separate")
    geom_groups[robot_collision_mask] = _ROBOT_COLLISION_RENDER_GROUP
    return count


def _tuple3(value: FloatArray) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _rotation_matrix(quaternion_wxyz: tuple[float, float, float, float]) -> FloatArray:
    w, x, y, z = quaternion_wxyz
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _contact_color(groups: frozenset[str]) -> tuple[float, float, float, float]:
    if groups.intersection({"right_arm", "tcp", "non_task_body"}):
        return (1.0, 0.1, 0.1, 1.0)
    if "thumb_side" in groups:
        return (0.15, 0.45, 1.0, 1.0)
    if "four_finger_opposing" in groups:
        return (0.1, 0.9, 0.25, 1.0)
    return (1.0, 0.75, 0.1, 1.0)


def build_debug_geometry(
    snapshot: TaskSnapshot,
    state: PhysicsState,
    contacts: ContactFrame,
    *,
    admission_position_error_m: float,
    show_robot_collision_geometries: bool = False,
) -> DebugGeometryPlan:
    """Build target frames, admission volume, and latest contact vectors."""

    if not math.isfinite(admission_position_error_m) or admission_position_error_m <= 0.0:
        raise ValueError("admission position threshold must be finite and positive")
    primitives: list[DebugPrimitive] = []
    current_axes = _rotation_matrix(
        (
            float(state.tcp_quaternion_wxyz[0]),
            float(state.tcp_quaternion_wxyz[1]),
            float(state.tcp_quaternion_wxyz[2]),
            float(state.tcp_quaternion_wxyz[3]),
        )
    )
    grasp_offset_tcp = current_axes.T @ (state.grasp_center_position_m - state.tcp_position_m)
    target_axes = _rotation_matrix(snapshot.target_tcp_quaternion_wxyz)
    target_grasp_array = np.asarray(snapshot.target_tcp_position_m, dtype=np.float64) + target_axes @ grasp_offset_tcp
    grasp_target = _tuple3(target_grasp_array)
    # The two 2 cm spheres make the scalar admission distance immediately
    # visible: target is magenta, current is green.
    primitives.append(DebugPrimitive("target_grasp", "sphere", grasp_target, None, 0.020, (1.0, 0.1, 0.8, 0.28)))
    primitives.append(
        DebugPrimitive(
            "current_grasp",
            "sphere",
            _tuple3(state.grasp_center_position_m),
            None,
            0.020,
            (0.1, 1.0, 0.2, 0.24),
        )
    )
    target_tcp = np.asarray(snapshot.target_tcp_position_m, dtype=np.float64)
    primitives.append(
        DebugPrimitive(
            "option_admission_position_volume",
            "sphere",
            snapshot.target_tcp_position_m,
            None,
            admission_position_error_m,
            (0.3, 0.8, 1.0, 0.16),
        )
    )
    axes = target_axes
    for index, (name, color) in enumerate(
        (
            ("target_tcp_x", (1.0, 0.1, 0.1, 1.0)),
            ("target_tcp_y", (0.1, 1.0, 0.1, 1.0)),
            ("target_tcp_z", (0.1, 0.3, 1.0, 1.0)),
        )
    ):
        end = target_tcp + 0.06 * axes[:, index]
        primitives.append(DebugPrimitive(name, "arrow", snapshot.target_tcp_position_m, _tuple3(end), 0.002, color))

    latest_substep = max((sample.substep_index for sample in contacts.contacts), default=-1)
    for index, sample in enumerate(contacts.contacts):
        if sample.substep_index != latest_substep:
            continue
        groups = frozenset((*sample.geom1_groups, *sample.geom2_groups))
        color = _contact_color(groups)
        start = np.asarray(sample.position_m, dtype=np.float64)
        normal = np.asarray(sample.normal_from_geom1_to_geom2, dtype=np.float64)
        normal_end = start + 0.025 * normal
        force = np.asarray(sample.force_on_geom2_world_n, dtype=np.float64)
        force_norm = float(np.linalg.norm(force))
        force_direction = normal if force_norm <= 1.0e-12 else force / force_norm
        force_length = min(0.08, 0.008 + 0.006 * math.log1p(force_norm))
        force_end = start + force_length * force_direction
        primitives.append(DebugPrimitive(f"contact_{index}", "sphere", _tuple3(start), None, 0.003, color))
        primitives.append(
            DebugPrimitive(f"contact_normal_{index}", "arrow", _tuple3(start), _tuple3(normal_end), 0.001, color)
        )
        primitives.append(
            DebugPrimitive(f"contact_force_{index}", "arrow", _tuple3(start), _tuple3(force_end), 0.0015, color)
        )

    if state.total_physics_steps != snapshot.physics_step:
        raise ValueError("physics state and task snapshot refer to different steps")
    return DebugGeometryPlan(tuple(primitives), show_robot_collision_geometries)


def apply_native_debug_geometry(viewer: NativeViewerScene, plan: DebugGeometryPlan) -> None:
    """Populate only ``user_scn`` and viewer options; never model/task state."""

    import mujoco  # type: ignore[import-untyped]

    scene = viewer.user_scn
    scene.ngeom = 0
    # Task collision geoms are the visible table/cylinder. Robot collision
    # volumes live in a separate render-only group and are opt-in diagnostics.
    viewer.opt.geomgroup[_TASK_COLLISION_RENDER_GROUP] = 1
    viewer.opt.geomgroup[_ROBOT_COLLISION_RENDER_GROUP] = int(plan.show_robot_collision_geometries)
    for primitive in plan.primitives:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        rgba = np.asarray(primitive.rgba, dtype=np.float32)
        if primitive.kind == "sphere":
            size = np.asarray((primitive.radius_m,) * 3, dtype=np.float64)
            position = np.asarray(primitive.start_m, dtype=np.float64)
            rotation = np.eye(3, dtype=np.float64).reshape(-1)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, position, rotation, rgba)
        else:
            assert primitive.end_m is not None
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                primitive.radius_m,
                np.asarray(primitive.start_m, dtype=np.float64),
                np.asarray(primitive.end_m, dtype=np.float64),
            )
            geom.rgba[:] = rgba
        scene.ngeom += 1


__all__ = [
    "DebugGeometryPlan",
    "DebugPrimitive",
    "apply_native_debug_geometry",
    "build_debug_geometry",
    "separate_robot_collision_render_group",
]
