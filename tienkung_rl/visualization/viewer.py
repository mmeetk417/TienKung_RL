"""Single-environment native MuJoCo viewer using the M12 task runtime."""

from __future__ import annotations

import argparse
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from rsl_rl.modules import ActorCritic

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.learning.checkpoint import CheckpointError, ValidatedCheckpoint, load_checkpoint_preflight
from tienkung_rl.paths import PROJECT_ROOT as REPOSITORY_ROOT
from tienkung_rl.sim.mujoco_backend import MujocoEnvironmentBackend
from tienkung_rl.task.curriculum import ForcedTargetAuthority
from tienkung_rl.task.environment import ReachThenCloseTask, TaskResetOverrides, TaskSnapshot
from tienkung_rl.task.randomization import ForcedCylinderParameters, ForcedParameterAuthority

from .debug_geometry import (
    DebugGeometryPlan,
    apply_native_debug_geometry,
    build_debug_geometry,
    separate_robot_collision_render_group,
)
from .overlays import apply_native_overlays, build_overlay_sections, clear_native_overlays

VIEWER_RUNTIME_VERSION: Final = "tienkung_native_viewer_v1"
_MODES: Final = frozenset({"policy", "random", "zero", "scripted"})


class ViewerError(RuntimeError):
    """Viewer preflight, controller, display, or native launch failed."""


@dataclass(frozen=True, slots=True)
class ParityRecord:
    """Task decisions that must be identical in viewer and headless assets."""

    trigger_requested: bool
    trigger_accepted: bool
    trigger_rejected: bool
    option_phase: str
    done: bool
    success: bool
    unsafe: bool
    primary_reason: str | None


@dataclass(slots=True)
class RealtimeClock:
    """Pace every completed policy step without dropping simulation work."""

    simulation_start_s: float
    wall_start_s: float
    monotonic: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    @classmethod
    def start(cls, simulation_time_s: float) -> RealtimeClock:
        return cls(simulation_time_s, time.monotonic())

    def factor(self, simulation_time_s: float) -> float:
        simulated = max(0.0, simulation_time_s - self.simulation_start_s)
        elapsed = max(0.0, self.monotonic() - self.wall_start_s)
        return 0.0 if elapsed == 0.0 else simulated / elapsed

    def pace(self, simulation_time_s: float) -> None:
        target = self.wall_start_s + max(0.0, simulation_time_s - self.simulation_start_s)
        remaining = target - self.monotonic()
        if remaining > 0.0:
            self.sleeper(remaining)


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def preflight_viewer_assets(config: ResolvedRunConfig) -> None:
    """Ensure configured assets exist before launch.

    Asset fingerprints are historical metadata only; editing an MJCF or
    manifest must not invalidate the viewer command.
    """

    config.validate()
    expected = (
        ("runtime MJCF", config.asset.runtime_mjcf_path),
        ("viewer MJCF", config.asset.viewer_mjcf_path),
    )
    for label, raw_path in expected:
        path = _path(raw_path).resolve()
        if not path.is_file():
            raise ViewerError(f"{label} is missing: {path}")


def _reset_overrides(args: argparse.Namespace) -> TaskResetOverrides | None:
    target = None if args.target_x_m is None else (float(args.target_x_m), float(args.target_y_m))
    cylinder_values = (
        args.cylinder_radius_m,
        args.cylinder_height_m,
        args.cylinder_mass_kg,
        args.hand_sliding_friction,
        args.table_sliding_friction,
    )
    cylinder = None
    if any(value is not None for value in cylinder_values):
        cylinder = ForcedCylinderParameters(
            radius_m=args.cylinder_radius_m,
            height_m=args.cylinder_height_m,
            mass_kg=args.cylinder_mass_kg,
            hand_sliding_friction=args.hand_sliding_friction,
            table_sliding_friction=args.table_sliding_friction,
        )
    if target is None and cylinder is None:
        return None
    return TaskResetOverrides(
        target_xy_m=target,
        cylinder=cylinder,
        target_authority=None if target is None else ForcedTargetAuthority.VIEWER,
        cylinder_authority=None if cylinder is None else ForcedParameterAuthority.VIEWER,
    )


def _new_policy(config: ResolvedRunConfig) -> ActorCritic:
    network = config.ppo.network
    return ActorCritic(
        num_actor_obs=106,
        num_critic_obs=106,
        num_actions=7,
        actor_hidden_dims=network.actor_hidden_dimensions,
        critic_hidden_dims=network.critic_hidden_dimensions,
        activation=network.activation,
        init_noise_std=config.ppo.initial_continuous_std,
        min_std=config.ppo.minimum_continuous_std,
        max_std=config.ppo.maximum_continuous_std,
    )


def _load_policy(config: ResolvedRunConfig, checkpoint: Path) -> tuple[ActorCritic, ValidatedCheckpoint]:
    # M15 validates the complete pair, all contracts, active source, dependency
    # versions and assets before a model or MuJoCo environment can be used.
    validated = load_checkpoint_preflight(checkpoint, config, mode="inference")
    model_state = validated.payload.get("model_state_dict")
    action_space = validated.payload.get("policy_action_space")
    if not isinstance(model_state, Mapping) or not isinstance(action_space, Mapping):
        raise CheckpointError("checkpoint policy payload is malformed")
    policy = _new_policy(config)
    if dict(action_space) != policy.action_space_metadata():
        raise CheckpointError("checkpoint policy is not the exact seven-joint arm contract")
    policy.load_state_dict(cast(Mapping[str, torch.Tensor], model_state), strict=True)
    policy.eval()
    return policy, validated


class ViewerController:
    """Four explicit sources that all emit the exact CPU 7-D arm action."""

    def __init__(
        self,
        mode: str,
        *,
        seed: int,
        policy: ActorCritic | None = None,
        scripted_arm_source: Callable[[TaskSnapshot], torch.Tensor] | None = None,
    ) -> None:
        if mode not in _MODES:
            raise ViewerError(f"unknown viewer mode {mode!r}")
        if type(seed) is not int or seed < 0:
            raise ViewerError("viewer seed must be a non-negative integer")
        if (mode == "policy") != (policy is not None):
            raise ViewerError("policy mode requires exactly one preflighted policy")
        if (mode == "scripted") != (scripted_arm_source is not None):
            raise ViewerError("scripted mode requires exactly one deterministic reach source")
        self.mode = mode
        self.policy = policy
        self.scripted_arm_source = scripted_arm_source
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def action(self, observation: torch.Tensor, snapshot: TaskSnapshot) -> torch.Tensor:
        if observation.device.type != "cpu" or observation.dtype != torch.float32 or observation.shape != (106,):
            raise ViewerError("viewer observation must be a float32 CPU tensor with shape (106,)")
        if self.mode == "policy":
            assert self.policy is not None
            with torch.inference_mode():
                action = self.policy.act_inference(observation.unsqueeze(0)).squeeze(0)
        elif self.mode == "random":
            action = 2.0 * torch.rand(7, generator=self.generator) - 1.0
        else:
            action = torch.zeros(7, dtype=torch.float32)
            if self.mode == "scripted":
                assert self.scripted_arm_source is not None
                action[:] = self.scripted_arm_source(snapshot)
        if action.device.type != "cpu" or action.dtype != torch.float32 or action.shape != (7,):
            raise ViewerError("controller produced an invalid arm action")
        if not bool(torch.isfinite(action).all()):
            raise ViewerError("controller produced a non-finite action")
        return action


def _quaternion_rotation_error(
    target_wxyz: tuple[float, float, float, float],
    current_wxyz: tuple[float, float, float, float],
) -> np.ndarray:
    target = np.asarray(target_wxyz, dtype=np.float64)
    current = np.asarray(current_wxyz, dtype=np.float64)
    cw, cx, cy, cz = current
    tw, tx, ty, tz = target
    error = np.asarray(
        (
            tw * cw + tx * cx + ty * cy + tz * cz,
            -tw * cx + tx * cw - ty * cz + tz * cy,
            -tw * cy + tx * cz + ty * cw - tz * cx,
            -tw * cz - tx * cy + ty * cx + tz * cw,
        ),
        dtype=np.float64,
    )
    if error[0] < 0.0:
        error *= -1.0
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, max(0.0, float(error[0])))
    return np.asarray(angle * error[1:] / vector_norm, dtype=np.float64)


def _rotate(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    scalar = float(quaternion_wxyz[0])
    imaginary = quaternion_wxyz[1:]
    return np.asarray(
        2.0 * np.dot(imaginary, vector) * imaginary
        + (scalar * scalar - np.dot(imaginary, imaginary)) * vector
        + 2.0 * scalar * np.cross(imaginary, vector),
        dtype=np.float64,
    )


def fixed_scripted_reach_action(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    snapshot: TaskSnapshot,
) -> torch.Tensor:
    """Deterministic DLS reach to the final TCP target.

    This stateless helper remains useful for diagnostics and tests.  The live
    viewer uses :class:`ScriptedReachController` below, which feeds it staged
    collision-avoiding TCP waypoints.
    """

    return _dls_reach_action(
        config,
        backend,
        np.asarray(snapshot.target_tcp_position_m, dtype=np.float64),
        snapshot.target_tcp_quaternion_wxyz,
    )


# The admission contract allows at most 0.20 rad/s; at the 0.020 s policy
# period this cap keeps the scripted teacher inside that speed envelope.
_SCRIPTED_MAX_JOINT_STEP_RAD: Final = 0.004


def _dls_reach_action(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    target_position: np.ndarray,
    target_quaternion: tuple[float, float, float, float],
) -> torch.Tensor:
    """Compute one bounded Cartesian DLS TCP waypoint action."""

    state = backend.read_state()
    positions, tcp_jacobian = backend.read_arm_site_jacobian("right_tcp_site")
    # The IsaacLab reach teacher is defined on the TCP pose.  Admission later
    # evaluates the derived grasp-center pose, but replacing the TCP Jacobian
    # with a grasp-center Jacobian here changes the migrated controller's task.
    position_error = np.asarray(target_position, dtype=np.float64) - state.tcp_position_m
    current_quaternion = (
        float(state.tcp_quaternion_wxyz[0]),
        float(state.tcp_quaternion_wxyz[1]),
        float(state.tcp_quaternion_wxyz[2]),
        float(state.tcp_quaternion_wxyz[3]),
    )
    rotation_error = _quaternion_rotation_error(
        target_quaternion,
        current_quaternion,
    )
    cartesian_error = np.concatenate((position_error, rotation_error))
    jacobian = tcp_jacobian
    damping_squared = 1.0e-6
    normal = jacobian @ jacobian.T + damping_squared * np.eye(6, dtype=np.float64)
    joint_delta = jacobian.T @ np.linalg.solve(normal, cartesian_error)
    # Match the IsaacLab Cartesian teacher by scaling the whole joint update,
    # rather than clipping each component.  Per-joint clipping (the previous
    # MuJoCo port) changes the DLS direction and makes the target unreachable.
    maximum_requested_step = float(np.max(np.abs(joint_delta)))
    if maximum_requested_step > _SCRIPTED_MAX_JOINT_STEP_RAD:
        joint_delta = joint_delta * (_SCRIPTED_MAX_JOINT_STEP_RAD / maximum_requested_step)
    desired = positions + joint_delta
    center = np.asarray(config.task.action.center_joint_positions_rad, dtype=np.float64)
    half_range = np.asarray(config.task.action.half_range_rad, dtype=np.float64)
    normalized = np.clip((desired - center) / half_range, -1.0, 1.0)
    result = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))
    if result.shape != (7,) or not bool(torch.isfinite(result).all()):
        raise ViewerError("fixed scripted reach produced an invalid arm action")
    return result


@dataclass(slots=True)
class ScriptedReachController:
    """Collision-avoiding TCP teacher used by the native scripted viewer."""

    config: ResolvedRunConfig
    backend: MujocoEnvironmentBackend
    stage: int = 0
    stable_steps: int = 0
    _episode_step: int = -1
    _waypoints: tuple[np.ndarray, ...] | None = None
    _waypoint_quaternions: tuple[tuple[float, float, float, float], ...] | None = None

    def reset(self) -> None:
        self.stage = 0
        self.stable_steps = 0
        self._episode_step = -1
        self._waypoints = None
        self._waypoint_quaternions = None

    def _initialize(self, snapshot: TaskSnapshot) -> None:
        state = self.backend.read_state()
        final = np.asarray(snapshot.target_tcp_position_m, dtype=np.float64)
        object_position = np.asarray(snapshot.cylinder_reset_position_m, dtype=np.float64)
        approach = final - object_position
        approach[2] = 0.0
        norm = float(np.linalg.norm(approach))
        if norm <= 1.0e-9:
            right = np.zeros(3, dtype=np.float64)
        else:
            # In the world XY plane, cross(approach, +Z) is the right-hand
            # side when looking from the object toward the target.
            right = np.asarray((approach[1], -approach[0], 0.0), dtype=np.float64) / norm
        self._waypoints = (
            np.asarray(state.tcp_position_m, dtype=np.float64) + 0.015 * right,
            final + 0.025 * right,
            final,
        )
        current_quaternion = tuple(float(value) for value in state.tcp_quaternion_wxyz)
        self._waypoint_quaternions = (
            current_quaternion,
            snapshot.target_tcp_quaternion_wxyz,
            snapshot.target_tcp_quaternion_wxyz,
        )

    def action(self, snapshot: TaskSnapshot) -> torch.Tensor:
        if snapshot.episode_step < self._episode_step:
            self.reset()
        self._episode_step = snapshot.episode_step
        if self._waypoints is None:
            self._initialize(snapshot)
        if self._waypoints is None or self._waypoint_quaternions is None:  # pragma: no cover
            raise ViewerError("scripted reach waypoints were not initialized")
        state = self.backend.read_state()
        target = self._waypoints[self.stage]
        target_quaternion = self._waypoint_quaternions[self.stage]
        position_error = float(np.linalg.norm(target - state.tcp_position_m))
        orientation_error = float(
            np.linalg.norm(_quaternion_rotation_error(target_quaternion, tuple(state.tcp_quaternion_wxyz)))
        )
        if position_error <= 0.01 and orientation_error <= 0.20:
            self.stable_steps += 1
            if self.stable_steps >= 5 and self.stage < len(self._waypoints) - 1:
                self.stage += 1
                self.stable_steps = 0
                target = self._waypoints[self.stage]
                target_quaternion = self._waypoint_quaternions[self.stage]
        else:
            self.stable_steps = 0
        return _dls_reach_action(self.config, self.backend, target, target_quaternion)


def run_case_trace(
    config: ResolvedRunConfig,
    actions: Sequence[torch.Tensor],
    *,
    seed: int,
    reset_overrides: TaskResetOverrides | None = None,
    viewer_asset: bool,
) -> tuple[ParityRecord, ...]:
    """Automated viewer-runtime/headless parity probe with no GL context."""

    preflight_viewer_assets(config)
    path = config.asset.viewer_mjcf_path if viewer_asset else config.asset.runtime_mjcf_path
    backend = MujocoEnvironmentBackend(config, mjcf_path=path)
    task = ReachThenCloseTask(
        config,
        backend=backend,
        seed=seed,
        reset_overrides=reset_overrides,
        curriculum_updates_enabled=False,
    )
    records: list[ParityRecord] = []
    try:
        task.reset()
        for action in actions:
            result = task.step(action)
            snapshot = result.transition.snapshot
            records.append(
                ParityRecord(
                    snapshot.trigger_requested,
                    snapshot.trigger_accepted,
                    snapshot.trigger_rejected,
                    snapshot.option_phase,
                    snapshot.done,
                    snapshot.success,
                    snapshot.unsafe,
                    snapshot.primary_termination_reason,
                )
            )
    finally:
        backend.close()
    return tuple(records)


def _require_display() -> None:
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise ViewerError(
            "native MuJoCo viewer requires DISPLAY or WAYLAND_DISPLAY; "
            "use --headless-steps for an automated no-GUI probe"
        )


def apply_viewer_brightness(model: Any, brightness: float) -> None:
    """Scale MuJoCo's headlight without changing simulation physics."""

    if not math.isfinite(brightness) or brightness < 0.0 or brightness > 10.0:
        raise ViewerError("viewer brightness must be between 0 and 10")
    headlight = model.vis.headlight
    for channel in ("ambient", "diffuse", "specular"):
        values = getattr(headlight, channel)
        values[...] = np.asarray(values, dtype=np.float64) * brightness


def _run_loop(
    config: ResolvedRunConfig,
    args: argparse.Namespace,
    backend: MujocoEnvironmentBackend,
    task: ReachThenCloseTask,
    controller: ViewerController,
) -> int:
    observation = task.reset().values
    headless_steps = args.headless_steps
    if headless_steps is not None:
        for _ in range(headless_steps):
            result = task.step(controller.action(observation, task.snapshot()))
            observation = result.observation.values
        final = task.snapshot()
        print(
            f"viewer headless probe PASS mode={controller.mode} steps={headless_steps} "
            f"physics_step={final.physics_step} phase={final.option_phase}"
        )
        return 0

    _require_display()
    import mujoco.viewer  # type: ignore[import-untyped]

    model, data = backend.native_viewer_handles()
    apply_viewer_brightness(model, args.brightness)
    separate_robot_collision_render_group(model)
    clock = RealtimeClock.start(task.snapshot().simulation_time_s)
    terminal_pose_held = False
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as native:
            while native.is_running():
                if terminal_pose_held:
                    native.sync()
                    time.sleep(1.0 / 60.0)
                    continue
                before = task.snapshot()
                result = task.step(controller.action(observation, before), auto_reset=False)
                observation = result.observation.values
                snapshot = result.transition.snapshot
                factor = clock.factor(snapshot.simulation_time_s)
                state = backend.read_state()
                contacts = backend.read_contacts()
                plan = build_debug_geometry(
                    snapshot,
                    state,
                    contacts,
                    admission_position_error_m=config.task.admission.position_error_m,
                    show_robot_collision_geometries=args.show_collisions,
                )
                if args.hide_debug_geometry:
                    plan = DebugGeometryPlan((), plan.show_robot_collision_geometries)
                with native.lock():
                    apply_native_debug_geometry(native, plan)
                    if args.hide_overlays:
                        clear_native_overlays(native)
                    else:
                        sections = build_overlay_sections(snapshot, mode=controller.mode, realtime_factor=factor)
                        apply_native_overlays(native, sections)
                native.sync()
                clock.pace(snapshot.simulation_time_s)
                if result.transition.done:
                    if args.pause_on_terminal:
                        terminal_pose_held = True
                    else:
                        observation = task.reset().values
                        clock = RealtimeClock.start(task.snapshot().simulation_time_s)
    except Exception as error:
        raise ViewerError(f"native MuJoCo viewer failed: {error}") from error
    return 0


def run(config: ResolvedRunConfig, args: argparse.Namespace) -> int:
    """Preflight all selected inputs, then construct exactly one task/backend."""

    preflight_viewer_assets(config)
    policy = None
    if args.mode == "policy":
        policy, _ = _load_policy(config, Path(args.checkpoint))
    overrides = _reset_overrides(args)
    backend = MujocoEnvironmentBackend(config, mjcf_path=config.asset.viewer_mjcf_path)
    task = ReachThenCloseTask(
        config,
        backend=backend,
        seed=config.run.seed,
        reset_overrides=overrides,
        curriculum_updates_enabled=False,
    )
    scripted_reach = ScriptedReachController(config, backend) if args.mode == "scripted" else None
    controller = ViewerController(
        args.mode,
        seed=config.run.seed,
        policy=policy,
        scripted_arm_source=((lambda snapshot: scripted_reach.action(snapshot)) if args.mode == "scripted" else None),
    )
    try:
        return _run_loop(config, args, backend, task, controller)
    finally:
        backend.close()


__all__ = [
    "VIEWER_RUNTIME_VERSION",
    "ParityRecord",
    "RealtimeClock",
    "ScriptedReachController",
    "ViewerController",
    "ViewerError",
    "fixed_scripted_reach_action",
    "preflight_viewer_assets",
    "run",
    "run_case_trace",
]
