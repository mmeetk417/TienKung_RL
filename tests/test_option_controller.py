"""M09 pure-state and real-MuJoCo Option controller acceptance tests."""

from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.task.contracts import ACTIVE_HAND_JOINT_NAMES, ARM_JOINT_NAMES, ARM_POLICY_DIMENSION
from tienkung_rl.task.option import (
    BILATERAL_PERSISTENCE_STEPS,
    CLOSE_STEPS,
    CONTACT_FORCE_THRESHOLD_N,
    HOLD_STEPS,
    LIFT_HOLD_STEPS,
    LIFT_STEPS,
    SETTLE_STEPS,
    SUPPORTED_LIFT_PERSISTENCE_STEPS,
    TOTAL_SCRIPT_STEPS,
    LiftPreparation,
    OptionController,
    OptionEpisodeState,
    OptionEvent,
    OptionPhase,
    OptionSubstepObservation,
    OptionTiming,
    process_reach_transition,
    smoothstep,
    solve_cartesian_lift_target,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config(ROOT / "configs/reach_then_close.yaml")


def _model_data(config: ResolvedRunConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(config.asset.runtime_mjcf_path)
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "robot_and_scene_home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    return model, data


def _controller(
    config: ResolvedRunConfig,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> OptionController:
    controller = OptionController(
        config.task.option,
        np.array(data.ctrl[7:], dtype=np.float64, order="C"),
        np.array(model.actuator_ctrlrange[:, 0], dtype=np.float64, order="C"),
        np.array(model.actuator_ctrlrange[:, 1], dtype=np.float64, order="C"),
    )
    controller.reset(np.array(data.ctrl, dtype=np.float64, order="C"))
    return controller


def _accept(controller: OptionController, arm_targets: np.ndarray) -> None:
    controller.begin_policy_step(arm_targets)
    for _ in range(4):
        controller.observe_substep(OptionSubstepObservation())
    state = controller.finish_policy_step(trigger_requested=True, option_admissible=True)
    assert state.phase is OptionPhase.ScriptPending


def _lift_preparation(model: mujoco.MjModel, data: mujoco.MjData) -> LiftPreparation:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tcp_site")
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacobian_position, jacobian_rotation, site_id)
    dof_addresses: list[int] = []
    qpos_addresses: list[int] = []
    for name in ARM_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    jacobian = np.vstack(
        (jacobian_position[:, dof_addresses], jacobian_rotation[:, dof_addresses])
    )
    return LiftPreparation(
        measured_arm_positions_rad=tuple(float(data.qpos[index]) for index in qpos_addresses),
        tcp_world_position_m=tuple(float(value) for value in data.site_xpos[site_id]),
        tcp_world_quaternion_wxyz=tuple(float(value) for value in quaternion),
        tcp_jacobian_6x7=tuple(tuple(float(value) for value in row) for row in jacobian),
    )


def test_state_enum_timing_and_contract_constants(config: ResolvedRunConfig) -> None:
    assert tuple(phase.value for phase in OptionPhase) == (
        "Reach",
        "ScriptPending",
        "Settle",
        "Close",
        "Hold",
        "Lift",
        "LiftHold",
        "Success",
        "Failure",
    )
    timing = OptionTiming.from_config(config.task.option)
    assert timing.phase_steps == (25, 175, 50, 100, 250)
    assert timing.total_steps == TOTAL_SCRIPT_STEPS == 600
    assert timing.phase_steps == (
        SETTLE_STEPS,
        CLOSE_STEPS,
        HOLD_STEPS,
        LIFT_STEPS,
        LIFT_HOLD_STEPS,
    )
    assert CONTACT_FORCE_THRESHOLD_N == 0.20
    assert BILATERAL_PERSISTENCE_STEPS == 3
    assert SUPPORTED_LIFT_PERSISTENCE_STEPS == 5
    assert config.task.option.active_hand_joint_names == ACTIVE_HAND_JOINT_NAMES
    with pytest.raises(ValueError, match="exactly"):
        OptionTiming(24, 80, 50, 100, 251)
    module = inspect.getmodule(process_reach_transition)
    assert module is not None
    assert "mujoco" not in module.__dict__


def test_reach_pure_transition_no_request_rejection_and_acceptance() -> None:
    arm = np.linspace(-0.3, 0.3, 7, dtype=np.float64)
    initial = OptionEpisodeState()
    no_request = process_reach_transition(
        initial,
        trigger_requested=False,
        option_admissible=True,
        executed_arm_targets_rad=arm,
    )
    assert no_request.phase is OptionPhase.Reach
    assert no_request.transition_events == ()
    assert no_request.request_count == 0

    rejected = process_reach_transition(
        no_request,
        trigger_requested=True,
        option_admissible=False,
        executed_arm_targets_rad=arm,
    )
    assert rejected.phase is OptionPhase.Reach
    assert rejected.trigger_requested and rejected.trigger_rejected and not rejected.trigger_accepted
    assert rejected.transition_events == (OptionEvent.TriggerRequested, OptionEvent.TriggerRejected)
    assert rejected.request_count == rejected.rejection_count == 1

    accepted = process_reach_transition(
        rejected,
        trigger_requested=True,
        option_admissible=True,
        executed_arm_targets_rad=arm,
    )
    assert accepted.phase is OptionPhase.ScriptPending
    assert accepted.trigger_accepted and accepted.acceptance_latched
    assert accepted.acceptance_count == 1
    assert accepted.latched_arm_targets_rad is not None
    assert accepted.latched_arm_targets_rad == pytest.approx(tuple(arm))
    assert accepted.transition_events == (OptionEvent.TriggerRequested, OptionEvent.TriggerAccepted)
    assert initial == OptionEpisodeState()


def test_accept_transition_is_pending_then_policy_is_ignored(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    admitted_arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, admitted_arm)
    pending = controller.state
    assert pending.script_steps_completed == 0
    assert pending.script_progress == 0.0
    assert pending.latched_arm_targets_rad is not None

    hostile_policy = np.array(model.actuator_ctrlrange[:7, 1], dtype=np.float64, order="C")
    first_script_target = controller.begin_policy_step(hostile_policy)
    assert controller.state.phase is OptionPhase.Settle
    np.testing.assert_allclose(first_script_target.values.numpy()[:7], pending.latched_arm_targets_rad, atol=0.0)
    for _ in range(4):
        controller.observe_substep(OptionSubstepObservation())
    state = controller.finish_policy_step(trigger_requested=True, option_admissible=True)
    assert state.phase is OptionPhase.Settle
    assert state.phase_step == 1
    assert state.acceptance_count == 1
    assert state.request_count == 1
    assert state.latched_arm_targets_rad == pending.latched_arm_targets_rad
    assert not state.trigger_requested and not state.trigger_accepted and not state.trigger_rejected


def test_close_and_lift_boundaries_are_continuous_limited_and_exact(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, arm)
    outputs: list[np.ndarray] = []
    phases: list[OptionPhase] = []
    preparation: LiftPreparation | None = None

    for _ in range(TOTAL_SCRIPT_STEPS):
        kwargs: dict[str, LiftPreparation] = {}
        if controller.state.phase is OptionPhase.Lift:
            preparation = _lift_preparation(model, data)
            kwargs["lift_preparation"] = preparation
        output = controller.begin_policy_step(np.full(7, 1.0, dtype=np.float64), **kwargs)
        phases.append(controller.state.phase)
        values = output.values.numpy().copy()
        outputs.append(values)
        data.ctrl[:] = values
        for joint_index, name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = values[joint_index]
        mujoco.mj_forward(model, data)
        for _ in range(4):
            # The pure target test deliberately supplies qualifying diagnostics;
            # real contact failure is covered by the MuJoCo integration below.
            controller.observe_substep(
                OptionSubstepObservation(
                    thumb_side_force_n=0.20,
                    opposing_side_force_n=0.20,
                    net_hand_force_n=0.20,
                    object_lift_height_m=0.05,
                )
            )
        controller.finish_policy_step(trigger_requested=True, option_admissible=True)

    assert Counter(phases) == {
        OptionPhase.Settle: 25,
        OptionPhase.Close: 175,
        OptionPhase.Hold: 50,
        OptionPhase.Lift: 100,
        OptionPhase.LiftHold: 250,
    }
    assert controller.state.phase is OptionPhase.Success
    assert controller.state.script_steps_completed == TOTAL_SCRIPT_STEPS
    assert controller.state.script_progress == 1.0
    assert controller.state.transition_events[-1] is OptionEvent.CompleteSuccess
    assert preparation is not None

    trajectory = np.asarray(outputs)
    assert np.all(trajectory >= model.actuator_ctrlrange[:, 0] - 1.0e-12)
    assert np.all(trajectory <= model.actuator_ctrlrange[:, 1] + 1.0e-12)
    initial = np.array((*arm, *data.ctrl[7:] * 0.0), dtype=np.float64)
    arm_deltas = np.diff(np.vstack((initial, trajectory)), axis=0)[:, :ARM_POLICY_DIMENSION]
    assert float(np.max(np.abs(arm_deltas))) <= 0.01 + 1.0e-12
    assert np.any(np.abs(trajectory[-1, :7] - arm) > 1.0e-6)
    np.testing.assert_allclose(trajectory[:SETTLE_STEPS, 7:], 0.0, atol=0.0)
    close_end = SETTLE_STEPS + CLOSE_STEPS - 1
    expected_closed = model.actuator_ctrlrange[7:, 1].copy()
    expected_closed[0] = 0.0
    np.testing.assert_allclose(trajectory[close_end, 7:], expected_closed, atol=1.0e-12)
    np.testing.assert_allclose(
        trajectory[close_end + 1 :, 7:],
        np.broadcast_to(expected_closed, (TOTAL_SCRIPT_STEPS - close_end - 1, 6)),
        atol=0.0,
    )
    assert controller.state.lift_final_arm_targets_rad is not None
    lift_end = SETTLE_STEPS + CLOSE_STEPS + HOLD_STEPS + LIFT_STEPS - 1
    np.testing.assert_allclose(
        trajectory[lift_end, :7],
        controller.state.lift_final_arm_targets_rad,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_contact_threshold_persistence_and_reset_are_exact(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, arm)

    for expected in (1, 2, 3):
        controller.begin_policy_step(arm)
        controller.observe_substep(
            OptionSubstepObservation(
                thumb_side_force_n=CONTACT_FORCE_THRESHOLD_N,
                opposing_side_force_n=CONTACT_FORCE_THRESHOLD_N,
                net_hand_force_n=CONTACT_FORCE_THRESHOLD_N,
                object_lift_height_m=0.05,
            )
        )
        state = controller.finish_policy_step(trigger_requested=False, option_admissible=False)
        assert state.bilateral_contact_steps == expected
    assert state.bilateral_persistent_ever
    assert state.support_progress == pytest.approx(3 / 5)

    controller.begin_policy_step(arm)
    controller.observe_substep(
        OptionSubstepObservation(
            thumb_side_force_n=np.nextafter(0.20, 0.0),
            opposing_side_force_n=0.20,
            net_hand_force_n=np.nextafter(0.20, 0.0),
            object_lift_height_m=0.05,
        )
    )
    state = controller.finish_policy_step(trigger_requested=False, option_admissible=False)
    assert state.bilateral_contact_steps == 0
    assert state.supported_lift_steps == 0
    assert state.bilateral_persistent_ever

    controller.reset(np.array(data.ctrl, dtype=np.float64, order="C"))
    assert controller.state == OptionEpisodeState()
    assert controller.state.phase is OptionPhase.Reach
    assert controller.state.script_progress == controller.state.support_progress == 0.0


@pytest.mark.parametrize(
    ("observation", "event"),
    [
        (OptionSubstepObservation(toppled=True), OptionEvent.Toppled),
        (OptionSubstepObservation(dropped=True), OptionEvent.Dropped),
        (OptionSubstepObservation(illegal_contact=True), OptionEvent.IllegalContact),
        (OptionSubstepObservation(state_finite=False), OptionEvent.NonFinite),
        (OptionSubstepObservation(thumb_side_force_n=float("nan")), OptionEvent.NonFinite),
    ],
)
def test_every_substep_hard_safety_event_fails_immediately(
    config: ResolvedRunConfig,
    observation: OptionSubstepObservation,
    event: OptionEvent,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, arm)
    controller.begin_policy_step(arm)
    state = controller.observe_substep(observation)
    assert state.phase is OptionPhase.Failure
    assert event in state.hard_safety_events
    assert event in state.transition_events


def test_simultaneous_hard_safety_events_keep_one_stable_primary(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, arm)
    controller.begin_policy_step(arm)
    state = controller.observe_substep(
        OptionSubstepObservation(
            toppled=True,
            dropped=True,
            illegal_contact=True,
            state_finite=False,
        )
    )
    assert state.hard_safety_events == (
        OptionEvent.NonFinite,
        OptionEvent.Toppled,
        OptionEvent.Dropped,
        OptionEvent.IllegalContact,
    )


def test_relative_displacement_is_not_a_script_failure(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    _accept(controller, arm)
    controller.begin_policy_step(arm)
    state = controller.observe_substep(OptionSubstepObservation(illegal_displacement=True))
    assert state.phase is OptionPhase.Settle
    assert OptionEvent.IllegalDisplacement not in state.hard_safety_events


def test_cartesian_lift_solver_preserves_linearized_tcp_pose(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    preparation = _lift_preparation(model, data)
    previous = np.asarray(preparation.measured_arm_positions_rad, dtype=np.float64)
    target = solve_cartesian_lift_target(
        preparation,
        start_tcp_position_m=preparation.tcp_world_position_m,
        start_tcp_quaternion_wxyz=preparation.tcp_world_quaternion_wxyz,
        waypoint_height_m=0.001,
        previous_arm_targets_rad=previous,
    )
    delta = target - previous
    linearized_motion = np.asarray(preparation.tcp_jacobian_6x7) @ delta
    np.testing.assert_allclose(linearized_motion[:2], 0.0, atol=2.0e-6)
    assert 0.0 < linearized_motion[2] <= 0.001
    np.testing.assert_allclose(linearized_motion[3:], 0.0, atol=2.0e-6)
    assert np.max(np.abs(delta)) <= 0.01


def _real_contact_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    initial_object_height_m: float,
) -> OptionSubstepObservation:
    thumb_geoms = {
        "right_thumb_1_collision",
        "right_thumb_2_collision",
        "right_thumb_3_collision",
        "right_thumb_4_collision",
    }
    opposing_geoms = {
        f"right_{finger}_{segment}_collision" for finger in ("index", "middle", "ring", "little") for segment in (1, 2)
    }
    thumb_force = 0.0
    opposing_force = 0.0
    net_hand_force = 0.0
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if "cylinder_collision" not in (geom1, geom2):
            continue
        counterpart = geom2 if geom1 == "cylinder_collision" else geom1
        force_torque = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, force_torque)
        force = float(np.linalg.norm(force_torque[:3]))
        if counterpart in thumb_geoms:
            thumb_force += force
        if counterpart in opposing_geoms:
            opposing_force += force
        if counterpart in thumb_geoms | opposing_geoms:
            net_hand_force = max(net_hand_force, force)
    cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    return OptionSubstepObservation(
        thumb_side_force_n=thumb_force,
        opposing_side_force_n=opposing_force,
        net_hand_force_n=net_hand_force,
        object_lift_height_m=max(0.0, float(data.xpos[cylinder_id, 2]) - initial_object_height_m),
    )


def test_real_mujoco_runs_all_505_targets_and_exits_failure_without_bilateral_grasp(
    config: ResolvedRunConfig,
) -> None:
    model, data = _model_data(config)
    controller = _controller(config, model, data)
    arm = np.array(data.ctrl[:7], dtype=np.float64, order="C")
    cylinder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cylinder")
    initial_object_height = float(data.xpos[cylinder_id, 2])

    data.ctrl[:] = controller.begin_policy_step(arm).values.numpy()
    for _ in range(config.physics.decimation):
        mujoco.mj_step(model, data)
        controller.observe_substep(
            _real_contact_observation(model, data, initial_object_height_m=initial_object_height)
        )
    controller.finish_policy_step(trigger_requested=True, option_admissible=True)

    targets: list[np.ndarray] = []
    opposing_contact_seen = False
    for _ in range(TOTAL_SCRIPT_STEPS):
        kwargs: dict[str, LiftPreparation] = {}
        if controller.state.phase is OptionPhase.Lift:
            kwargs["lift_preparation"] = _lift_preparation(model, data)
        target = controller.begin_policy_step(np.full(7, -1.0, dtype=np.float64), **kwargs).values.numpy()
        targets.append(target.copy())
        data.ctrl[:] = target
        for _ in range(config.physics.decimation):
            mujoco.mj_step(model, data)
            observation = _real_contact_observation(
                model,
                data,
                initial_object_height_m=initial_object_height,
            )
            opposing_contact_seen |= observation.opposing_side_force_n >= CONTACT_FORCE_THRESHOLD_N
            controller.observe_substep(observation)
        controller.finish_policy_step(trigger_requested=True, option_admissible=True)

    trajectory = np.asarray(targets)
    assert trajectory.shape == (TOTAL_SCRIPT_STEPS, 13)
    assert np.isfinite(trajectory).all()
    assert np.all(trajectory >= model.actuator_ctrlrange[:, 0] - 1.0e-12)
    assert np.all(trajectory <= model.actuator_ctrlrange[:, 1] + 1.0e-12)
    assert np.max(np.abs(np.diff(trajectory[:, :7], axis=0))) <= 0.01 + 1.0e-12
    assert opposing_contact_seen
    assert not controller.state.bilateral_persistent_ever
    assert controller.state.phase is OptionPhase.Failure
    assert controller.state.script_steps_completed == TOTAL_SCRIPT_STEPS
    assert controller.state.transition_events[-1] is OptionEvent.ScriptFailure


def test_smoothstep_boundaries() -> None:
    assert smoothstep(-1.0) == 0.0
    assert smoothstep(0.0) == 0.0
    assert smoothstep(0.5) == 0.5
    assert smoothstep(1.0) == 1.0
    assert smoothstep(2.0) == 1.0
    with pytest.raises(ValueError, match="finite"):
        smoothstep(float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        OptionSubstepObservation(thumb_side_force_n=-0.01)
