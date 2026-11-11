"""M07 action/observation contract and extracted Isaac-reference tests."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from tienkung_rl.config.load import load_run_config
from tienkung_rl.sim.mujoco_backend import MujocoEnvironmentBackend
from tienkung_rl.sim.state import ContactFrame, CylinderParameters, PhysicsState, ResetSpec
from tienkung_rl.task.actions import (
    ArmTargetController,
    assemble_reach_physical_action,
    normalized_arm_targets,
    policy_arm_action,
    rate_limit_arm_targets,
)
from tienkung_rl.task.contracts import (
    ACTIVE_HAND_JOINT_NAMES,
    ARM_JOINT_NAMES,
    HAND_CONTACT_LINK_NAMES,
    MAXIMUM_ARM_TARGET_STEP_RAD,
    OBSERVATION_BLOCK_SLICES,
    OBSERVATION_BLOCKS,
    OBSERVATION_DIMENSION,
    OBSERVED_JOINT_NAMES,
    PHYSICAL_ACTION_NAMES,
    POLICY_ACTION_NAMES,
    ArmPolicyAction,
    Observation106,
    PhysicalAction,
)
from tienkung_rl.task.observations import (
    ObservationInputs,
    ObservationNonFiniteError,
    build_observation,
    build_observation_blocks,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_FIXTURE = ROOT / "tests/fixtures/isaac_106_reference.json"


@pytest.fixture(scope="module")
def config():  # type: ignore[no-untyped-def]
    return load_run_config(ROOT / "configs/reach_then_close.yaml")


def _state(*, target_sign: float = 1.0) -> PhysicsState:
    del target_sign
    sqrt_half = np.sqrt(0.5)
    return PhysicsState(
        joint_names=OBSERVED_JOINT_NAMES,
        joint_positions_rad=np.arange(19, dtype=np.float64) / 10.0,
        joint_velocities_rad_s=-np.arange(19, dtype=np.float64) / 5.0,
        body_names=("cylinder",),
        body_positions_m=np.asarray([[1.6, 2.4, 3.55]], dtype=np.float64),
        body_quaternions_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        site_names=(),
        site_positions_m=np.empty((0, 3), dtype=np.float64),
        site_quaternions_wxyz=np.empty((0, 4), dtype=np.float64),
        site_linear_velocities_m_s=np.empty((0, 3), dtype=np.float64),
        site_angular_velocities_rad_s=np.empty((0, 3), dtype=np.float64),
        tcp_position_m=np.asarray([1.5, 2.5, 3.4], dtype=np.float64),
        tcp_quaternion_wxyz=np.asarray([sqrt_half, 0.0, 0.0, sqrt_half], dtype=np.float64),
        grasp_center_position_m=np.asarray([1.5, 2.5, 3.5], dtype=np.float64),
        grasp_center_quaternion_wxyz=np.asarray([sqrt_half, 0.0, 0.0, sqrt_half], dtype=np.float64),
        cylinder_position_m=np.asarray([1.6, 2.4, 3.55], dtype=np.float64),
        cylinder_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        cylinder_linear_velocity_m_s=np.asarray([0.4, -0.2, 0.1], dtype=np.float64),
        cylinder_angular_velocity_rad_s=np.asarray([0.2, 0.4, -0.8], dtype=np.float64),
        actuator_names=tuple(f"{name}_position" for name in PHYSICAL_ACTION_NAMES),
        actuator_targets_rad=np.arange(13, dtype=np.float64) / 10.0,
        actuator_forces_n_m=np.zeros(13, dtype=np.float64),
        cylinder_parameters=CylinderParameters(),
        episode_physics_steps=4,
        total_physics_steps=4,
        simulation_time_s=0.02,
        total_simulation_time_s=0.02,
    )


def _contacts() -> ContactFrame:
    forces = np.arange(1, 13, dtype=np.float64)
    return ContactFrame(
        contacts=(),
        substeps_collected=4,
        hand_link_names=HAND_CONTACT_LINK_NAMES,
        hand_contact_force_norms_n=forces,
        maximum_contact_force_n=12.0,
        maximum_contact_impulse_n_s=0.0,
        maximum_illegal_contact_impulse_n_s=0.0,
        contact_capacity=256,
        maximum_contacts_seen=0,
        physics_step=4,
        simulation_time_s=0.02,
    )


def _physical_action() -> PhysicalAction:
    return PhysicalAction(torch.arange(13, dtype=torch.float64) / 10.0)


def _inputs(*, target_sign: float = -1.0) -> ObservationInputs:
    return ObservationInputs(
        environment_origin_m=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
        grasp_center_linear_velocity_m_s=np.asarray([0.1, -0.1, 0.0], dtype=np.float64),
        target_tcp_position_m=np.asarray([1.51, 2.48, 3.4], dtype=np.float64),
        target_tcp_quaternion_wxyz=np.asarray([target_sign, 0.0, 0.0, 0.0], dtype=np.float64),
        grasp_center_offset_tcp_m=np.asarray([0.0, 0.0, 0.1], dtype=np.float64),
        option_active=1.0,
        option_progress=0.25,
        option_support_progress=0.75,
    )


def test_frozen_names_dimensions_versions_and_slices(config) -> None:  # type: ignore[no-untyped-def]
    assert len(OBSERVED_JOINT_NAMES) == 19
    assert OBSERVED_JOINT_NAMES[:7] == ARM_JOINT_NAMES
    assert len(OBSERVATION_BLOCKS) == 17
    assert sum(block.dimension for block in OBSERVATION_BLOCKS) == 106
    assert ARM_JOINT_NAMES == POLICY_ACTION_NAMES
    assert PHYSICAL_ACTION_NAMES == ARM_JOINT_NAMES + ACTIVE_HAND_JOINT_NAMES
    assert len(POLICY_ACTION_NAMES) == 7
    assert len(PHYSICAL_ACTION_NAMES) == 13
    assert tuple(OBSERVATION_BLOCK_SLICES) == tuple(block.name for block in OBSERVATION_BLOCKS)
    assert config.task.action.version == "reach_action_7_auto_trigger_v2"
    assert config.task.observation.version == "reach_observation_106_v1"


@pytest.mark.parametrize(
    ("factory", "value", "error"),
    (
        (ArmPolicyAction, torch.zeros(6), ValueError),
        (ArmPolicyAction, torch.zeros(7, dtype=torch.float64), TypeError),
        (ArmPolicyAction, torch.full((7,), torch.nan), ValueError),
        (PhysicalAction, torch.zeros(13, dtype=torch.float32), TypeError),
        (Observation106, torch.zeros(105), ValueError),
    ),
)
def test_contract_structures_fail_closed(factory, value, error) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(error):
        factory(value)


def test_contract_structures_reject_non_cpu_device() -> None:
    with pytest.raises(ValueError, match="must be on CPU"):
        ArmPolicyAction(torch.empty(7, dtype=torch.float32, device="meta"))


def test_policy_arm_action_requires_exact_7d_float32_cpu_finite() -> None:
    arm = policy_arm_action(torch.zeros(7, dtype=torch.float32))
    assert arm.values.shape == (7,)
    for bad in (
        torch.zeros(6),
        torch.zeros(8),
        torch.zeros(9),
        torch.zeros(7, dtype=torch.float64),
        torch.tensor([0.0] * 6 + [float("nan")], dtype=torch.float32),
    ):
        with pytest.raises((TypeError, ValueError)):
            policy_arm_action(bad)


def test_normalized_absolute_targets_hit_both_joint_endpoints(config) -> None:  # type: ignore[no-untyped-def]
    action_config = config.task.action
    lower = normalized_arm_targets(ArmPolicyAction(torch.full((7,), -1.0)), action_config)
    upper = normalized_arm_targets(ArmPolicyAction(torch.full((7,), 1.0)), action_config)
    center = np.asarray(action_config.center_joint_positions_rad, dtype=np.float64)
    half_range = np.asarray(action_config.half_range_rad, dtype=np.float64)
    np.testing.assert_array_equal(lower, center - half_range)
    np.testing.assert_array_equal(upper, center + half_range)
    with pytest.raises(ValueError, match="outside"):
        normalized_arm_targets(ArmPolicyAction(torch.full((7,), 1.0001)), action_config)


def test_rate_limit_signed_deltas_and_reset_first_step(config) -> None:  # type: ignore[no-untyped-def]
    previous = np.zeros(7, dtype=np.float64)
    desired = np.asarray([1.0, -1.0, 0.009, -0.009, 0.01, -0.01, 0.0], dtype=np.float64)
    limited = rate_limit_arm_targets(desired, previous)
    np.testing.assert_array_equal(
        limited,
        np.asarray([0.01, -0.01, 0.009, -0.009, 0.01, -0.01, 0.0], dtype=np.float64),
    )
    controller = ArmTargetController(config.task.action)
    measured = np.asarray(config.task.action.center_joint_positions_rad, dtype=np.float64) + 0.1
    with pytest.raises(RuntimeError, match="must be reset"):
        controller.step(ArmPolicyAction(torch.zeros(7)))
    controller.reset(measured)
    first = controller.step(ArmPolicyAction(torch.zeros(7)))
    assert np.all(np.abs(first - measured) <= MAXIMUM_ARM_TARGET_STEP_RAD + 1e-15)


def test_reach_physical_action_has_no_trigger_path() -> None:
    arm = np.arange(7, dtype=np.float64)
    hand = np.arange(6, dtype=np.float64) + 10.0
    physical = assemble_reach_physical_action(arm, hand)
    assert tuple(inspect.signature(assemble_reach_physical_action).parameters) == (
        "arm_targets_rad",
        "pregrasp_hand_targets_rad",
    )
    assert physical.values.shape == (13,)
    torch.testing.assert_close(physical.arm_targets_rad, torch.from_numpy(arm))
    torch.testing.assert_close(physical.hand_targets_rad, torch.from_numpy(hand))
    assert "trigger" not in PHYSICAL_ACTION_NAMES


def test_all_17_blocks_match_read_only_isaac_reference(config) -> None:  # type: ignore[no-untyped-def]
    fixture = json.loads(REFERENCE_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["source"] == (
        "legacy/isaaclab/source/legged_lab/envs/manipulation/reach_then_close_base_env.py:1592"
    )
    blocks = build_observation_blocks(
        state=_state(),
        contacts=_contacts(),
        physical_action=_physical_action(),
        inputs=_inputs(),
        config=config.task.observation,
    )
    frozen_contracts = [
        [
            block.name,
            block.dimension,
            block.scale,
            block.frame,
            block.quaternion_order,
            block.canonicalize_scalar_nonnegative,
        ]
        for block in OBSERVATION_BLOCKS
    ]
    assert frozen_contracts == fixture["block_contracts"]
    assert tuple(name for name, _ in blocks) == tuple(fixture["block_order"])
    for name, actual in blocks:
        expected = np.asarray(fixture["blocks"][name], dtype=np.float64)
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12, err_msg=name)


def test_observation_is_106_float32_cpu_history_one_and_clipped(config) -> None:  # type: ignore[no-untyped-def]
    contacts = _contacts()
    mutable_forces = contacts.hand_contact_force_norms_n.copy()
    mutable_forces[0] = 1_000.0
    contacts = replace(contacts, hand_contact_force_norms_n=mutable_forces, maximum_contact_force_n=1_000.0)
    observation = build_observation(
        state=_state(),
        contacts=contacts,
        physical_action=_physical_action(),
        inputs=_inputs(),
        config=config.task.observation,
    )
    assert observation.values.shape == (OBSERVATION_DIMENSION,)
    assert observation.values.dtype == torch.float32
    assert observation.values.device.type == "cpu"
    assert observation.history_length == 1
    assert bool(torch.isfinite(observation.values).all())
    assert float(observation.values.max()) <= 10.0
    assert float(observation.values.min()) >= -10.0
    force_slice = OBSERVATION_BLOCK_SLICES["right_hand_contact_force_norms"]
    assert float(observation.values[force_slice][0]) == 10.0


def test_relative_target_quaternion_is_wxyz_scalar_nonnegative_and_q_sign_invariant(config) -> None:  # type: ignore[no-untyped-def]
    kwargs = dict(
        state=_state(),
        contacts=_contacts(),
        physical_action=_physical_action(),
        config=config.task.observation,
    )
    positive = dict(build_observation_blocks(inputs=_inputs(target_sign=1.0), **kwargs))
    negative = dict(build_observation_blocks(inputs=_inputs(target_sign=-1.0), **kwargs))
    name = "current_tcp_relative_target_quaternion"
    np.testing.assert_allclose(positive[name], negative[name], rtol=0.0, atol=1e-12)
    assert positive[name][0] >= 0.0


def test_world_positions_subtract_origin_but_world_difference_does_not(config) -> None:  # type: ignore[no-untyped-def]
    blocks = dict(
        build_observation_blocks(
            state=_state(),
            contacts=_contacts(),
            physical_action=_physical_action(),
            inputs=_inputs(),
            config=config.task.observation,
        )
    )
    np.testing.assert_allclose(blocks["grasp_center_world_position"], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(blocks["cylinder_world_position"], [0.6, 0.4, 0.55])
    np.testing.assert_allclose(blocks["cylinder_to_grasp_center_position"], [0.1, -0.1, 0.05])


def test_nonfinite_input_returns_field_locatable_safe_termination(config) -> None:  # type: ignore[no-untyped-def]
    inputs = _inputs()
    velocity = inputs.grasp_center_linear_velocity_m_s.copy()
    velocity[1] = np.nan
    inputs = replace(inputs, grasp_center_linear_velocity_m_s=velocity)
    with pytest.raises(ObservationNonFiniteError) as caught:
        build_observation(
            state=_state(),
            contacts=_contacts(),
            physical_action=_physical_action(),
            inputs=inputs,
            config=config.task.observation,
        )
    assert caught.value.field_name == "grasp_center_linear_velocity_m_s"
    assert caught.value.indices == (1,)
    assert caught.value.safe_termination_reason == ("nonfinite_observation:grasp_center_linear_velocity_m_s")


def test_real_mujoco_state_and_contact_mapping_builds_finite_observation(config) -> None:  # type: ignore[no-untyped-def]
    with MujocoEnvironmentBackend(config) as backend:
        state = backend.reset(ResetSpec())
        contacts = backend.read_contacts()
        physical = PhysicalAction(torch.from_numpy(state.actuator_targets_rad.copy()))
        observation = build_observation(
            state=state,
            contacts=contacts,
            physical_action=physical,
            inputs=ObservationInputs(
                environment_origin_m=np.zeros(3, dtype=np.float64),
                grasp_center_linear_velocity_m_s=np.zeros(3, dtype=np.float64),
                target_tcp_position_m=state.tcp_position_m.copy(),
                target_tcp_quaternion_wxyz=state.tcp_quaternion_wxyz.copy(),
                grasp_center_offset_tcp_m=np.asarray([0.0, 0.0, 0.1], dtype=np.float64),
                option_active=0.0,
                option_progress=0.0,
                option_support_progress=0.0,
            ),
            config=config.task.observation,
        )
    assert state.joint_names == OBSERVED_JOINT_NAMES
    assert contacts.hand_link_names == HAND_CONTACT_LINK_NAMES
    assert observation.values.shape == (106,)
    assert bool(torch.isfinite(observation.values).all())
