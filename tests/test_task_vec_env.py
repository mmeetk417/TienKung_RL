"""M12 task-runtime and synchronous CPU VecEnv integration tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import (
    ContactFrame,
    ContactSample,
    ParallelStepResult,
    PhysicsSubstepTrace,
    is_illegal_contact,
)
from tienkung_rl.task import (
    OptionEpisodeState,
    OptionPhase,
    OutcomeEpisodeState,
    ReachThenCloseTask,
    TaskVecEnv,
    TerminationReason,
)
from tienkung_rl.task.environment import (
    TASK_EXTRAS_SCHEMA_VERSION,
    PreparedPolicyStep,
    ReachThenCloseEnvironment,
    TaskStepTransition,
)


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config()


def _empty_action() -> torch.Tensor:
    return torch.zeros(7, dtype=torch.float32)


def _make_handoff_admissible(task: ReachThenCloseTask) -> None:
    environment = task.environment
    episode = environment.episode
    state = environment._last_physics_state
    assert episode is not None
    episode.target_tcp_position_m = np.array(state.tcp_position_m, dtype=np.float64, order="C", copy=True)
    episode.target_tcp_quaternion_wxyz = np.array(
        state.tcp_quaternion_wxyz,
        dtype=np.float64,
        order="C",
        copy=True,
    )


def _controlled_result(
    task: ReachThenCloseTask,
    action: torch.Tensor,
    *,
    transient_displacement_substep: int | None = None,
    transient_illegal_contact_substep: int | None = None,
    transient_contact_force_n: float = 6.0,
) -> tuple[ParallelStepResult, PreparedPolicyStep]:
    environment = task.environment
    lift = None
    if environment.needs_lift_kinematics:
        lift = task.backend.read_arm_site_jacobian("right_tcp_site")
    prepared = environment.prepare_policy_step(action, lift_kinematics=lift)
    task.backend.apply_targets(
        np.array(prepared.physical_action.values.numpy(), dtype=np.float64, order="C", copy=True)
    )
    physics = task.backend.step_substeps(4)
    episode = environment.episode
    assert episode is not None
    state = replace(
        physics.state,
        joint_velocities_rad_s=np.zeros_like(physics.state.joint_velocities_rad_s),
        site_linear_velocities_m_s=np.zeros_like(physics.state.site_linear_velocities_m_s),
        cylinder_position_m=episode.cylinder_reset_position_m,
        cylinder_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        cylinder_linear_velocity_m_s=np.zeros(3, dtype=np.float64),
        cylinder_angular_velocity_rad_s=np.zeros(3, dtype=np.float64),
    )
    positions = np.repeat(episode.cylinder_reset_position_m[None, :], 4, axis=0)
    if transient_displacement_substep is not None:
        positions[transient_displacement_substep, 0] += task.config.task.termination.maximum_safe_displacement_m + 0.001
    quaternions = np.repeat(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), 4, axis=0)
    trace = PhysicsSubstepTrace(
        cylinder_positions_m=positions,
        cylinder_quaternions_wxyz=quaternions,
        cylinder_linear_velocities_m_s=np.zeros((4, 3), dtype=np.float64),
        cylinder_angular_velocities_rad_s=np.zeros((4, 3), dtype=np.float64),
        finite=np.ones(4, dtype=np.bool_),
        simulation_times_s=physics.substep_trace.simulation_times_s,
    )
    contact_force_n = float(transient_contact_force_n)
    contact_impulse_n_s = contact_force_n * task.config.physics.physics_dt_s
    samples: tuple[ContactSample, ...] = ()
    if transient_illegal_contact_substep is not None:
        samples = (
            ContactSample(
                substep_index=transient_illegal_contact_substep,
                physics_step=physics.diagnostics.physics_step - (3 - transient_illegal_contact_substep),
                geom1="right_forearm_collision",
                geom2="table_top_collision",
                body1="right_forearm_link",
                body2="table",
                geom1_groups=("right_arm",),
                geom2_groups=("table",),
                position_m=np.zeros(3, dtype=np.float64),
                normal_from_geom1_to_geom2=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
                force_on_geom2_world_n=np.asarray((0.0, 0.0, contact_force_n), dtype=np.float64),
                normal_force_n=contact_force_n,
                impulse_n_s=contact_impulse_n_s,
                signed_distance_m=-0.001,
                penetration_m=0.001,
            ),
        )
    contacts = ContactFrame(
        contacts=samples,
        substeps_collected=4,
        hand_link_names=physics.contacts.hand_link_names,
        hand_contact_force_norms_n=np.zeros(12, dtype=np.float64),
        maximum_contact_force_n=contact_force_n if samples else 0.0,
        maximum_contact_impulse_n_s=contact_impulse_n_s if samples else 0.0,
        maximum_illegal_contact_impulse_n_s=(
            contact_impulse_n_s
            if samples
            and is_illegal_contact(
                samples[0].geom1_groups,
                samples[0].geom2_groups,
                contact_force_n,
                right_arm_table_force_limit_n=task.config.task.termination.right_arm_table_contact_force_limit_n,
            )
            else 0.0
        ),
        contact_capacity=physics.contacts.contact_capacity,
        maximum_contacts_seen=1 if samples else 0,
        physics_step=physics.contacts.physics_step,
        simulation_time_s=physics.contacts.simulation_time_s,
    )
    result = ParallelStepResult(
        env_id=0,
        state=state,
        contacts=contacts,
        diagnostics=replace(
            physics.diagnostics,
            warnings=(),
            contact_capacity_exceeded=False,
            solver_anomaly=False,
            nonfinite_state=False,
        ),
        substeps_executed=4,
        physics_step=physics.diagnostics.physics_step,
        substep_trace=trace,
    )
    return result, prepared


def _controlled_step(
    task: ReachThenCloseTask,
    action: torch.Tensor,
    **kwargs: int | None,
) -> tuple[TaskStepTransition, PreparedPolicyStep]:
    result, prepared = _controlled_result(task, action, **kwargs)
    return task.environment.finish_policy_step(result), prepared


def test_automatic_dwell_acceptance_delayed_script_masks_and_return_factors(config: ResolvedRunConfig) -> None:
    with ReachThenCloseTask(config, seed=31) as task:
        task.reset()
        episode = task.environment.episode
        assert episode is not None
        episode.target_tcp_position_m = episode.target_tcp_position_m + np.asarray((0.5, 0.0, 0.0))
        waiting, _ = _controlled_step(task, _empty_action())
        assert not waiting.snapshot.trigger_requested
        assert not waiting.snapshot.trigger_rejected
        assert not waiting.snapshot.trigger_accepted
        assert waiting.snapshot.option_phase == OptionPhase.Reach.value
        assert waiting.actor_update_mask
        assert waiting.discount_factor == config.ppo.gamma
        assert waiting.gae_lambda_factor == config.ppo.gae_lambda

        _make_handoff_admissible(task)
        for dwell_step in range(1, 15):
            dwelling, _ = _controlled_step(task, _empty_action())
            assert dwelling.snapshot.admission_consecutive_steps == dwell_step
            assert not dwelling.snapshot.trigger_requested
        accepted, accepted_prepared = _controlled_step(task, _empty_action())
        assert accepted.admission.option_admissible
        assert accepted.snapshot.trigger_requested
        assert accepted.snapshot.trigger_accepted
        assert accepted.snapshot.option_phase == OptionPhase.ScriptPending.value
        assert accepted.snapshot.option_script_steps == 0
        assert accepted.actor_update_mask
        latched = task.environment.option_controller
        assert latched is not None and latched.state.latched_arm_targets_rad is not None

        ignored = torch.ones(7, dtype=torch.float32)
        scripted, scripted_prepared = _controlled_step(task, ignored)
        assert scripted.snapshot.option_phase == OptionPhase.Settle.value
        assert scripted.snapshot.option_script_steps == 1
        assert not scripted.snapshot.trigger_requested
        assert not scripted.actor_update_mask
        assert scripted.critic_update_mask
        assert scripted.discount_factor == 1.0
        assert scripted.gae_lambda_factor == 1.0
        np.testing.assert_allclose(
            scripted_prepared.physical_action.values.numpy()[:7],
            accepted_prepared.physical_action.values.numpy()[:7],
            rtol=0.0,
            atol=0.0,
        )


def test_second_substep_transient_illegal_contact_is_latched(
    config: ResolvedRunConfig,
) -> None:
    with ReachThenCloseTask(config, seed=32) as task:
        task.reset()
        _make_handoff_admissible(task)
        transition, _ = _controlled_step(task, _empty_action(), transient_illegal_contact_substep=1)
        assert transition.done
        assert transition.outcome.unsafe
        assert transition.outcome.primary_reason is TerminationReason.ILLEGAL_CONTACT
        assert transition.snapshot.primary_termination_reason == TerminationReason.ILLEGAL_CONTACT.value


def test_right_arm_table_contact_at_limit_is_allowed(
    config: ResolvedRunConfig,
) -> None:
    with ReachThenCloseTask(config, seed=320) as task:
        task.reset()
        _make_handoff_admissible(task)
        transition, _ = _controlled_step(
            task,
            _empty_action(),
            transient_illegal_contact_substep=1,
            transient_contact_force_n=config.task.termination.right_arm_table_contact_force_limit_n,
        )
        assert not transition.done
        assert not transition.outcome.unsafe
        assert transition.snapshot.illegal_contact_impulse_n_s == 0.0


def test_solver_anomaly_with_stalled_time_reaches_unsafe_transition(
    config: ResolvedRunConfig,
) -> None:
    with ReachThenCloseTask(config, seed=321) as task:
        task.reset()
        result, _ = _controlled_result(task, _empty_action())
        times = np.array(result.substep_trace.simulation_times_s, dtype=np.float64, copy=True)
        times[2:] = times[1]
        anomalous_trace = replace(
            result.substep_trace,
            finite=np.asarray((True, False, False, False), dtype=np.bool_),
            simulation_times_s=times,
        )
        anomalous_result = replace(
            result,
            diagnostics=replace(result.diagnostics, solver_anomaly=True),
            substep_trace=anomalous_trace,
        )

        transition = task.environment.finish_policy_step(anomalous_result)
        assert transition.done
        assert transition.outcome.unsafe
        assert transition.outcome.primary_reason is TerminationReason.NONFINITE_OR_SIMULATION_ERROR
        assert not transition.timed_out


def test_danger_terminal_snapshot_precedes_reset_and_reset_clears_all_latches(
    config: ResolvedRunConfig,
) -> None:
    with ReachThenCloseTask(config, seed=33) as task:
        task.reset()
        _make_handoff_admissible(task)
        for _ in range(15):
            accepted, _ = _controlled_step(task, _empty_action())
        assert accepted.snapshot.trigger_accepted
        terminal, _ = _controlled_step(task, _empty_action(), transient_illegal_contact_substep=1)
        assert terminal.done and terminal.snapshot.option_accepted_ever
        assert terminal.snapshot.primary_termination_reason == "illegal_contact"
        assert terminal.discount_factor == 1.0

        task.reset()
        reset = task.snapshot()
        assert reset.episode_step == 0
        assert reset.option_phase == OptionPhase.Reach.value
        assert not reset.trigger_requested
        assert not reset.trigger_accepted
        assert not reset.trigger_rejected
        assert not reset.option_accepted_ever
        assert not reset.strict_ready_ever
        assert not reset.done
        assert reset.primary_termination_reason is None


def test_snapshot_is_frozen_and_nested_mappings_are_readonly(config: ResolvedRunConfig) -> None:
    with ReachThenCloseTask(config, seed=34) as task:
        task.reset()
        snapshot = task.snapshot()
        with pytest.raises(FrozenInstanceError):
            snapshot.episode_step = 99  # type: ignore[misc]
        with pytest.raises(TypeError):
            snapshot.reward_terms["x"] = 1.0  # type: ignore[index]
        with pytest.raises(TypeError):
            snapshot.randomization["stage_at_reset"] = "S3"  # type: ignore[index]
        cylinder = snapshot.randomization["cylinder"]
        assert isinstance(cylinder, Mapping)
        with pytest.raises(TypeError):
            cylinder["radius_m"] = 99.0  # type: ignore[index]
        assert snapshot.position_error_m >= 0.0
        assert snapshot.orientation_error_rad >= 0.0
        assert snapshot.step_reward == 0.0
        assert snapshot.episode_reward_total == 0.0
        assert snapshot.strict_ready_consecutive_steps == 0
        assert snapshot.physics_step >= 0
        assert snapshot.simulation_time_s >= 0.0


def test_asset_object_relative_target_transform_is_fixed_across_sampled_resets(
    config: ResolvedRunConfig,
) -> None:
    with ReachThenCloseTask(config, seed=35) as task:
        task.reset()
        first_episode = task.environment.episode
        first_state = task.environment._last_physics_state
        relative = task.environment._object_to_target_tcp_quaternion_wxyz
        assert first_episode is not None and relative is not None
        first_target = first_episode.target_tcp_quaternion_wxyz.copy()
        first_position = first_episode.cylinder_reset_position_m.copy()
        assert not np.allclose(first_target, first_state.tcp_quaternion_wxyz, rtol=0.0, atol=1.0e-12)

        task.reset()
        second_episode = task.environment.episode
        second_state = task.environment._last_physics_state
        assert second_episode is not None
        assert not np.allclose(first_position, second_episode.cylinder_reset_position_m, rtol=0.0, atol=0.0)
        second_relative = task.environment._object_to_target_tcp_quaternion_wxyz
        assert second_relative is not None
        assert bool(np.array_equal(second_relative, relative))
        np.testing.assert_allclose(second_episode.target_tcp_quaternion_wxyz, first_target, rtol=0.0, atol=0.0)
        assert not np.allclose(
            second_episode.target_tcp_quaternion_wxyz,
            second_state.tcp_quaternion_wxyz,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_real_vec_env_cpu_shapes_preallocation_and_stable_extras(config: ResolvedRunConfig) -> None:
    with TaskVecEnv(config, num_envs=3, worker_threads=2, seed=41) as environment:
        observations, extras = environment.reset()
        observation_pointer = observations.data_ptr()
        reward_pointer = environment._rewards.data_ptr()
        assert observations.shape == (3, 106)
        assert observations.dtype == torch.float32
        assert observations.device.type == "cpu"
        assert observations.is_contiguous()
        assert environment.num_actions == 7
        assert environment.max_episode_length == 3000
        assert environment.episode_length_buf.dtype == torch.int64
        assert extras["schema_version"] == TASK_EXTRAS_SCHEMA_VERSION

        next_observations, rewards, dones, next_extras = environment.step(torch.zeros((3, 7), dtype=torch.float32))
        assert next_observations.data_ptr() == observation_pointer
        assert rewards.data_ptr() == reward_pointer
        assert rewards.shape == dones.shape == (3,)
        assert rewards.dtype == torch.float32
        assert dones.dtype == torch.bool
        assert next_extras.keys() == extras.keys()
        assert next_extras["actor_update_mask"].shape == (3,)
        assert next_extras["critic_update_mask"].all()
        assert tuple(item.env_id for item in next_extras["transitions"]) == (0, 1, 2)


def test_vec_env_asynchronous_timeout_reset_preserves_terminal_extras(config: ResolvedRunConfig) -> None:
    with TaskVecEnv(config, num_envs=3, worker_threads=2, seed=42) as environment:
        environment.reset()
        episode0 = environment.environments[0].episode
        assert episode0 is not None
        episode0.episode_step = int(environment.max_episode_length) - 1
        _, _, dones, extras = environment.step(torch.zeros((3, 7), dtype=torch.float32))
        assert bool(dones[0])
        terminal = extras["terminal_episode"]
        assert terminal[0] is not None
        assert terminal[0].episode_step == environment.max_episode_length
        assert terminal[0].primary_termination_reason == TerminationReason.TIMEOUT.value
        assert environment.episode_length_buf[0].item() == 0
        assert environment.episode_length_buf[1].item() == 1
        assert environment.episode_length_buf[2].item() == 1
        assert extras["time_outs"].tolist() == [True, False, False]
        assert extras["discount_factor"][0].item() == pytest.approx(config.ppo.gamma)


def test_timeout_flag_respects_primary_reason_priority(config: ResolvedRunConfig) -> None:
    with ReachThenCloseTask(config, seed=44) as task:
        task.reset()
        episode = task.environment.episode
        assert episode is not None
        episode.episode_step = config.task.scene.maximum_episode_steps - 1
        transition, _ = _controlled_step(
            task,
            _empty_action(),
            transient_illegal_contact_substep=1,
        )
        assert transition.done
        assert transition.outcome.primary_reason is TerminationReason.ILLEGAL_CONTACT
        assert not transition.timed_out


def _prime_script_terminal(runtime: ReachThenCloseEnvironment, *, success: bool) -> None:
    episode = runtime.episode
    controller = runtime.option_controller
    assert episode is not None and controller is not None
    arm = tuple(float(item) for item in episode.current_arm_targets_rad)
    controller.state = OptionEpisodeState(
        phase=OptionPhase.LiftHold,
        phase_step=249,
        script_steps_completed=464,
        acceptance_count=1,
        acceptance_latched=True,
        latched_arm_targets_rad=arm,
        lift_start_arm_targets_rad=arm,
        lift_final_arm_targets_rad=arm,
        bilateral_contact_steps=3,
        supported_lift_steps=5 if success else 0,
        maximum_supported_lift_steps=5 if success else 0,
        bilateral_contact_ever=True,
        bilateral_persistent_ever=True,
        supported_lift_ever=success,
        lift_5cm_event=True,
    )
    runtime.outcome_classifier._state = OutcomeEpisodeState(
        option_accepted_ever=True,
        lift_5cm_ever=True,
        bilateral_contact_steps=3,
        bilateral_persistent_ever=True,
        lift_support_steps=5 if success else 0,
        lift_support_persistent_ever=success,
    )


def _safe_result_from_current(runtime: ReachThenCloseEnvironment) -> ParallelStepResult:
    state = runtime._last_physics_state
    episode = runtime.episode
    assert episode is not None
    safe_state = replace(
        state,
        joint_velocities_rad_s=np.zeros_like(state.joint_velocities_rad_s),
        site_linear_velocities_m_s=np.zeros_like(state.site_linear_velocities_m_s),
        cylinder_position_m=episode.cylinder_reset_position_m,
        cylinder_quaternion_wxyz=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
        cylinder_linear_velocity_m_s=np.zeros(3, dtype=np.float64),
        cylinder_angular_velocity_rad_s=np.zeros(3, dtype=np.float64),
    )
    base_time = runtime._last_diagnostics.simulation_time_s
    trace = PhysicsSubstepTrace(
        cylinder_positions_m=np.repeat(episode.cylinder_reset_position_m[None, :], 4, axis=0),
        cylinder_quaternions_wxyz=np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            4,
            axis=0,
        ),
        cylinder_linear_velocities_m_s=np.zeros((4, 3), dtype=np.float64),
        cylinder_angular_velocities_rad_s=np.zeros((4, 3), dtype=np.float64),
        finite=np.ones(4, dtype=np.bool_),
        simulation_times_s=base_time + np.arange(1, 5, dtype=np.float64) * 0.005,
    )
    contacts = ContactFrame(
        contacts=(),
        substeps_collected=4,
        hand_link_names=runtime._last_contacts.hand_link_names,
        hand_contact_force_norms_n=np.zeros(12, dtype=np.float64),
        maximum_contact_force_n=0.0,
        maximum_contact_impulse_n_s=0.0,
        maximum_illegal_contact_impulse_n_s=0.0,
        contact_capacity=runtime._last_contacts.contact_capacity,
        maximum_contacts_seen=0,
        physics_step=runtime._last_diagnostics.physics_step + 4,
        simulation_time_s=base_time + 0.020,
    )
    diagnostics = replace(
        runtime._last_diagnostics,
        warnings=(),
        contact_capacity_exceeded=False,
        solver_anomaly=False,
        nonfinite_state=False,
        physics_step=runtime._last_diagnostics.physics_step + 4,
        simulation_time_s=base_time + 0.020,
    )
    return ParallelStepResult(
        env_id=runtime.env_id,
        state=safe_state,
        contacts=contacts,
        diagnostics=diagnostics,
        substeps_executed=4,
        physics_step=diagnostics.physics_step,
        substep_trace=trace,
    )


def test_vec_env_asynchronous_success_failure_timeout_and_terminal_retention(
    config: ResolvedRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with TaskVecEnv(config, num_envs=3, worker_threads=2, seed=45) as environment:
        environment.reset()
        _prime_script_terminal(environment.environments[0], success=True)
        _prime_script_terminal(environment.environments[1], success=False)
        timeout_episode = environment.environments[2].episode
        assert timeout_episode is not None
        timeout_episode.episode_step = int(environment.max_episode_length) - 1

        def controlled_step(requests: object) -> Any:
            del requests
            return SimpleNamespace(results=tuple(_safe_result_from_current(item) for item in environment.environments))

        monkeypatch.setattr(environment.executor, "step", controlled_step)

        _, _, dones, extras = environment.step(torch.zeros((3, 7), dtype=torch.float32))
        assert dones.tolist() == [True, True, True]
        terminal = extras["terminal_episode"]
        assert terminal[0].success and terminal[0].primary_termination_reason is None
        assert terminal[1].primary_termination_reason == TerminationReason.SCRIPT_FAILURE.value
        assert terminal[2].primary_termination_reason == TerminationReason.TIMEOUT.value
        assert extras["time_outs"].tolist() == [False, False, True]
        assert extras["actor_update_mask"].tolist() == [False, False, True]
        assert extras["critic_update_mask"].tolist() == [True, True, True]
        assert extras["discount_factor"].tolist() == pytest.approx([1.0, 1.0, config.ppo.gamma])
        assert environment.episode_length_buf.tolist() == [0, 0, 0]


def test_vec_env_frequent_independent_resets_remain_sorted(config: ResolvedRunConfig) -> None:
    with TaskVecEnv(config, num_envs=2, worker_threads=2, seed=43) as environment:
        environment.reset()
        for expected_done in (0, 1, 0, 1):
            episode = environment.environments[expected_done].episode
            assert episode is not None
            episode.episode_step = int(environment.max_episode_length) - 1
            _, _, dones, extras = environment.step(torch.zeros((2, 7), dtype=torch.float32))
            assert bool(dones[expected_done])
            assert tuple(item.env_id for item in extras["transitions"]) == (0, 1)
            assert extras["terminal_episode"][expected_done] is not None
            assert environment.episode_length_buf[expected_done].item() == 0


def test_direct_viewer_mode_matches_batch_env_zero_for_frozen_case(config: ResolvedRunConfig) -> None:
    seed = 51
    with (
        ReachThenCloseTask(config, seed=seed) as direct,
        TaskVecEnv(
            config,
            num_envs=1,
            worker_threads=1,
            seed=seed,
        ) as batch,
    ):
        direct_observation = direct.reset()
        batch_observation, _ = batch.reset()
        torch.testing.assert_close(direct_observation.values, batch_observation[0], rtol=0.0, atol=0.0)
        assert direct.snapshot() == batch.environments[0].snapshot()
        for _ in range(2):
            direct_step = direct.step(_empty_action())
            batch_observation, _, _, extras = batch.step(_empty_action().unsqueeze(0))
            torch.testing.assert_close(direct_step.observation.values, batch_observation[0], rtol=0.0, atol=0.0)
            assert direct_step.transition.snapshot == extras["transitions"][0].snapshot
