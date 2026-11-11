"""Synchronous CPU VecEnv adapter for the single task runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import (
    EnvironmentRngManifest,
    ParallelResetRequest,
    ParallelStepRequest,
    ThreadPoolParallelExecutor,
)

from .contracts import OBSERVATION_DIMENSION, POLICY_ACTION_DIMENSION
from .curriculum import CurriculumCheckpointState, CurriculumManager
from .environment import (
    TASK_EXTRAS_SCHEMA_VERSION,
    ReachThenCloseEnvironment,
    TaskSnapshot,
    TaskStepTransition,
)
from .randomization import RandomizationRngState


class TaskVecEnv(VecEnv):
    """RSL-RL boundary with stable preallocated CPU tensor buffers."""

    def __init__(
        self,
        config: ResolvedRunConfig,
        *,
        num_envs: int | None = None,
        worker_threads: int | None = None,
        seed: int | None = None,
        executor: ThreadPoolParallelExecutor | None = None,
    ) -> None:
        config.validate()
        self.cfg = config
        self.config = config
        self.device = torch.device("cpu")
        self.num_envs = config.platform.num_envs if num_envs is None else num_envs
        workers = config.platform.worker_threads if worker_threads is None else worker_threads
        root_seed = config.run.seed if seed is None else seed
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_actions = POLICY_ACTION_DIMENSION
        self.max_episode_length = config.task.scene.maximum_episode_steps
        self.executor = (
            ThreadPoolParallelExecutor(
                config,
                num_envs=self.num_envs,
                worker_threads=workers,
                seed=root_seed,
            )
            if executor is None
            else executor
        )
        if self.executor.num_envs != self.num_envs:
            raise ValueError("executor num_envs differs from TaskVecEnv")
        self._owns_executor = executor is None
        self.curriculum = CurriculumManager(config, root_seed)
        ranges = self.executor.read_actuator_control_ranges()
        if tuple(item.env_id for item in ranges) != tuple(range(self.num_envs)):
            raise RuntimeError("executor actuator-range results are not in env_id order")
        self.environments = tuple(
            ReachThenCloseEnvironment(
                config,
                env_id=env_id,
                curriculum=self.curriculum,
                randomization_seed=root_seed + 1_000_003 * env_id,
                actuator_ranges=ranges[env_id].ranges,
            )
            for env_id in range(self.num_envs)
        )

        self._observations_numpy = np.empty((self.num_envs, OBSERVATION_DIMENSION), dtype=np.float32)
        self._rewards_numpy = np.zeros(self.num_envs, dtype=np.float32)
        self._dones_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._timeouts_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._actor_masks_numpy = np.ones(self.num_envs, dtype=np.bool_)
        self._critic_masks_numpy = np.ones(self.num_envs, dtype=np.bool_)
        self._discounts_numpy = np.full(self.num_envs, config.ppo.gamma, dtype=np.float32)
        self._lambdas_numpy = np.full(self.num_envs, config.ppo.gae_lambda, dtype=np.float32)
        self._option_admissible_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._trigger_requested_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._trigger_accepted_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._trigger_rejected_numpy = np.zeros(self.num_envs, dtype=np.bool_)
        self._episode_lengths_numpy = np.zeros(self.num_envs, dtype=np.int64)
        self._curriculum_targets_numpy = np.zeros((self.num_envs, 2), dtype=np.float32)

        self._observations = torch.from_numpy(self._observations_numpy)
        self._rewards = torch.from_numpy(self._rewards_numpy)
        self._dones = torch.from_numpy(self._dones_numpy)
        self._timeouts = torch.from_numpy(self._timeouts_numpy)
        self._actor_masks = torch.from_numpy(self._actor_masks_numpy)
        self._critic_masks = torch.from_numpy(self._critic_masks_numpy)
        self._discounts = torch.from_numpy(self._discounts_numpy)
        self._lambdas = torch.from_numpy(self._lambdas_numpy)
        self._option_admissible = torch.from_numpy(self._option_admissible_numpy)
        self._trigger_requested = torch.from_numpy(self._trigger_requested_numpy)
        self._trigger_accepted = torch.from_numpy(self._trigger_accepted_numpy)
        self._trigger_rejected = torch.from_numpy(self._trigger_rejected_numpy)
        self.episode_length_buf = torch.from_numpy(self._episode_lengths_numpy)
        self.scratch_curriculum_target_xy = torch.from_numpy(self._curriculum_targets_numpy)
        self.policy_option_configuration = MappingProxyType(
            {
                "version": config.task.option.version,
                "state_machine_version": config.task.option.state_machine_version,
                "automatic_trigger_on_admission_dwell": config.task.option.automatic_trigger_on_admission_dwell,
                "admission_dwell_steps": round(
                    config.task.admission.dwell_time_s / config.physics.policy_dt_s
                ),
                "actor_mask_while_active": config.task.option.actor_update_mask_while_active,
                "return_semantics": config.ppo.option_return_semantics,
            }
        )
        self._initialized = False
        self._last_extras: dict[str, Any] = self._base_extras(
            terminal_episodes=(None,) * self.num_envs,
            transitions=(),
        )

    def _copy_observation(self, env_id: int) -> None:
        observation = self.environments[env_id].observation
        if observation is None:
            raise RuntimeError(f"environment {env_id} has no observation")
        np.copyto(self._observations_numpy[env_id], observation.values.numpy())

    def _base_extras(
        self,
        *,
        terminal_episodes: tuple[TaskSnapshot | None, ...],
        transitions: tuple[TaskStepTransition, ...],
    ) -> dict[str, Any]:
        snapshots = tuple(
            environment.snapshot() if environment.episode is not None else None for environment in self.environments
        )
        return {
            "schema_version": TASK_EXTRAS_SCHEMA_VERSION,
            "observations": {},
            "time_outs": self._timeouts,
            "actor_update_mask": self._actor_masks,
            "critic_update_mask": self._critic_masks,
            "discount_factor": self._discounts,
            "gae_lambda_factor": self._lambdas,
            "option_metrics": {
                "option_admissible": self._option_admissible,
                "trigger_requested": self._trigger_requested,
                "trigger_accepted": self._trigger_accepted,
                "trigger_rejected": self._trigger_rejected,
            },
            "transition_events": {
                "trigger_requested": self._trigger_requested,
                "trigger_accepted": self._trigger_accepted,
                "trigger_rejected": self._trigger_rejected,
            },
            "terminal_episode": terminal_episodes,
            "admission_failure_reasons": tuple(() if item is None else item.admission_failures for item in snapshots),
            "option_phase": tuple(None if item is None else item.option_phase for item in snapshots),
            "curriculum": self.curriculum.diagnostics(),
            "randomization": tuple(None if item is None else item.randomization for item in snapshots),
            "simulation": tuple(
                None
                if item is None
                else {
                    "warning_names": item.simulation_warning_names,
                    "primary_termination_reason": item.primary_termination_reason,
                    "secondary_termination_reasons": item.secondary_termination_reasons,
                }
                for item in snapshots
            ),
            "task_snapshots": snapshots,
            "transitions": transitions,
        }

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        pending = tuple(environment.make_reset() for environment in self.environments)
        batch = self.executor.reset(
            tuple(
                ParallelResetRequest(env_id=env_id, reset_spec=item.reset_spec) for env_id, item in enumerate(pending)
            )
        )
        if tuple(item.env_id for item in batch.results) != tuple(range(self.num_envs)):
            raise RuntimeError("reset results are not in env_id order")
        for result in batch.results:
            environment = self.environments[result.env_id]
            observation = environment.complete_reset(result.state, result.diagnostics)
            np.copyto(self._observations_numpy[result.env_id], observation.values.numpy())
            self._episode_lengths_numpy[result.env_id] = 0
            episode = environment.episode
            if episode is None:
                raise RuntimeError("reset did not install episode state")
            self._curriculum_targets_numpy[result.env_id] = episode.cylinder_reset_position_m[:2]
        self._rewards_numpy.fill(0.0)
        self._dones_numpy.fill(False)
        self._timeouts_numpy.fill(False)
        self._actor_masks_numpy.fill(True)
        self._critic_masks_numpy.fill(True)
        self._discounts_numpy.fill(self.config.ppo.gamma)
        self._lambdas_numpy.fill(self.config.ppo.gae_lambda)
        self._option_admissible_numpy.fill(False)
        self._trigger_requested_numpy.fill(False)
        self._trigger_accepted_numpy.fill(False)
        self._trigger_rejected_numpy.fill(False)
        self._initialized = True
        self._last_extras = self._base_extras(
            terminal_episodes=(None,) * self.num_envs,
            transitions=(),
        )
        return self._observations, self._last_extras

    def get_observations(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self._initialized:
            return self.reset()
        self._last_extras = self._base_extras(
            terminal_episodes=(None,) * self.num_envs,
            transitions=(),
        )
        return self._observations, self._last_extras

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if not self._initialized:
            self.reset()
        if not isinstance(actions, torch.Tensor):
            raise TypeError("actions must be a torch.Tensor")
        if actions.device.type != "cpu":
            raise ValueError("actions must be on CPU")
        if actions.dtype != torch.float32:
            raise TypeError("actions must have dtype torch.float32")
        if tuple(actions.shape) != (self.num_envs, self.num_actions):
            raise ValueError(f"actions must have shape ({self.num_envs}, {self.num_actions})")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("actions contain a non-finite value")

        lift_ids = tuple(environment.env_id for environment in self.environments if environment.needs_lift_kinematics)
        lift_queries = (
            self.executor.read_arm_site_jacobian("right_tcp_site", lift_ids)
            if lift_ids
            else ()
        )
        lift_by_id = {
            item.env_id: (item.joint_positions_rad, item.site_jacobian_6x7)
            for item in lift_queries
        }
        prepared = tuple(
            environment.prepare_policy_step(
                actions[environment.env_id],
                lift_kinematics=lift_by_id.get(environment.env_id),
            )
            for environment in self.environments
        )
        batch = self.executor.step(
            tuple(
                ParallelStepRequest(
                    env_id=env_id,
                    targets_rad=np.array(item.physical_action.values.numpy(), dtype=np.float64, order="C", copy=True),
                    substeps=4,
                )
                for env_id, item in enumerate(prepared)
            )
        )
        transitions = tuple(self.environments[result.env_id].finish_policy_step(result) for result in batch.results)
        if tuple(item.env_id for item in transitions) != tuple(range(self.num_envs)):
            raise RuntimeError("task transitions are not in env_id order")

        terminal: list[TaskSnapshot | None] = [None] * self.num_envs
        done_ids: list[int] = []
        for transition in transitions:
            env_id = transition.env_id
            np.copyto(self._observations_numpy[env_id], transition.terminal_observation.values.numpy())
            self._rewards_numpy[env_id] = transition.reward
            self._dones_numpy[env_id] = transition.done
            self._timeouts_numpy[env_id] = transition.timed_out
            self._actor_masks_numpy[env_id] = transition.actor_update_mask
            self._critic_masks_numpy[env_id] = transition.critic_update_mask
            self._discounts_numpy[env_id] = transition.discount_factor
            self._lambdas_numpy[env_id] = transition.gae_lambda_factor
            self._option_admissible_numpy[env_id] = transition.admission.option_admissible
            self._trigger_requested_numpy[env_id] = transition.snapshot.trigger_requested
            self._trigger_accepted_numpy[env_id] = transition.snapshot.trigger_accepted
            self._trigger_rejected_numpy[env_id] = transition.snapshot.trigger_rejected
            self._episode_lengths_numpy[env_id] = transition.snapshot.episode_step
            if transition.done:
                terminal[env_id] = transition.snapshot
                done_ids.append(env_id)

        if done_ids:
            pending = {env_id: self.environments[env_id].make_reset() for env_id in done_ids}
            reset_batch = self.executor.reset(
                tuple(ParallelResetRequest(env_id=env_id, reset_spec=pending[env_id].reset_spec) for env_id in done_ids)
            )
            for result in reset_batch.results:
                environment = self.environments[result.env_id]
                observation = environment.complete_reset(result.state, result.diagnostics)
                np.copyto(self._observations_numpy[result.env_id], observation.values.numpy())
                self._episode_lengths_numpy[result.env_id] = 0
                episode = environment.episode
                if episode is None:
                    raise RuntimeError("automatic reset did not install episode state")
                self._curriculum_targets_numpy[result.env_id] = episode.cylinder_reset_position_m[:2]

        self._last_extras = self._base_extras(
            terminal_episodes=tuple(terminal),
            transitions=transitions,
        )
        return self._observations, self._rewards, self._dones, self._last_extras

    def close(self) -> None:
        if self._owns_executor:
            self.executor.close()

    def checkpoint_rng_state(self) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
        """Capture curriculum plus both per-environment PCG64 RNG families."""

        executor_manifest = self.executor.rng_manifest()
        return (
            self.curriculum.state().to_dict(),
            tuple(
                {
                    "env_id": env_id,
                    "randomization": self.environments[env_id].randomizer.state().to_dict(),
                    "executor": {
                        "env_id": executor_manifest[env_id].env_id,
                        "root_seed": executor_manifest[env_id].root_seed,
                        "bit_generator": executor_manifest[env_id].bit_generator,
                        "state": json.loads(executor_manifest[env_id].state_json),
                        "successful_resets": executor_manifest[env_id].successful_resets,
                    },
                }
                for env_id in range(self.num_envs)
            ),
        )

    def restore_checkpoint_rng_state(
        self,
        curriculum_state: Mapping[str, object],
        environment_states: Sequence[object],
    ) -> None:
        """Restore all sampling streams before the mandatory uniform reset."""

        if len(environment_states) != self.num_envs:
            raise RuntimeError("checkpoint environment RNG count differs from configured num_envs")
        desired_curriculum = CurriculumCheckpointState.from_dict(curriculum_state)
        desired_randomization: list[RandomizationRngState] = []
        desired_executor: list[EnvironmentRngManifest] = []
        for env_id, raw in enumerate(environment_states):
            if not isinstance(raw, Mapping) or set(raw) != {"env_id", "randomization", "executor"}:
                raise TypeError("each environment RNG checkpoint must contain env_id/randomization/executor")
            if raw["env_id"] != env_id:
                raise ValueError("environment RNG checkpoints must be in exact env_id order")
            randomization = raw["randomization"]
            executor = raw["executor"]
            if not isinstance(randomization, Mapping) or not isinstance(executor, Mapping):
                raise TypeError("environment RNG sub-states must be mappings")
            executor_fields = {"env_id", "root_seed", "bit_generator", "state", "successful_resets"}
            if set(executor) != executor_fields:
                raise ValueError("executor RNG checkpoint fields are incomplete or unknown")
            state_json = json.dumps(executor["state"], sort_keys=True, separators=(",", ":"))
            desired_randomization.append(RandomizationRngState.from_dict(randomization))
            desired_executor.append(
                EnvironmentRngManifest(
                    env_id=executor["env_id"],
                    root_seed=executor["root_seed"],
                    bit_generator=executor["bit_generator"],
                    state_json=state_json,
                    successful_resets=executor["successful_resets"],
                )
            )

        original_curriculum = self.curriculum.state()
        original_randomization = tuple(environment.randomizer.state() for environment in self.environments)
        original_executor = self.executor.rng_manifest()
        try:
            self.executor.restore_rng_manifest(desired_executor)
            for environment, state in zip(self.environments, desired_randomization, strict=True):
                environment.randomizer.restore(state)
            # Curriculum is restored last because its validator is mutation-atomic.
            self.curriculum.restore(desired_curriculum)
        except Exception as error:
            self.executor.restore_rng_manifest(original_executor)
            for environment, state in zip(self.environments, original_randomization, strict=True):
                environment.randomizer.restore(state)
            # The failed curriculum restore has not mutated its state.
            if self.curriculum.state() != original_curriculum:  # pragma: no cover - defensive contract guard
                raise RuntimeError("curriculum restore violated its atomicity contract") from error
            raise

    def discard_incomplete_episodes_and_reset_all(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Apply the checkpoint boundary shared by continued and resumed runs."""

        return self.reset()

    def __enter__(self) -> TaskVecEnv:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["TaskVecEnv"]
