from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners.on_policy_runner import (
    _capture_rng_state,
    _format_duration,
    _restore_rng_state,
    _training_progress_text,
    _validate_checkpoint_action_space,
)
from rsl_rl.storage import RolloutStorage


class ReachPpoTests(unittest.TestCase):
    def _policy(self, actions: int = 3) -> ActorCritic:
        return ActorCritic(
            5,
            5,
            actions,
            actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8],
            init_noise_std=0.02,
            min_std=0.001,
            max_std=0.05,
        )

    def test_checkpoint_action_space_requires_exact_metadata(self) -> None:
        expected = {
            "contract_version": "reach_action_7_auto_trigger_v2",
            "type": "bounded_tanh_normal_auto_option",
            "action_count": 7,
        }
        _validate_checkpoint_action_space(expected, expected)
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            _validate_checkpoint_action_space(None, expected)
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            _validate_checkpoint_action_space({**expected, "action_count": 8}, expected)

    def test_runner_rng_state_round_trip_restores_all_cpu_generators(self) -> None:
        random.seed(41)
        np.random.seed(41)
        torch.manual_seed(41)
        state = _capture_rng_state()
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        _restore_rng_state(state)
        actual = (random.random(), float(np.random.random()), torch.rand(4))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_terminal_progress_panel_reports_status_metrics_and_eta(self) -> None:
        text = _training_progress_text(
            iteration=9,
            final_iteration=20,
            fps=10855,
            collection_time=3.615,
            learning_time=0.913,
            losses={
                "value_function": 0.328,
                "surrogate": -0.0111,
                "entropy": 28.0764,
                "actor_anchor": 0.0,
                "actor_anchor_weight": 0.0,
                "actor_update_enabled": 1.0,
                "actor_learning_rate": 3.0e-7,
                "critic_learning_rate": 1.0e-4,
            },
            mean_reward=-8.07,
            mean_episode_length=49.27,
            episode_metrics={"/reach_then_close/grasp_lift_success": 0.8},
            total_timesteps=491520,
            total_time=45.0,
            eta_seconds=5344.0,
            algorithm_update_count=10,
            critic_warmup_updates=5,
        )
        self.assertIn("Learning iteration 10/20", text)
        self.assertIn("Actor update", text)
        self.assertIn("ETA", text)
        self.assertEqual(_format_duration(3661.0), "01:01:01")

    def test_option_steps_propagate_terminal_reward_without_decision_decay(self) -> None:
        storage = RolloutStorage(1, 4, [1], [1], [1])
        storage.rewards[:, 0, 0] = torch.tensor((0.0, 0.0, 0.0, 10.0))
        storage.dones[3, 0, 0] = True
        storage.actor_update_masks[:, 0, 0] = torch.tensor((True, False, False, False))
        storage.discount_factors[:, 0, 0] = torch.tensor((0.99, 1.0, 1.0, 1.0))
        storage.gae_lambda_factors[:, 0, 0] = torch.tensor((0.95, 1.0, 1.0, 1.0))
        storage.compute_returns(torch.zeros(1, 1), gamma=0.99, lam=0.95, normalize_advantage=False)
        torch.testing.assert_close(
            storage.returns[:, 0, 0],
            torch.tensor((9.405, 10.0, 10.0, 10.0)),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_feed_forward_update_is_finite_and_changes_actor(self) -> None:
        torch.manual_seed(7)
        policy = self._policy()
        algorithm = PPO(policy, num_learning_epochs=2, num_mini_batches=2, learning_rate=1.0e-3)
        algorithm.init_storage(4, 4, [5], [5], [3])
        actor_before = {name: value.detach().clone() for name, value in policy.actor.state_dict().items()}
        observations = torch.randn(4, 5)
        for _ in range(4):
            actions = algorithm.act(observations, observations)
            self.assertEqual(tuple(actions.shape), (4, 3))
            algorithm.process_env_step(torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
            observations = torch.randn(4, 5)
        algorithm.compute_returns(observations)
        losses = algorithm.update()
        self.assertEqual(
            set(losses),
            {
                "value_function",
                "surrogate",
                "entropy",
                "kl",
                "log_probability",
                "actor_anchor",
                "actor_anchor_weight",
                "actor_update_enabled",
                "actor_learning_rate",
                "critic_learning_rate",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in losses.values()))
        self.assertTrue(any(not torch.equal(actor_before[name], value) for name, value in policy.actor.state_dict().items()))

    def test_all_masked_rollout_updates_only_the_critic(self) -> None:
        policy = self._policy()
        algorithm = PPO(policy, num_learning_epochs=1, num_mini_batches=1, learning_rate=1.0e-3)
        algorithm.init_storage(2, 3, [5], [5], [3])
        actor_before = {name: value.detach().clone() for name, value in policy.actor.state_dict().items()}
        critic_before = {name: value.detach().clone() for name, value in policy.critic.state_dict().items()}
        observations = torch.randn(2, 5)
        for _ in range(3):
            algorithm.act(observations, observations)
            algorithm.process_env_step(
                torch.ones(2),
                torch.zeros(2, dtype=torch.bool),
                {"actor_update_mask": torch.zeros(2, dtype=torch.bool)},
            )
            observations = torch.randn(2, 5)
        algorithm.compute_returns(observations)
        algorithm.update()
        self.assertTrue(all(torch.equal(actor_before[name], value) for name, value in policy.actor.state_dict().items()))
        self.assertTrue(any(not torch.equal(critic_before[name], value) for name, value in policy.critic.state_dict().items()))


if __name__ == "__main__":
    unittest.main()
