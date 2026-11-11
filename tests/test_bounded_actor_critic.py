"""CPU tests for the grasp policy's tanh-squashed Gaussian distribution."""

from __future__ import annotations

import copy
import math
import unittest

import torch
from rsl_rl.modules import ActorCritic


class TestBoundedActorCritic(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.policy = ActorCritic(
            num_actor_obs=5,
            num_critic_obs=5,
            num_actions=3,
            actor_hidden_dims=[16, 8],
            critic_hidden_dims=[16, 8],
            init_noise_std=0.8,
            min_std=0.05,
            max_std=1.0,
            squash_epsilon=1.0e-6,
        )

    def test_sample_and_inference_actions_are_finite_and_bounded(self) -> None:
        observations = torch.randn(2048, 5)
        sampled_actions = self.policy.act(observations)
        inference_actions = self.policy.act_inference(observations)
        action_limit = 1.0 - self.policy.squash_epsilon

        self.assertTrue(torch.isfinite(sampled_actions).all())
        self.assertTrue(torch.isfinite(inference_actions).all())
        self.assertTrue(torch.all(sampled_actions.abs() <= action_limit))
        self.assertTrue(torch.all(inference_actions.abs() <= action_limit))

    def test_actor_export_path_matches_deterministic_inference(self) -> None:
        observations = torch.randn(8, 5)
        # Isaac Lab's JIT/ONNX exporters copy policy.actor directly.
        with self.assertWarns(DeprecationWarning):
            exported_actor = torch.jit.script(copy.deepcopy(self.policy.actor))
        torch.testing.assert_close(
            exported_actor(observations),
            self.policy.act_inference(observations),
        )

    def test_log_prob_has_exact_tanh_jacobian_correction(self) -> None:
        observations = torch.randn(32, 5)
        actions = self.policy.act(observations)
        actual = self.policy.get_actions_log_prob(actions)

        scale = 1.0 - self.policy.squash_epsilon
        normalized = torch.clamp(
            actions / scale,
            -1.0 + self.policy.squash_epsilon,
            1.0 - self.policy.squash_epsilon,
        )
        latent = torch.atanh(normalized)
        log_det = math.log(scale) + torch.log1p(-torch.tanh(latent).square())
        distribution = self.policy.distribution
        self.assertIsNotNone(distribution)
        assert distribution is not None
        expected = (distribution.log_prob(latent) - log_det).sum(dim=-1)

        torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-5)

    def test_unchanged_policy_gives_unit_ppo_ratio_and_finite_gradients(self) -> None:
        observations = torch.randn(64, 5)
        actions = self.policy.act(observations).detach()
        old_log_prob = self.policy.get_actions_log_prob(actions).detach()

        self.policy.act(observations)
        new_log_prob = self.policy.get_actions_log_prob(actions)
        ratio = torch.exp(new_log_prob - old_log_prob)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))

        loss = -(new_log_prob.mean() + 0.01 * self.policy.entropy.mean())
        loss.backward()
        gradients = [parameter.grad for parameter in self.policy.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_std_bounds_are_enforced_on_parameters_and_distribution(self) -> None:
        with torch.no_grad():
            self.policy.std.fill_(20.0)
        self.policy.enforce_std_bounds()
        self.policy.update_distribution(torch.zeros(4, 5))
        torch.testing.assert_close(self.policy.action_std, torch.ones(4, 3))

        with torch.no_grad():
            self.policy.std.fill_(-5.0)
        self.policy.enforce_std_bounds()
        self.policy.update_distribution(torch.zeros(4, 5))
        torch.testing.assert_close(self.policy.action_std, torch.full((4, 3), 0.05))


if __name__ == "__main__":
    unittest.main()
