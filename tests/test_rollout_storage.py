"""Continuous rollout storage and Option-return contract tests."""

from __future__ import annotations

import pytest
import torch
from rsl_rl.storage import OptionPhaseCode, RolloutStorage


def _storage() -> RolloutStorage:
    return RolloutStorage(2, 2, [3], [3], [7])


def _transition(storage: RolloutStorage) -> RolloutStorage.Transition:
    transition = RolloutStorage.Transition()
    transition.observations = torch.zeros(2, 3)
    transition.privileged_observations = torch.zeros(2, 3)
    transition.actions = torch.zeros(2, 7)
    transition.rewards = torch.zeros(2)
    transition.dones = torch.zeros(2, dtype=torch.bool)
    transition.timeouts = torch.zeros(2, dtype=torch.bool)
    transition.timeout_bootstrap_values = torch.zeros(2, 1)
    transition.values = torch.zeros(2, 1)
    transition.actions_log_prob = torch.zeros(2)
    transition.action_mean = torch.zeros(2, 7)
    transition.action_sigma = torch.ones(2, 7)
    transition.actor_update_mask = torch.ones(2, dtype=torch.bool)
    transition.discount_factor = torch.full((2,), 0.99)
    transition.gae_lambda_factor = torch.full((2,), 0.95)
    transition.trigger_requested = torch.zeros(2, dtype=torch.bool)
    transition.trigger_accepted = torch.zeros(2, dtype=torch.bool)
    transition.trigger_rejected = torch.zeros(2, dtype=torch.bool)
    transition.option_phase_code = torch.full((2,), int(OptionPhaseCode.REACH), dtype=torch.int64)
    return transition


def test_transition_schema_contains_only_current_action_and_option_fields() -> None:
    storage = _storage()
    assert storage.transition_schema["actions"].shape == (2, 7)
    assert storage.checkpoint_metadata()["action_count"] == 7


def test_add_transition_and_masked_actor_batch() -> None:
    storage = _storage()
    first = _transition(storage)
    first.actor_update_mask = torch.tensor((True, False))
    storage.add_transitions(first)
    second = _transition(storage)
    second.actor_update_mask = torch.tensor((False, True))
    storage.add_transitions(second)
    storage.compute_returns(torch.zeros(2, 1), 0.99, 0.95)
    batch = next(storage.mini_batch_generator(1, 1))
    assert batch.actor.sample_count == 2
    assert batch.actor.actions.shape == (2, 7)
    assert batch.critic.observations.shape == (4, 3)


def test_transition_validation_rejects_bad_action_shape_and_invalid_events() -> None:
    storage = _storage()
    bad_shape = _transition(storage)
    bad_shape.actions = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="actions shape"):
        storage.add_transitions(bad_shape)

    invalid_event = _transition(storage)
    invalid_event.trigger_accepted = torch.tensor((True, False))
    with pytest.raises(ValueError, match="requested events"):
        storage.add_transitions(invalid_event)


def test_timeout_uses_explicit_terminal_bootstrap() -> None:
    storage = RolloutStorage(1, 1, [1], [1], [1])
    transition = RolloutStorage.Transition()
    transition.observations = torch.zeros(1, 1)
    transition.privileged_observations = torch.zeros(1, 1)
    transition.actions = torch.zeros(1, 1)
    transition.rewards = torch.ones(1)
    transition.dones = torch.ones(1, dtype=torch.bool)
    transition.timeouts = torch.ones(1, dtype=torch.bool)
    transition.timeout_bootstrap_values = torch.full((1, 1), 2.0)
    transition.values = torch.zeros(1, 1)
    transition.actions_log_prob = torch.zeros(1)
    transition.action_mean = torch.zeros(1, 1)
    transition.action_sigma = torch.ones(1, 1)
    transition.actor_update_mask = torch.ones(1, dtype=torch.bool)
    transition.discount_factor = torch.full((1,), 0.99)
    transition.gae_lambda_factor = torch.full((1,), 0.95)
    transition.trigger_requested = torch.zeros(1, dtype=torch.bool)
    transition.trigger_accepted = torch.zeros(1, dtype=torch.bool)
    transition.trigger_rejected = torch.zeros(1, dtype=torch.bool)
    transition.option_phase_code = torch.zeros(1, dtype=torch.int64)
    storage.add_transitions(transition)
    storage.compute_returns(torch.zeros(1, 1), 0.99, 0.95)
    assert storage.returns.item() == pytest.approx(2.98)
