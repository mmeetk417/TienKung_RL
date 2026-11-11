# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# Modifications are licensed under the BSD-3-Clause license.

"""Bounded continuous Actor-Critic used by the seven-joint reach policy."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules.module import _IncompatibleKeys

from rsl_rl.utils import resolve_nn_activation

ACTION_COUNT = 7
ACTION_SPACE_TYPE = "bounded_tanh_normal_auto_option"
ACTION_CONTRACT_VERSION = "reach_action_7_auto_trigger_v2"


class TanhActionSquash(nn.Module):
    """Map latent Gaussian actions bijectively into a bounded action range."""

    def __init__(self, epsilon: float) -> None:
        super().__init__()
        if not 0.0 < epsilon < 1.0:
            raise ValueError(f"squash_epsilon must be in (0, 1), got {epsilon}")
        self.epsilon = epsilon
        self.scale = 1.0 - epsilon

    def forward(self, latent_actions: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(latent_actions)

    def log_abs_det_jacobian(self, latent_actions: torch.Tensor) -> torch.Tensor:
        return math.log(self.scale) + 2.0 * (
            math.log(2.0) - latent_actions - F.softplus(-2.0 * latent_actions)
        )

    def inverse(self, actions: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(actions):
            raise TypeError("Actions must use a floating-point dtype")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("Actions must be finite")
        tolerance = 10.0 * torch.finfo(actions.dtype).eps
        if bool((actions.abs() > 1.0 + tolerance).any()):
            maximum = float(actions.abs().max())
            raise ValueError(f"tanh-normal log_prob received action outside [-1, 1]: max_abs={maximum}")
        inverse_margin = max(self.epsilon, torch.finfo(actions.dtype).eps)
        normalized = torch.clamp(
            actions / self.scale,
            min=-1.0 + inverse_margin,
            max=1.0 - inverse_margin,
        )
        return torch.atanh(normalized)


class ActorCritic(nn.Module):
    """Feed-forward Actor-Critic with one bounded continuous action vector."""

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Sequence[int] | None = None,
        critic_hidden_dims: Sequence[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        min_std: float | None = None,
        max_std: float | None = None,
        squash_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        actor_hidden_dims = tuple(actor_hidden_dims or (256, 256, 256))
        critic_hidden_dims = tuple(critic_hidden_dims or (256, 256, 256))
        if num_actor_obs <= 0 or num_critic_obs <= 0 or num_actions <= 0:
            raise ValueError("Observation and action dimensions must be positive")
        if not actor_hidden_dims or not critic_hidden_dims:
            raise ValueError("Actor and critic must each have at least one hidden layer")
        if any(width <= 0 for width in (*actor_hidden_dims, *critic_hidden_dims)):
            raise ValueError("Hidden dimensions must be positive")
        if not math.isfinite(init_noise_std) or init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be finite and positive")
        if min_std is not None and (not math.isfinite(min_std) or min_std <= 0.0):
            raise ValueError("min_std must be finite and positive")
        if max_std is not None and (not math.isfinite(max_std) or max_std <= 0.0):
            raise ValueError("max_std must be finite and positive")
        if min_std is not None and max_std is not None and min_std > max_std:
            raise ValueError("min_std must not exceed max_std")

        self.num_actions = num_actions
        self.min_std = min_std
        self.max_std = max_std
        self.squash_epsilon = squash_epsilon
        squash = TanhActionSquash(squash_epsilon)

        actor_modules: list[nn.Module] = []
        actor_input = num_actor_obs
        for width in actor_hidden_dims:
            actor_modules.extend((nn.Linear(actor_input, width), resolve_nn_activation(activation)))
            actor_input = width
        actor_modules.extend((nn.Linear(actor_input, num_actions), squash))
        self.actor = nn.Sequential(*actor_modules)

        critic_modules: list[nn.Module] = []
        critic_input = num_critic_obs
        for width in critic_hidden_dims:
            critic_modules.extend((nn.Linear(critic_input, width), resolve_nn_activation(activation)))
            critic_input = width
        critic_modules.append(nn.Linear(critic_input, 1))
        self.critic = nn.Sequential(*critic_modules)

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.enforce_std_bounds()
        self.distribution: Normal | None = None
        self._last_latent_action: torch.Tensor | None = None
        Normal.set_default_validate_args(False)

    def _require_distribution(self) -> Normal:
        if self.distribution is None:
            raise RuntimeError("update_distribution() or act() must be called first")
        return self.distribution

    @property
    def action_mean(self) -> torch.Tensor:
        return self._require_distribution().mean

    @property
    def action_std(self) -> torch.Tensor:
        return self._require_distribution().stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self._last_latent_action is None:
            raise RuntimeError("act() must be called before requesting entropy")
        return (
            self._require_distribution().entropy()
            + self._log_squash_abs_det_jacobian(self._last_latent_action)
        ).sum(dim=-1)

    @property
    def deterministic_action_mean(self) -> torch.Tensor:
        return self.actor[-1](self._require_distribution().mean)

    @property
    def last_latent_action(self) -> torch.Tensor:
        if self._last_latent_action is None:
            raise RuntimeError("act() must be called before accessing latent actions")
        return self._last_latent_action

    def _actor_latent_mean(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor[:-1](observations)

    def _effective_std(self) -> torch.Tensor:
        if self.min_std is None and self.max_std is None:
            return self.std
        return torch.clamp(self.std, min=self.min_std, max=self.max_std)

    def enforce_std_bounds(self) -> None:
        if self.min_std is None and self.max_std is None:
            return
        with torch.no_grad():
            self.std.clamp_(min=self.min_std, max=self.max_std)

    def _log_squash_abs_det_jacobian(self, latent_actions: torch.Tensor) -> torch.Tensor:
        squash = self.actor[-1]
        if not isinstance(squash, TanhActionSquash):
            raise RuntimeError("Actor is missing its tanh action squash")
        return squash.log_abs_det_jacobian(latent_actions)

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self._actor_latent_mean(observations)
        std = self._effective_std().expand_as(mean)
        if not bool(torch.isfinite(mean).all()):
            raise RuntimeError("Actor produced non-finite means")
        if not bool(torch.isfinite(std).all()) or not bool((std > 0.0).all()):
            raise RuntimeError("Actor produced non-finite or non-positive std")
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **_: object) -> torch.Tensor:
        self.update_distribution(observations)
        self._last_latent_action = self._require_distribution().rsample()
        return self.actor[-1](self._last_latent_action)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != self.num_actions:
            raise ValueError(f"Expected {self.num_actions} actions, got {actions.shape[-1]}")
        squash = self.actor[-1]
        if not isinstance(squash, TanhActionSquash):
            raise RuntimeError("Actor is missing its tanh action squash")
        latent = squash.inverse(actions)
        result = (
            self._require_distribution().log_prob(latent)
            - squash.log_abs_det_jacobian(latent)
        ).sum(dim=-1)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("Action log-probability is non-finite")
        return result

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def deterministic_export_module(self) -> nn.Module:
        return copy.deepcopy(self.actor)

    def action_space_metadata(self) -> dict[str, object]:
        return {
            "contract_version": (
                ACTION_CONTRACT_VERSION if self.num_actions == ACTION_COUNT else "generic_continuous_reference_v1"
            ),
            "type": ACTION_SPACE_TYPE if self.num_actions == ACTION_COUNT else "bounded_tanh_normal",
            "action_count": self.num_actions,
        }

    def evaluate(self, critic_observations: torch.Tensor, **_: object) -> torch.Tensor:
        value = self.critic(critic_observations)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Critic produced non-finite values")
        return value

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        std = state_dict.get("std")
        if not isinstance(std, torch.Tensor) or std.shape != (self.num_actions,):
            raise RuntimeError(f"State dict must contain exactly {self.num_actions} std values")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.enforce_std_bounds()
        return result
