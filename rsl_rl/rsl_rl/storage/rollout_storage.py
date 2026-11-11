"""Strict rollout storage for the continuous reach policy.

The storage owns the boundary between environment transitions and PPO. It
keeps all critic samples, while exposing an already-filtered Actor view for
transitions on which the policy was actually in control.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Final

import torch

STORAGE_CONTRACT_VERSION: Final = "reach_then_close_continuous_rollout_storage_v2"
OPTION_EVENT_CONTRACT_VERSION: Final = "reach_then_close_option_events_v1"


class OptionPhaseCode(IntEnum):
    """Stable numeric Option phase codes stored without training-time strings."""

    REACH = 0
    SCRIPT_PENDING = 1
    SETTLE = 2
    CLOSE = 3
    HOLD = 4
    LIFT = 5
    LIFT_HOLD = 6
    SUCCESS = 7
    FAILURE = 8


_OPTION_PHASE_BY_NAME: Final = MappingProxyType(
    {
        "Reach": OptionPhaseCode.REACH,
        "ScriptPending": OptionPhaseCode.SCRIPT_PENDING,
        "Settle": OptionPhaseCode.SETTLE,
        "Close": OptionPhaseCode.CLOSE,
        "Hold": OptionPhaseCode.HOLD,
        "Lift": OptionPhaseCode.LIFT,
        "LiftHold": OptionPhaseCode.LIFT_HOLD,
        "Success": OptionPhaseCode.SUCCESS,
        "Failure": OptionPhaseCode.FAILURE,
    }
)


def option_phase_code(value: object) -> OptionPhaseCode:
    """Encode the public task phase at the environment boundary."""

    if isinstance(value, OptionPhaseCode):
        return value
    if not isinstance(value, str) or value not in _OPTION_PHASE_BY_NAME:
        raise ValueError(f"Unknown Option phase {value!r}")
    return _OPTION_PHASE_BY_NAME[value]


@dataclass(frozen=True, slots=True)
class TransitionFieldSpec:
    """Exact source-tensor contract for one transition field."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    finite: bool


@dataclass(frozen=True, slots=True)
class ActorMiniBatchView:
    """Only samples on which the Actor controlled the environment."""

    observations: torch.Tensor
    actions: torch.Tensor
    advantages: torch.Tensor
    old_actions_log_prob: torch.Tensor
    old_mu: torch.Tensor
    old_sigma: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.observations.shape[0])


@dataclass(frozen=True, slots=True)
class CriticMiniBatchView:
    """All transition samples, including fixed-script transitions."""

    observations: torch.Tensor
    target_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True, slots=True)
class RolloutMiniBatch:
    """Index-aligned PPO and diagnostic fields for one mini-batch."""

    actor: ActorMiniBatchView
    critic: CriticMiniBatchView
    actor_update_mask: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    timeouts: torch.Tensor
    discount_factors: torch.Tensor
    gae_lambda_factors: torch.Tensor
    trigger_requested: torch.Tensor
    trigger_accepted: torch.Tensor
    trigger_rejected: torch.Tensor
    option_phase_codes: torch.Tensor


class RolloutStorage:
    """Fixed-size, strict-shape feed-forward rollout buffer."""

    class Transition:
        """One vectorized environment transition before storage insertion."""

        observations: torch.Tensor | None
        privileged_observations: torch.Tensor | None
        actions: torch.Tensor | None
        rewards: torch.Tensor | None
        dones: torch.Tensor | None
        timeouts: torch.Tensor | None
        timeout_bootstrap_values: torch.Tensor | None
        values: torch.Tensor | None
        actions_log_prob: torch.Tensor | None
        action_mean: torch.Tensor | None
        action_sigma: torch.Tensor | None
        actor_update_mask: torch.Tensor | None
        discount_factor: torch.Tensor | None
        gae_lambda_factor: torch.Tensor | None
        trigger_requested: torch.Tensor | None
        trigger_accepted: torch.Tensor | None
        trigger_rejected: torch.Tensor | None
        option_phase_code: torch.Tensor | None

        def __init__(self) -> None:
            self.clear()

        def clear(self) -> None:
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.timeouts = None
            self.timeout_bootstrap_values = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.actor_update_mask = None
            self.discount_factor = None
            self.gae_lambda_factor = None
            self.trigger_requested = None
            self.trigger_accepted = None
            self.trigger_rejected = None
            self.option_phase_code = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: list[int],
        privileged_obs_shape: list[int] | None,
        actions_shape: list[int],
        device: str = "cpu",
    ) -> None:
        if num_envs <= 0 or num_transitions_per_env <= 0:
            raise ValueError("Rollout dimensions must be positive")
        if len(obs_shape) != 1 or obs_shape[0] <= 0:
            raise ValueError("Actor observation shape must be one positive dimension")
        if privileged_obs_shape is not None and (len(privileged_obs_shape) != 1 or privileged_obs_shape[0] <= 0):
            raise ValueError("Critic observation shape must be one positive dimension")
        if len(actions_shape) != 1 or actions_shape[0] <= 0:
            raise ValueError("Action shape must be one positive dimension")
        self.device = torch.device(device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = tuple(obs_shape)
        self.privileged_obs_shape = None if privileged_obs_shape is None else tuple(privileged_obs_shape)
        self.actions_shape = tuple(actions_shape)
        prefix = (num_transitions_per_env, num_envs)
        self.observations = torch.zeros((*prefix, *self.obs_shape), device=self.device, dtype=torch.float32)
        self.privileged_observations = (
            torch.zeros(
                (*prefix, *self.privileged_obs_shape),
                device=self.device,
                dtype=torch.float32,
            )
            if self.privileged_obs_shape is not None
            else None
        )
        self.rewards = torch.zeros((*prefix, 1), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros((*prefix, *self.actions_shape), device=self.device, dtype=torch.float32)
        self.dones = torch.zeros((*prefix, 1), device=self.device, dtype=torch.bool)
        self.timeouts = torch.zeros_like(self.dones)
        self.timeout_bootstrap_values = torch.zeros((*prefix, 1), device=self.device, dtype=torch.float32)
        self.values = torch.zeros((*prefix, 1), device=self.device, dtype=torch.float32)
        self.actions_log_prob = torch.zeros_like(self.values)
        self.mu = torch.zeros(
            (*prefix, *self.actions_shape),
            device=self.device,
            dtype=torch.float32,
        )
        self.sigma = torch.zeros_like(self.mu)
        self.actor_update_masks = torch.ones_like(self.dones)
        self.discount_factors = torch.zeros_like(self.values)
        self.gae_lambda_factors = torch.zeros_like(self.values)
        self.trigger_requested = torch.zeros_like(self.dones)
        self.trigger_accepted = torch.zeros_like(self.dones)
        self.trigger_rejected = torch.zeros_like(self.dones)
        self.option_phase_codes = torch.zeros((*prefix, 1), device=self.device, dtype=torch.int64)
        self.returns = torch.zeros_like(self.values)
        self.raw_advantages = torch.zeros_like(self.values)
        # Public compatibility field: this is the already-masked Actor view.
        self.advantages = torch.zeros_like(self.values)
        self.step = 0
        source_prefix = (num_envs,)
        specs = {
            "observations": TransitionFieldSpec((*source_prefix, *self.obs_shape), torch.float32, self.device, True),
            "actions": TransitionFieldSpec((*source_prefix, *self.actions_shape), torch.float32, self.device, True),
            "rewards": TransitionFieldSpec(source_prefix, torch.float32, self.device, True),
            "dones": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "timeouts": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "timeout_bootstrap_values": TransitionFieldSpec((*source_prefix, 1), torch.float32, self.device, True),
            "values": TransitionFieldSpec((*source_prefix, 1), torch.float32, self.device, True),
            "actions_log_prob": TransitionFieldSpec(source_prefix, torch.float32, self.device, True),
            "action_mean": TransitionFieldSpec(
                (*source_prefix, *self.actions_shape), torch.float32, self.device, True
            ),
            "action_sigma": TransitionFieldSpec(
                (*source_prefix, *self.actions_shape), torch.float32, self.device, True
            ),
            "actor_update_mask": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "discount_factor": TransitionFieldSpec(source_prefix, torch.float32, self.device, True),
            "gae_lambda_factor": TransitionFieldSpec(source_prefix, torch.float32, self.device, True),
            "trigger_requested": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "trigger_accepted": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "trigger_rejected": TransitionFieldSpec(source_prefix, torch.bool, self.device, False),
            "option_phase_code": TransitionFieldSpec(source_prefix, torch.int64, self.device, False),
        }
        if self.privileged_obs_shape is not None:
            specs["privileged_observations"] = TransitionFieldSpec(
                (*source_prefix, *self.privileged_obs_shape), torch.float32, self.device, True
            )
        self.transition_schema: Mapping[str, TransitionFieldSpec] = MappingProxyType(specs)

    @staticmethod
    def _require_tensor(
        name: str,
        value: torch.Tensor | None,
        spec: TransitionFieldSpec,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"transition.{name} must be a tensor")
        if tuple(value.shape) != spec.shape:
            raise ValueError(f"transition.{name} shape must be {spec.shape}, got {tuple(value.shape)}")
        if value.dtype != spec.dtype:
            raise TypeError(f"transition.{name} dtype must be {spec.dtype}, got {value.dtype}")
        if value.device != spec.device:
            raise ValueError(f"transition.{name} device must be {spec.device}, got {value.device}")
        if spec.finite and not bool(torch.isfinite(value).all()):
            raise ValueError(f"transition.{name} contains NaN or Inf")
        return value

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow; call clear() before adding transitions")
        values = {
            name: self._require_tensor(name, getattr(transition, name), spec)
            for name, spec in self.transition_schema.items()
        }
        sigma = values["action_sigma"]
        if not bool((sigma > 0.0).all()):
            raise ValueError("transition.action_sigma must be strictly positive")
        for name in ("discount_factor", "gae_lambda_factor"):
            factor = values[name]
            if not bool(((factor > 0.0) & (factor <= 1.0)).all()):
                raise ValueError(f"transition.{name} must lie in (0, 1]")
        dones = values["dones"]
        timeouts = values["timeouts"]
        if bool((timeouts & ~dones).any()):
            raise ValueError("A timeout must also be terminal")
        accepted = values["trigger_accepted"]
        rejected = values["trigger_rejected"]
        requested = values["trigger_requested"]
        if bool((accepted & rejected).any()) or bool(((accepted | rejected) & ~requested).any()):
            raise ValueError("Option accepted/rejected events must be exclusive requested events")
        phase_codes = values["option_phase_code"]
        if not bool(((phase_codes >= int(min(OptionPhaseCode))) & (phase_codes <= int(max(OptionPhaseCode)))).all()):
            raise ValueError("transition.option_phase_code is outside the stable phase enum")
        index = self.step
        self.observations[index].copy_(values["observations"])
        if self.privileged_observations is not None:
            self.privileged_observations[index].copy_(values["privileged_observations"])
        self.actions[index].copy_(values["actions"])
        self.rewards[index, :, 0].copy_(values["rewards"])
        self.dones[index, :, 0].copy_(dones)
        self.timeouts[index, :, 0].copy_(timeouts)
        self.timeout_bootstrap_values[index].copy_(values["timeout_bootstrap_values"])
        self.values[index].copy_(values["values"])
        self.actions_log_prob[index, :, 0].copy_(values["actions_log_prob"])
        self.mu[index].copy_(values["action_mean"])
        self.sigma[index].copy_(sigma)
        self.actor_update_masks[index, :, 0].copy_(values["actor_update_mask"])
        self.discount_factors[index, :, 0].copy_(values["discount_factor"])
        self.gae_lambda_factors[index, :, 0].copy_(values["gae_lambda_factor"])
        self.trigger_requested[index, :, 0].copy_(requested)
        self.trigger_accepted[index, :, 0].copy_(accepted)
        self.trigger_rejected[index, :, 0].copy_(rejected)
        self.option_phase_codes[index, :, 0].copy_(phase_codes)
        self.step += 1

    def clear(self) -> None:
        """Erase all transition and derived state before storage reuse."""

        for value in (
            self.observations,
            self.privileged_observations,
            self.rewards,
            self.actions,
            self.dones,
            self.timeouts,
            self.timeout_bootstrap_values,
            self.values,
            self.actions_log_prob,
            self.mu,
            self.sigma,
            self.actor_update_masks,
            self.discount_factors,
            self.gae_lambda_factors,
            self.trigger_requested,
            self.trigger_accepted,
            self.trigger_rejected,
            self.option_phase_codes,
            self.returns,
            self.raw_advantages,
            self.advantages,
        ):
            if value is not None:
                value.zero_()
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute GAE using stored per-transition discount/lambda factors."""

        if not (0.0 < gamma <= 1.0 and 0.0 < lam <= 1.0):
            raise ValueError("gamma and lam must lie in (0, 1]")
        expected = (self.num_envs, 1)
        if tuple(last_values.shape) != expected or last_values.dtype != torch.float32:
            raise ValueError(f"last_values must be float32 with shape {expected}")
        if last_values.device != self.device or not bool(torch.isfinite(last_values).all()):
            raise ValueError("last_values must be finite and on the storage device")
        if not bool(((self.discount_factors > 0.0) & (self.discount_factors <= 1.0)).all()):
            raise RuntimeError("Rollout has missing or invalid per-transition discount factors")
        if not bool(((self.gae_lambda_factors > 0.0) & (self.gae_lambda_factors <= 1.0)).all()):
            raise RuntimeError("Rollout has missing or invalid per-transition lambda factors")

        next_advantage = torch.zeros((self.num_envs, 1), device=self.device)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            terminal = self.dones[step]
            timeout = self.timeouts[step]
            next_is_not_terminal = ~terminal
            discount = self.discount_factors[step]
            lambda_factor = self.gae_lambda_factors[step]
            bootstrap_values = torch.where(timeout, self.timeout_bootstrap_values[step], next_values)
            bootstrap_mask = next_is_not_terminal | timeout
            delta = (
                self.rewards[step] + bootstrap_mask.to(torch.float32) * discount * bootstrap_values - self.values[step]
            )
            next_advantage = delta + next_is_not_terminal.to(torch.float32) * discount * lambda_factor * next_advantage
            self.raw_advantages[step].copy_(next_advantage)
            self.returns[step].copy_(next_advantage + self.values[step])

        if not bool(torch.isfinite(self.returns).all()):
            raise RuntimeError("Return computation produced NaN or Inf")
        mask = self.actor_update_masks
        if normalize_advantage:
            active = self.raw_advantages[mask]
            if active.numel() == 0:
                self.advantages.zero_()
            else:
                mean = active.mean()
                scale = torch.clamp(active.std(unbiased=False), min=torch.finfo(active.dtype).eps)
                self.advantages.copy_(
                    torch.where(mask, (self.raw_advantages - mean) / scale, torch.zeros_like(self.raw_advantages))
                )
        else:
            self.advantages.copy_(torch.where(mask, self.raw_advantages, torch.zeros_like(self.raw_advantages)))
        if not bool(torch.isfinite(self.advantages).all()):
            raise RuntimeError("Actor advantage view contains NaN or Inf")

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 8,
    ) -> Iterator[RolloutMiniBatch]:
        """Yield index-aligned batches with masked Actor and full Critic views."""

        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("Mini-batch and epoch counts must be positive")
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches != 0:
            raise ValueError("Rollout batch size must be divisible by mini-batch count")
        mini_batch_size = batch_size // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError("Rollout batch is smaller than the configured mini-batch count")

        observations = self.observations.flatten(0, 1)
        critic_observations = (
            self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else observations
        )
        tensors = {
            "actions": self.actions.flatten(0, 1),
            "rewards": self.rewards.flatten(0, 1),
            "dones": self.dones.flatten(0, 1),
            "timeouts": self.timeouts.flatten(0, 1),
            "values": self.values.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "raw_advantages": self.raw_advantages.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "log_prob": self.actions_log_prob.flatten(0, 1),
            "mu": self.mu.flatten(0, 1),
            "sigma": self.sigma.flatten(0, 1),
            "mask": self.actor_update_masks.flatten(0, 1),
            "discount": self.discount_factors.flatten(0, 1),
            "lambda": self.gae_lambda_factors.flatten(0, 1),
            "requested": self.trigger_requested.flatten(0, 1),
            "accepted": self.trigger_accepted.flatten(0, 1),
            "rejected": self.trigger_rejected.flatten(0, 1),
            "phase": self.option_phase_codes.flatten(0, 1),
        }
        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, requires_grad=False, device=self.device)
            for index in range(num_mini_batches):
                batch_indices = indices[index * mini_batch_size : (index + 1) * mini_batch_size]
                actor_mask = tensors["mask"][batch_indices, 0]
                actor_indices = batch_indices[actor_mask]
                yield RolloutMiniBatch(
                    actor=ActorMiniBatchView(
                        observations=observations[actor_indices],
                        actions=tensors["actions"][actor_indices],
                        advantages=tensors["advantages"][actor_indices],
                        old_actions_log_prob=tensors["log_prob"][actor_indices],
                        old_mu=tensors["mu"][actor_indices],
                        old_sigma=tensors["sigma"][actor_indices],
                    ),
                    critic=CriticMiniBatchView(
                        observations=critic_observations[batch_indices],
                        target_values=tensors["values"][batch_indices],
                        returns=tensors["returns"][batch_indices],
                        advantages=tensors["raw_advantages"][batch_indices],
                    ),
                    actor_update_mask=tensors["mask"][batch_indices],
                    actions=tensors["actions"][batch_indices],
                    rewards=tensors["rewards"][batch_indices],
                    dones=tensors["dones"][batch_indices],
                    timeouts=tensors["timeouts"][batch_indices],
                    discount_factors=tensors["discount"][batch_indices],
                    gae_lambda_factors=tensors["lambda"][batch_indices],
                    trigger_requested=tensors["requested"][batch_indices],
                    trigger_accepted=tensors["accepted"][batch_indices],
                    trigger_rejected=tensors["rejected"][batch_indices],
                    option_phase_codes=tensors["phase"][batch_indices],
                )

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return stable metadata required before restoring a rollout contract."""

        return {
            "contract_version": STORAGE_CONTRACT_VERSION,
            "option_event_contract_version": OPTION_EVENT_CONTRACT_VERSION,
            "actor_observation_shape": list(self.obs_shape),
            "critic_observation_shape": (
                None if self.privileged_obs_shape is None else list(self.privileged_obs_shape)
            ),
            "action_count": self.actions_shape[0],
            "has_per_transition_discount_factor": True,
            "has_per_transition_gae_lambda_factor": True,
            "has_actor_update_mask": True,
            "has_timeout_bootstrap": True,
            "option_phase_encoding": "int64_enum_v1",
        }

    def validate_checkpoint_metadata(self, metadata: object) -> None:
        """Fail closed for missing fields, legacy shapes, or unknown schemas."""

        if not isinstance(metadata, Mapping):
            raise RuntimeError("Checkpoint is missing rollout_storage metadata")
        actual = dict(metadata)
        expected = self.checkpoint_metadata()
        missing = sorted(set(expected) - set(actual))
        if missing:
            raise RuntimeError(f"Rollout storage checkpoint metadata is incomplete: missing={missing}")
        unknown = sorted(set(actual) - set(expected))
        if unknown:
            raise RuntimeError(f"Rollout storage checkpoint metadata has unknown fields: {unknown}")
        if actual != expected:
            raise RuntimeError(
                f"Rollout storage checkpoint contract mismatch: checkpoint={actual}, configured={expected}"
            )


__all__ = [
    "OPTION_EVENT_CONTRACT_VERSION",
    "STORAGE_CONTRACT_VERSION",
    "ActorMiniBatchView",
    "CriticMiniBatchView",
    "OptionPhaseCode",
    "RolloutMiniBatch",
    "RolloutStorage",
    "TransitionFieldSpec",
    "option_phase_code",
]
