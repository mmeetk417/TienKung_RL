"""Feed-forward PPO for the bounded continuous reach policy."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import OptionPhaseCode, RolloutStorage, option_phase_code


def normalize_masked_advantages(
    advantages: torch.Tensor,
    actor_update_mask: torch.Tensor,
) -> torch.Tensor:
    """Normalize only Actor-eligible samples and zero all other entries."""

    if advantages.shape != actor_update_mask.shape:
        raise ValueError(
            "advantages and actor_update_mask must have identical shapes, got "
            f"{tuple(advantages.shape)} and {tuple(actor_update_mask.shape)}"
        )
    mask = actor_update_mask.to(dtype=torch.bool)
    active = advantages[mask]
    if active.numel() == 0:
        return torch.zeros_like(advantages)
    if not bool(torch.isfinite(active).all()):
        raise RuntimeError("Actor-eligible advantages contain NaN or Inf")
    mean = active.mean()
    standard_deviation = active.std(unbiased=False)
    denominator = torch.clamp(
        standard_deviation,
        min=torch.finfo(advantages.dtype).eps,
    )
    normalized = (advantages - mean) / denominator
    result = torch.where(mask, normalized, torch.zeros_like(advantages))
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Masked advantage normalization produced NaN or Inf")
    return result


def _masked_actor_mean(
    values: torch.Tensor,
    actor_update_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    mask = torch.squeeze(actor_update_mask, dim=-1).to(dtype=torch.bool)
    if values.shape != mask.shape:
        raise ValueError(f"Actor quantity shape {tuple(values.shape)} does not match mask shape {tuple(mask.shape)}")
    sample_count = int(mask.sum().item())
    if sample_count == 0:
        return values.new_zeros(()), 0
    active = values[mask]
    if not bool(torch.isfinite(active).all()):
        raise RuntimeError("Actor objective contains NaN or Inf")
    return active.mean(), sample_count


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains NaN or Inf")


class PPO:
    policy: ActorCritic

    def __init__(
        self,
        policy: ActorCritic,
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1.0e-3,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
        critic_warmup_updates: int = 0,
        actor_anchor_weight: float = 0.0,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float | None = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1

        if actor_learning_rate is None:
            actor_learning_rate = learning_rate
        if critic_learning_rate is None:
            critic_learning_rate = learning_rate
        for name, value in (
            ("actor_learning_rate", actor_learning_rate),
            ("critic_learning_rate", critic_learning_rate),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if critic_warmup_updates < 0:
            raise ValueError("critic_warmup_updates must be non-negative")
        if not math.isfinite(actor_anchor_weight) or actor_anchor_weight < 0.0:
            raise ValueError("actor_anchor_weight must be finite and non-negative")

        self.policy = policy.to(device)
        self.policy.enforce_std_bounds()
        self.actor_parameters = [*self.policy.actor.parameters(), self.policy.std]
        self.critic_parameters = list(self.policy.critic.parameters())
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.optimizer = optim.Adam(
            (
                {
                    "params": self.actor_parameters,
                    "lr": self.actor_learning_rate,
                    "name": "actor",
                },
                {
                    "params": self.critic_parameters,
                    "lr": self.critic_learning_rate,
                    "name": "critic",
                },
            )
        )
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()
        self.critic_warmup_updates = critic_warmup_updates
        self.actor_anchor_weight = actor_anchor_weight
        self.update_count = 0
        self.actor_anchor: nn.Sequential | None = None
        if self.actor_anchor_weight > 0.0:
            self.set_actor_anchor()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        # Retain the public attribute used by the runner and adaptive schedule;
        # it now denotes the Actor learning rate.
        self.learning_rate = self.actor_learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def set_actor_anchor(self) -> None:
        """Freeze the current deterministic Actor as a behavior reference."""

        self.actor_anchor = copy.deepcopy(self.policy.actor).to(self.device)
        self.actor_anchor.eval()
        for parameter in self.actor_anchor.parameters():
            parameter.requires_grad_(False)

    def get_checkpoint_state(self) -> dict[str, object]:
        """Return PPO state not represented by policy or optimizer tensors."""

        return {
            "schema_version": "continuous_ppo_runtime_state_v2",
            "update_count": self.update_count,
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_anchor_state_dict": (self.actor_anchor.state_dict() if self.actor_anchor is not None else None),
        }

    def load_checkpoint_state(self, state: dict[str, object]) -> None:
        """Restore warmup progress and the original Actor behavior anchor."""

        required = {
            "schema_version",
            "update_count",
            "actor_learning_rate",
            "critic_learning_rate",
            "actor_anchor_state_dict",
        }
        if set(state) != required:
            raise RuntimeError("PPO checkpoint state fields are incomplete or unknown")
        if state["schema_version"] != "continuous_ppo_runtime_state_v2":
            raise RuntimeError("Unknown PPO checkpoint state schema")
        update_count = state["update_count"]
        if not isinstance(update_count, int) or isinstance(update_count, bool) or update_count < 0:
            raise RuntimeError("Checkpoint update_count must be a non-negative integer")
        self.update_count = update_count

        def checkpoint_rate(name: str) -> float:
            value = state[name]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise RuntimeError(f"Checkpoint {name} must be finite and positive")
            return float(value)

        self.actor_learning_rate = checkpoint_rate("actor_learning_rate")
        self.critic_learning_rate = checkpoint_rate("critic_learning_rate")
        self.learning_rate = self.actor_learning_rate
        for parameter_group in self.optimizer.param_groups:
            group_name = parameter_group.get("name")
            if group_name == "actor":
                parameter_group["lr"] = self.actor_learning_rate
            elif group_name == "critic":
                parameter_group["lr"] = self.critic_learning_rate
            else:
                raise RuntimeError("PPO optimizer checkpoint has an unknown parameter group")
        anchor_state = state["actor_anchor_state_dict"]
        if anchor_state is not None:
            if not isinstance(anchor_state, Mapping):
                raise RuntimeError("Actor anchor checkpoint state must be a mapping")
            self.set_actor_anchor()
            if self.actor_anchor is None:
                raise RuntimeError("Failed to construct Actor anchor")
            self.actor_anchor.load_state_dict(anchor_state, strict=True)

    def _actor_anchor_loss(
        self,
        observations: torch.Tensor,
        actor_update_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.actor_anchor is None:
            raise RuntimeError("actor_anchor_weight is positive but no Actor anchor is set")
        with torch.no_grad():
            anchor_actions = self.actor_anchor(observations)
        current_actions = self.policy.act_inference(observations)
        per_sample_loss = torch.mean(
            torch.square(current_actions - anchor_actions),
            dim=-1,
        )
        actor_mask = torch.squeeze(actor_update_mask, dim=-1).to(per_sample_loss.dtype)
        actor_sample_count = torch.clamp(actor_mask.sum(), min=1.0)
        return (per_sample_loss * actor_mask).sum() / actor_sample_count

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: list[int],
        critic_obs_shape: list[int],
        actions_shape: list[int],
    ) -> None:
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            self.device,
        )

    def act(self, observations: torch.Tensor, critic_observations: torch.Tensor) -> torch.Tensor:
        actions = self.policy.act(observations).detach()
        transition = self.transition
        transition.actions = actions
        transition.values = self.policy.evaluate(critic_observations).detach()
        transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
        transition.action_mean = self.policy.action_mean.detach()
        transition.action_sigma = self.policy.action_std.detach()
        transition.observations = observations
        transition.privileged_observations = critic_observations
        return actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict[str, Any],
    ) -> None:
        if self.storage is None:
            raise RuntimeError("PPO storage has not been initialized")
        transition = self.transition
        transition.rewards = rewards.clone()
        transition.dones = dones.to(device=self.device, dtype=torch.bool)
        actor_update_mask = infos.get("actor_update_mask")
        if actor_update_mask is None:
            actor_update_mask = torch.ones_like(dones, dtype=torch.bool)
        else:
            actor_update_mask = actor_update_mask.to(
                device=self.device,
                dtype=torch.bool,
            )
        transition.actor_update_mask = actor_update_mask

        discount_factor = infos.get("discount_factor")
        if discount_factor is None:
            discount_factor = torch.where(
                actor_update_mask,
                torch.full_like(rewards, self.gamma),
                torch.ones_like(rewards),
            )
        discount_factor = discount_factor.to(self.device)
        transition.discount_factor = discount_factor

        gae_lambda_factor = infos.get("gae_lambda_factor")
        if gae_lambda_factor is None:
            gae_lambda_factor = torch.where(
                actor_update_mask,
                torch.full_like(rewards, self.lam),
                torch.ones_like(rewards),
            )
        gae_lambda_factor = gae_lambda_factor.to(self.device)
        transition.gae_lambda_factor = gae_lambda_factor
        time_outs = infos.get("time_outs")
        if time_outs is None:
            time_outs = torch.zeros_like(dones, dtype=torch.bool, device=self.device)
        elif not isinstance(time_outs, torch.Tensor):
            raise TypeError("infos['time_outs'] must be a tensor")
        else:
            time_outs = time_outs.to(device=self.device, dtype=torch.bool)
        transition.timeouts = time_outs
        timeout_bootstrap_values = infos.get("timeout_bootstrap_values")
        if timeout_bootstrap_values is None:
            if bool(time_outs.any()):
                raise RuntimeError("Timeout transitions require V(s_terminal) in infos['timeout_bootstrap_values']")
            timeout_bootstrap_values = torch.zeros((rewards.shape[0], 1), device=self.device, dtype=rewards.dtype)
        elif not isinstance(timeout_bootstrap_values, torch.Tensor):
            raise TypeError("infos['timeout_bootstrap_values'] must be a tensor")
        else:
            timeout_bootstrap_values = timeout_bootstrap_values.to(self.device)
        transition.timeout_bootstrap_values = timeout_bootstrap_values

        events = infos.get("transition_events", {})
        if not isinstance(events, Mapping):
            raise TypeError("infos['transition_events'] must be a mapping")
        for field in ("trigger_requested", "trigger_accepted", "trigger_rejected"):
            event = events.get(field)
            if event is None:
                event = torch.zeros_like(dones, dtype=torch.bool, device=self.device)
            elif not isinstance(event, torch.Tensor):
                raise TypeError(f"infos['transition_events']['{field}'] must be a tensor")
            else:
                event = event.to(device=self.device, dtype=torch.bool)
            setattr(transition, field, event)
        phase_codes = infos.get("option_phase_code")
        if phase_codes is None:
            phases = infos.get("option_phase")
            if phases is None:
                phase_codes = torch.full(
                    (rewards.shape[0],),
                    int(OptionPhaseCode.REACH),
                    device=self.device,
                    dtype=torch.int64,
                )
            else:
                if not isinstance(phases, (tuple, list)) or len(phases) != rewards.shape[0]:
                    raise TypeError("infos['option_phase'] must be one phase per environment")
                phase_codes = torch.tensor(
                    [int(option_phase_code(phase)) for phase in phases],
                    device=self.device,
                    dtype=torch.int64,
                )
        elif not isinstance(phase_codes, torch.Tensor):
            raise TypeError("infos['option_phase_code'] must be a tensor")
        else:
            phase_codes = phase_codes.to(device=self.device, dtype=torch.int64)
        transition.option_phase_code = phase_codes
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_observations: torch.Tensor) -> None:
        if self.storage is None:
            raise RuntimeError("PPO storage has not been initialized")
        last_values = self.policy.evaluate(last_critic_observations).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self) -> dict[str, float]:
        if self.storage is None:
            raise RuntimeError("PPO storage has not been initialized")
        actor_update_enabled = self.update_count >= self.critic_warmup_updates
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_actor_anchor_loss = 0.0
        mean_kl = 0.0
        mean_log_probability = 0.0
        actor_metric_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            actor = batch.actor
            critic = batch.critic
            advantages = actor.advantages
            if self.normalize_advantage_per_mini_batch and actor.sample_count > 0:
                with torch.no_grad():
                    advantages = normalize_masked_advantages(
                        advantages,
                        torch.ones_like(advantages, dtype=torch.bool),
                    )

            values = self.policy.evaluate(critic.observations)
            _require_finite("Critic values", values)
            _require_finite("Returns", critic.returns)
            actor_sample_count = actor.sample_count
            actor_update_for_batch = actor_update_enabled and actor_sample_count > 0

            actions_log_prob = torch.empty(0, device=self.device)
            entropy = torch.empty(0, device=self.device)
            policy_kl = torch.empty(0, device=self.device)
            mu = actor.old_mu
            sigma = actor.old_sigma
            if actor_sample_count > 0:
                self.policy.act(actor.observations)
                actions_log_prob = self.policy.get_actions_log_prob(actor.actions)
                mu = self.policy.action_mean
                sigma = self.policy.action_std
                entropy = self.policy.entropy
                _require_finite("Current log-probability", actions_log_prob)
                _require_finite("Old log-probability", actor.old_actions_log_prob)
                with torch.inference_mode():
                    policy_kl = (
                        torch.log(sigma / actor.old_sigma)
                        + (torch.square(actor.old_sigma) + torch.square(actor.old_mu - mu))
                        / (2.0 * torch.square(sigma))
                        - 0.5
                    ).sum(dim=-1)
                    _require_finite("Policy KL", policy_kl)

            if actor_update_for_batch and self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    _require_finite("Adaptive KL", policy_kl)
                    kl_mean = policy_kl.mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.actor_learning_rate = max(
                                1.0e-8,
                                self.actor_learning_rate / 1.5,
                            )
                        elif 0.0 < kl_mean < self.desired_kl / 2.0:
                            self.actor_learning_rate = min(
                                1.0e-2,
                                self.actor_learning_rate * 1.5,
                            )
                    if self.is_multi_gpu:
                        learning_rate = torch.tensor(
                            self.actor_learning_rate,
                            device=self.device,
                        )
                        torch.distributed.broadcast(learning_rate, src=0)
                        self.actor_learning_rate = learning_rate.item()
                    self.learning_rate = self.actor_learning_rate
                    for parameter_group in self.optimizer.param_groups:
                        if parameter_group.get("name") == "actor":
                            parameter_group["lr"] = self.actor_learning_rate

            surrogate_loss = torch.zeros((), device=self.device)
            entropy_loss = torch.zeros((), device=self.device)
            if actor_sample_count > 0:
                log_ratio = actions_log_prob - torch.squeeze(actor.old_actions_log_prob)
                _require_finite("PPO log-ratio", log_ratio)
                maximum_log_ratio = -math.log(torch.finfo(log_ratio.dtype).eps)
                ratio = torch.exp(
                    torch.clamp(
                        log_ratio,
                        min=-maximum_log_ratio,
                        max=maximum_log_ratio,
                    )
                )
                surrogate = -torch.squeeze(advantages) * ratio
                surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
                entropy_loss = entropy.mean()

            if self.use_clipped_value_loss:
                values_clipped = critic.target_values + (values - critic.target_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - critic.returns).pow(2)
                value_losses_clipped = (values_clipped - critic.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (critic.returns - values).pow(2).mean()
            _require_finite("Critic loss", value_loss)

            actor_anchor_loss = torch.zeros((), device=self.device)
            loss = self.value_loss_coef * value_loss
            if actor_update_for_batch:
                if self.actor_anchor_weight > 0.0:
                    actor_anchor_loss = self._actor_anchor_loss(
                        actor.observations,
                        torch.ones(
                            (actor_sample_count, 1),
                            dtype=torch.bool,
                            device=self.device,
                        ),
                    )
                loss = (
                    loss
                    + surrogate_loss
                    - self.entropy_coef * entropy_loss
                    + self.actor_anchor_weight * actor_anchor_loss
                )
            _require_finite("Total PPO loss", loss)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            parameters_to_check = [*self.critic_parameters]
            if actor_update_for_batch:
                parameters_to_check.extend(self.actor_parameters)
            for parameter in parameters_to_check:
                if parameter.grad is not None:
                    _require_finite("Optimizer gradient", parameter.grad)
            if actor_update_for_batch:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor_parameters,
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                _require_finite("Actor gradient norm", actor_grad_norm)
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic_parameters,
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            _require_finite("Critic gradient norm", critic_grad_norm)
            self.optimizer.step()
            self.policy.enforce_std_bounds()
            for parameter in parameters_to_check:
                _require_finite("Optimizer parameter", parameter)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_loss.item()
            mean_actor_anchor_loss += actor_anchor_loss.item()
            if actor_sample_count > 0:
                mean_kl += policy_kl.mean().item()
                mean_log_probability += actions_log_prob.mean().item()
                actor_metric_updates += 1

        num_updates = self.num_learning_epochs * self.num_mini_batches
        actor_denominator = max(actor_metric_updates, 1)
        self.storage.clear()
        self.update_count += 1
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / actor_denominator,
            "log_probability": mean_log_probability / actor_denominator,
            "actor_anchor": mean_actor_anchor_loss / num_updates,
            "actor_anchor_weight": self.actor_anchor_weight,
            "actor_update_enabled": float(actor_update_enabled),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }

    def broadcast_parameters(self) -> None:
        model_parameters = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_parameters, src=0)
        self.policy.load_state_dict(model_parameters[0])

    def reduce_parameters(self) -> None:
        gradients = [parameter.grad.view(-1) for parameter in self.policy.parameters() if parameter.grad is not None]
        all_gradients = torch.cat(gradients)
        torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
        all_gradients /= self.gpu_world_size
        offset = 0
        for parameter in self.policy.parameters():
            if parameter.grad is None:
                continue
            size = parameter.numel()
            parameter.grad.data.copy_(all_gradients[offset : offset + size].view_as(parameter.grad.data))
            offset += size
