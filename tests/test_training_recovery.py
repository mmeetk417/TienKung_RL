"""M15 training, atomic checkpoint, and complete recovery acceptance tests."""

from __future__ import annotations

import argparse
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import sha256_fingerprint
from tienkung_rl.learning import runner
from tienkung_rl.learning.checkpoint import (
    CheckpointError,
    build_sidecar_metadata,
    load_checkpoint_preflight,
    save_checkpoint,
)
from tienkung_rl.learning.runner import (
    RunnerState,
    build_training_components,
    collect_rollout,
    effective_config,
    seed_all,
    terminal_bootstrap_values,
)
from tienkung_rl.task import TaskVecEnv
from tienkung_rl.telemetry.metrics import PPOUpdateEvent, TelemetryAggregator


def _small_config(tmp_path: Path, *, episode_steps: int = 1200) -> ResolvedRunConfig:
    config = load_run_config()
    config = replace(
        config,
        platform=replace(config.platform, num_envs=2, worker_threads=2),
        task=replace(config.task, scene=replace(config.task.scene, maximum_episode_steps=episode_steps)),
        ppo=replace(
            config.ppo,
            rollout_steps_per_env=2,
            learning_epochs=1,
            mini_batches=1,
            network=replace(
                config.ppo.network,
                actor_hidden_dimensions=(16,),
                critic_hidden_dimensions=(16,),
            ),
        ),
        run=replace(
            config.run,
            output_dir=str(tmp_path),
            maximum_iterations=2,
            save_interval_iterations=1,
        ),
    )
    config.validate()
    return config


def _record_update(telemetry: TelemetryAggregator, losses: Mapping[str, float]) -> None:
    telemetry.record_ppo_update(
        PPOUpdateEvent(
            iteration=1,
            policy_loss=losses["surrogate"],
            value_loss=losses["value_function"],
            entropy=losses["entropy"],
            kl=0.0,
            total_loss=losses["surrogate"] + losses["value_function"],
            gradient_norm=1.0,
            gradients_finite=True,
            log_probability=losses["log_probability"],
        )
    )


def _save_one_update(
    path: Path,
    config: ResolvedRunConfig,
    environment: TaskVecEnv,
    components: runner.TrainingComponents,
    telemetry: TelemetryAggregator,
) -> RunnerState:
    observations, _ = environment.reset()
    collect_rollout(environment, components.ppo, observations, iteration=1, telemetry=telemetry)
    completed_episodes = int(components.ppo.storage.dones.sum().item()) if components.ppo.storage else 0
    losses = components.ppo.update()
    _record_update(telemetry, losses)
    actor_norm = runner.gradient_diagnostics(components.policy)
    state = RunnerState(
        completed_iterations=1,
        total_policy_steps=4,
        total_physics_steps=16,
        completed_episodes=completed_episodes,
        sampling_seconds=0.25,
        learning_seconds=0.10,
        total_seconds=0.35,
        last_losses={name: float(value) for name, value in losses.items()},
        last_actor_gradient_norm=actor_norm,
    )
    storage = components.ppo.storage
    assert storage is not None
    curriculum, environment_rng = environment.checkpoint_rng_state()

    sidecar = build_sidecar_metadata(
        config,
        iteration=1,
        kind="periodic",
        run_id="m15-test-run",
        dependencies={"test_runtime": "m15"},
        storage_metadata=storage.checkpoint_metadata(),
    )
    save_checkpoint(
        path,
        sidecar_metadata=sidecar,
        model_state_dict=components.policy.state_dict(),
        optimizer_state_dict=components.ppo.optimizer.state_dict(),
        ppo_state_dict=components.ppo.get_checkpoint_state(),
        rollout_storage_metadata=storage.checkpoint_metadata(),
        policy_action_space=components.policy.action_space_metadata(),
        runner_state_dict=state.to_dict(),
        telemetry_state_dict=telemetry.state_dict(),
        process_rng_state_dict=runner.capture_process_rng_state(),
        curriculum_state_dict=curriculum,
        environment_rng_state_dicts=environment_rng,
    )
    return state


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert isinstance(right, Sequence) and not isinstance(right, (str, bytes))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_effective_cli_values_are_visible_in_sidecar(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    args = argparse.Namespace(max_iterations=7, save_interval=3)
    effective = effective_config(config, args)
    assert effective.run.maximum_iterations == 7
    assert effective.run.save_interval_iterations == 3
    assert sha256_fingerprint(effective) != sha256_fingerprint(config)
    components = build_training_components(effective)
    storage = components.ppo.storage
    assert storage is not None
    sidecar = build_sidecar_metadata(
        effective,
        iteration=1,
        kind="periodic",
        run_id="effective-values",
        dependencies={"test": "runtime"},
        storage_metadata=storage.checkpoint_metadata(),
    )
    training = sidecar["training"]
    assert isinstance(training, Mapping)
    assert training["maximum_iterations"] == 7
    assert training["save_interval_iterations"] == 3


def test_timeout_bootstrap_uses_batched_terminal_observations_not_pre_step_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(tmp_path, episode_steps=1)
    seed_all(9)
    components = build_training_components(config)
    captured: dict[str, object] = {}
    original = components.ppo.process_env_step

    def recording_process(rewards: torch.Tensor, dones: torch.Tensor, infos: dict[str, Any]) -> None:
        captured["values"] = infos["timeout_bootstrap_values"].clone()
        captured["transitions"] = infos["transitions"]
        captured["timeouts"] = infos["time_outs"].clone()
        original(rewards, dones, infos)

    monkeypatch.setattr(components.ppo, "process_env_step", recording_process)
    with TaskVecEnv(config) as environment:
        observations, _ = environment.reset()
        collect_rollout(
            environment,
            components.ppo,
            observations,
            iteration=1,
            telemetry=TelemetryAggregator(),
        )
    transitions = captured["transitions"]
    assert isinstance(transitions, tuple)
    expected = terminal_bootstrap_values(components.policy, transitions)
    actual = captured["values"]
    timeouts = captured["timeouts"]
    assert isinstance(actual, torch.Tensor) and isinstance(timeouts, torch.Tensor)
    assert bool(timeouts.all())
    assert torch.equal(actual, expected)
    storage = components.ppo.storage
    assert storage is not None
    assert not torch.allclose(actual, storage.values[0], rtol=0.0, atol=1.0e-7)


def test_save_continue_and_save_restore_continue_are_exact_at_discard_boundary(tmp_path: Path) -> None:
    config = _small_config(tmp_path, episode_steps=2)
    checkpoint = tmp_path / "checkpoints" / "model_000001.pt"

    seed_all(123)
    original_components = build_training_components(config)
    original_telemetry = TelemetryAggregator()
    with TaskVecEnv(config) as original_environment:
        expected_state = _save_one_update(
            checkpoint,
            config,
            original_environment,
            original_components,
            original_telemetry,
        )
        continued_observations, _ = original_environment.discard_incomplete_episodes_and_reset_all()
        continued_reset_observations = continued_observations.clone()
        continued_samples = (random.random(), float(np.random.random()), torch.rand(4))
        continued_observations = collect_rollout(
            original_environment,
            original_components.ppo,
            continued_observations,
            iteration=2,
            telemetry=original_telemetry,
        )
        continued_losses = original_components.ppo.update()
        _record_update(original_telemetry, continued_losses)
        continued_environment_rng = original_environment.checkpoint_rng_state()
        continued_process_rng = runner.capture_process_rng_state()

        validated = load_checkpoint_preflight(
            checkpoint,
            config,
        )
        seed_all(999)
        restored_components = build_training_components(config)
        restored_telemetry = TelemetryAggregator()
        runner._preflight_resume_components(validated, restored_components)
        with TaskVecEnv(config) as restored_environment:
            restored_state = runner._restore_training_state(
                validated,
                restored_components,
                restored_environment,
                restored_telemetry,
            )
            restored_observations, _ = restored_environment.discard_incomplete_episodes_and_reset_all()
            restored_samples = (random.random(), float(np.random.random()), torch.rand(4))
            assert torch.equal(continued_reset_observations, restored_observations)
            restored_observations = collect_rollout(
                restored_environment,
                restored_components.ppo,
                restored_observations,
                iteration=2,
                telemetry=restored_telemetry,
            )
            restored_losses = restored_components.ppo.update()
            _record_update(restored_telemetry, restored_losses)
            restored_environment_rng = restored_environment.checkpoint_rng_state()
            restored_process_rng = runner.capture_process_rng_state()

        assert restored_state.to_dict() == expected_state.to_dict()
        assert continued_samples[:2] == restored_samples[:2]
        assert torch.equal(continued_samples[2], restored_samples[2])
        assert torch.equal(continued_observations, restored_observations)
        assert continued_losses == restored_losses
        assert original_telemetry.state_dict() == restored_telemetry.state_dict()
        _assert_nested_equal(continued_environment_rng, restored_environment_rng)
        _assert_nested_equal(continued_process_rng, restored_process_rng)
        _assert_nested_equal(original_components.policy.state_dict(), restored_components.policy.state_dict())
        _assert_nested_equal(
            original_components.ppo.optimizer.state_dict(),
            restored_components.ppo.optimizer.state_dict(),
        )
        assert original_components.ppo.get_checkpoint_state() == restored_components.ppo.get_checkpoint_state()
        # Every environment state contains both M06 and M05 RNG contracts.
        for item in continued_environment_rng[1]:
            assert set(item) == {"env_id", "randomization", "executor"}
            executor = item["executor"]
            assert isinstance(executor, Mapping)
            assert executor["bit_generator"] == "PCG64"


def test_invalid_resume_fails_before_task_vec_env_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_config(tmp_path)
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a checkpoint")
    invalid.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    constructed = False

    def forbidden_environment(*args: object, **kwargs: object) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("environment construction must follow checkpoint preflight")

    monkeypatch.setattr(runner, "TaskVecEnv", forbidden_environment)
    with pytest.raises(CheckpointError):
        runner.train(config, resume_path=invalid)
    assert not constructed


def test_small_real_ppo_update_has_finite_arm_gradients(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    seed_all(29)
    components = build_training_components(config)
    telemetry = TelemetryAggregator()
    with TaskVecEnv(config) as environment:
        observations, _ = environment.reset()
        collect_rollout(environment, components.ppo, observations, iteration=1, telemetry=telemetry)
        losses = components.ppo.update()
    assert all(np.isfinite(value) for value in losses.values())
    actor_norm = runner.gradient_diagnostics(components.policy)
    assert math_is_finite_positive(actor_norm)
    assert components.ppo.update_count == 1


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def test_checkpoint_is_self_contained_and_ignores_optional_json(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    components = build_training_components(config)
    checkpoint = tmp_path / "model_000001.pt"
    with TaskVecEnv(config) as environment:
        _save_one_update(checkpoint, config, environment, components, TelemetryAggregator())
    checkpoint.with_suffix(".json").unlink()
    loaded = load_checkpoint_preflight(checkpoint, config)
    assert loaded.iteration == 1
    assert loaded.sidecar["run_id"] == loaded.payload["run_id"]
    assert "payload_sha256" not in loaded.sidecar
    assert "bundle_id" not in loaded.payload
    checkpoint.with_suffix(".json").write_text("invalid JSON", encoding="utf-8")
    assert load_checkpoint_preflight(checkpoint, config).iteration == 1


def test_completed_run_can_resume_and_replace_final_checkpoint(tmp_path: Path) -> None:
    config = _small_config(tmp_path)
    assert runner.train(config).completed_iterations == 2
    checkpoint = tmp_path / "checkpoints" / "model_final.pt"
    resumed_config = replace(config, run=replace(config.run, maximum_iterations=3))
    assert runner.train(resumed_config, resume_path=checkpoint).completed_iterations == 3
    assert load_checkpoint_preflight(checkpoint, resumed_config).iteration == 3

    from tienkung_rl import evaluate
    from tienkung_rl.learning.manifests import ManifestKind, generate_manifest, write_manifest

    evaluation_config = replace(
        resumed_config,
        task=replace(resumed_config.task, scene=replace(resumed_config.task.scene, maximum_episode_steps=4)),
    )
    manifest_path = tmp_path / "validation.json"
    write_manifest(
        manifest_path,
        generate_manifest(evaluation_config, kind=ManifestKind.VALIDATION, generation_seed=123, case_count=2),
        evaluation_config,
    )
    args = evaluate.build_parser().parse_args(["--checkpoint", str(checkpoint), "--manifest", str(manifest_path)])
    from tienkung_rl.learning.evaluation import run as run_evaluation

    assert run_evaluation(evaluation_config, args) == 0
    assert list((tmp_path / "evaluation").glob("*.json"))
