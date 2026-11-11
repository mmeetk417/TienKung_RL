"""M16 frozen-manifest, deterministic replay, reporting, and gate tests."""

from __future__ import annotations

import csv
import json
import runpy
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest import mock

import numpy as np
import pytest
import torch
from rsl_rl.modules import ActorCritic

from tienkung_rl import evaluate
from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json
from tienkung_rl.learning.checkpoint import CheckpointError, ValidatedCheckpoint
from tienkung_rl.learning.evaluation import (
    EpisodeEvaluation,
    EvaluationSummary,
    MetricRatio,
    aggregate_episodes,
    deterministic_policy_action,
    final_gate,
    preflight_evaluation_inputs,
    preliminary_gate,
    record_final_manifest_usage,
    replay_case,
    write_evaluation_reports,
)
from tienkung_rl.learning.manifests import (
    ManifestKind,
    bundled_manifest_path,
    generate_manifest,
    load_manifest,
    write_manifest,
)
from tienkung_rl.task.curriculum import ForcedTargetAuthority, sample_curriculum_target
from tienkung_rl.task.environment import ReachThenCloseTask, TaskResetOverrides

_BASE_CONFIG = load_run_config()


def _config(*, episode_steps: int = 1200) -> ResolvedRunConfig:
    config = _BASE_CONFIG
    config = replace(
        config,
        platform=replace(config.platform, num_envs=1, worker_threads=1),
        task=replace(config.task, scene=replace(config.task.scene, maximum_episode_steps=episode_steps)),
    )
    config.validate()
    return config


def _policy(config: ResolvedRunConfig) -> ActorCritic:
    network = config.ppo.network
    return ActorCritic(
        num_actor_obs=106,
        num_critic_obs=106,
        num_actions=7,
        actor_hidden_dims=network.actor_hidden_dimensions,
        critic_hidden_dims=network.critic_hidden_dimensions,
        activation=network.activation,
    )


def _episode(
    case_id: str,
    *,
    training_seed: int = 42,
    accepted: bool = False,
    lifted: bool = False,
    success: bool = False,
    unsafe: bool = False,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        training_seed=training_seed,
        evaluation_seed=1001,
        case_id=case_id,
        target_x_m=0.5,
        target_y_m=-0.2,
        radius_m=0.015,
        height_m=0.100,
        mass_kg=0.115,
        hand_sliding_friction=0.9,
        table_sliding_friction=0.9,
        trigger_requested=accepted,
        option_accepted=accepted,
        lift_5cm=lifted,
        success=success,
        unsafe=unsafe,
        primary_reason="timeout" if not success else None,
        auxiliary_reasons=(),
        final_position_error_m=0.01,
        final_orientation_error_rad=0.1,
        episode_steps=1,
    )


def _metric(value: float, *, denominator: int = 256) -> MetricRatio:
    return MetricRatio(value, value * denominator, denominator)


def _summary(
    kind: ManifestKind,
    seed: int,
    *,
    accepted: float,
    lifted: float,
    success: float,
    dangerous: float,
    manifest_hash: str = "a" * 64,
    complete: bool = True,
) -> EvaluationSummary:
    count = 256 if complete else 255
    return EvaluationSummary(
        training_seed=seed,
        manifest_kind=kind,
        manifest_hash=manifest_hash,
        completed_episodes=count,
        expected_episodes=256,
        option_accepted_rate=_metric(accepted, denominator=count),
        lift_5cm_rate=_metric(lifted, denominator=count),
        complete_success_rate=_metric(success, denominator=count),
        dangerous_rate=_metric(dangerous, denominator=count),
        success_given_accepted=_metric(0.0, denominator=0),
        unsafe_given_accepted=_metric(0.0, denominator=0),
    )


@pytest.mark.parametrize(
    "kind,count",
    ((ManifestKind.VALIDATION, 17), (ManifestKind.FINAL_STANDARD, 256), (ManifestKind.FINAL_RANDOMIZED, 256)),
)
def test_manifest_generation_is_byte_deterministic_strict_and_hashed(
    tmp_path: Path,
    kind: ManifestKind,
    count: int,
) -> None:
    config = _config()
    first = generate_manifest(config, kind=kind, generation_seed=991, case_count=count)
    second = generate_manifest(config, kind=kind, generation_seed=991, case_count=count)
    assert canonical_json(first.to_dict()) == canonical_json(second.to_dict())
    assert len({case.case_id for case in first.cases}) == count
    components = tuple(case.position_component for case in first.cases)
    assert components.count("full_range") == round(0.70 * count)
    if count - round(0.70 * count) >= 3:
        assert {"boundary_corner_0", "boundary_corner_1", "boundary_corner_2"}.issubset(components)
    path = write_manifest(tmp_path / f"{kind.value}.json", first, config)
    assert load_manifest(path, config) == first
    if kind is ManifestKind.FINAL_STANDARD:
        assert {
            (
                case.radius_m,
                case.height_m,
                case.mass_kg,
                case.hand_sliding_friction,
                case.table_sliding_friction,
            )
            for case in first.cases
        } == {(0.015, 0.100, 0.115, 0.9, 0.9)}
    if kind is ManifestKind.FINAL_RANDOMIZED:
        columns = tuple(
            tuple(getattr(case, field) for case in first.cases)
            for field in (
                "radius_m",
                "height_m",
                "mass_kg",
                "hand_sliding_friction",
                "table_sliding_friction",
            )
        )
        assert all(len(set(column)) == 256 for column in columns)


def test_validation_is_independent_and_case_edits_can_be_loaded(tmp_path: Path) -> None:
    config = _config()
    validation = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=101, case_count=64)
    final = generate_manifest(config, kind=ManifestKind.FINAL_RANDOMIZED, generation_seed=202)
    assert {case.case_id for case in validation.cases}.isdisjoint(case.case_id for case in final.cases)
    path = write_manifest(tmp_path / "validation.json", validation, config)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["cases"][0]["cylinder"]["mass_kg"] = 0.116
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = load_manifest(path, config)
    assert loaded.cases[0].mass_kg == 0.116


@pytest.mark.parametrize("kind", tuple(ManifestKind))
def test_every_bundled_manifest_target_is_accepted_by_runtime_curriculum(kind: ManifestKind) -> None:
    config = _config()
    manifest = load_manifest(bundled_manifest_path(kind), config)
    source = np.random.Generator(np.random.PCG64(1))

    for case in manifest.cases:
        placement = sample_curriculum_target(
            config.curriculum,
            "S3",
            source,
            forced_xy_m=case.target_xy_m,
            authority=ForcedTargetAuthority.FROZEN_MANIFEST,
        )
        assert placement.target_xy_m == case.target_xy_m


def test_evaluation_preflight_allows_asset_edits_and_checks_checkpoint(tmp_path: Path) -> None:
    config = _config()
    manifest = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=212, case_count=2)
    path = write_manifest(tmp_path / "manifest.json", manifest, config)
    changed = replace(
        config,
        asset=replace(config.asset, runtime_mjcf_sha256="f" * 64),
    )
    changed.validate()
    with pytest.raises(CheckpointError, match="cannot deserialize checkpoint"):
        preflight_evaluation_inputs(
            changed,
            checkpoint_path=tmp_path / "never-opened.pt",
            manifest_path=path,
        )


def test_cli_requires_explicit_checkpoint_and_manifest_and_preserves_paths() -> None:
    parser = evaluate.build_parser()
    with mock.patch.object(evaluate, "load_config", return_value=_config()):
        with pytest.raises(SystemExit):
            evaluate.main([])
        with pytest.raises(SystemExit):
            evaluate.main(["--checkpoint", "model.pt"])
    args = parser.parse_args(["--checkpoint", "model.pt", "--manifest", "frozen.json"])
    assert args.checkpoint == Path("model.pt")
    assert args.manifest == Path("frozen.json")


def test_deterministic_policy_uses_tanh_mean_for_seven_arm_joints() -> None:
    config = _config()
    policy = _policy(config)
    observation = torch.linspace(-1.0, 1.0, 106)
    with torch.no_grad():
        latent = policy.actor[:-1](observation.unsqueeze(0))
        expected_continuous = policy.actor[-1](latent).squeeze(0)
    action = deterministic_policy_action(policy, observation)
    assert torch.equal(action, expected_continuous)
    assert action.shape == (7,)


def test_manual_aggregate_preserves_zero_denominators_and_all_episode_denominators() -> None:
    config = _config()
    manifest = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=303, case_count=2)
    episodes = tuple(_episode(case.case_id) for case in manifest.cases)
    summary = aggregate_episodes(episodes, manifest=manifest, training_seed=42)
    assert summary.option_accepted_rate == MetricRatio(0.0, 0, 2)
    assert summary.success_given_accepted == MetricRatio(0.0, 0, 0)
    assert summary.unsafe_given_accepted == MetricRatio(0.0, 0, 0)


def test_preliminary_gate_exact_threshold_and_per_seed_danger_override() -> None:
    config = _config()
    standard = tuple(
        _summary(
            ManifestKind.FINAL_STANDARD,
            seed,
            accepted=0.45,
            lifted=0.39,
            success=0.35,
            dangerous=0.10,
        )
        for seed in (42, 43, 44)
    )
    randomized = tuple(
        _summary(
            ManifestKind.FINAL_RANDOMIZED,
            seed,
            accepted=0.0,
            lifted=0.0,
            success=0.0,
            dangerous=1.0,
        )
        for seed in (42, 43, 44)
    )
    assert preliminary_gate(standard, randomized, config).passed
    dangerous = (
        replace(standard[0], dangerous_rate=_metric(0.11)),
        replace(standard[1], dangerous_rate=_metric(0.095)),
        replace(standard[2], dangerous_rate=_metric(0.095)),
    )
    result = preliminary_gate(dangerous, randomized, config)
    assert not result.passed
    assert any("seed 42" in failure for failure in result.datasets[0].failures)


def test_preliminary_rejects_randomized_hash_drift_and_final_rejects_one_set_failure() -> None:
    config = _config()
    passing_standard = tuple(
        _summary(
            ManifestKind.FINAL_STANDARD,
            seed,
            accepted=0.90,
            lifted=0.85,
            success=0.75,
            dangerous=0.05,
        )
        for seed in (42, 43, 44)
    )
    passing_randomized = tuple(
        _summary(
            ManifestKind.FINAL_RANDOMIZED,
            seed,
            accepted=0.90,
            lifted=0.85,
            success=0.75,
            dangerous=0.05,
        )
        for seed in (42, 43, 44)
    )
    drifted = (passing_randomized[0], replace(passing_randomized[1], manifest_hash="b" * 64), passing_randomized[2])
    assert not preliminary_gate(passing_standard, drifted, config).passed
    failed_random = tuple(replace(item, complete_success_rate=_metric(0.69)) for item in passing_randomized)
    result = final_gate(passing_standard, failed_random, config)
    assert not result.passed
    assert result.datasets[0].passed
    assert not result.datasets[1].passed


def test_final_usage_audit_warns_on_repeat_and_validation_is_not_recorded(tmp_path: Path) -> None:
    config = _config()
    final = generate_manifest(config, kind=ManifestKind.FINAL_STANDARD, generation_seed=404)
    validation = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=405, case_count=2)
    path = tmp_path / "usage.jsonl"
    assert (
        record_final_manifest_usage(
            path,
            run_id="dev-run",
            manifest=validation,
            checkpoint_bundle_id="bundle-v",
            command="validate",
        )
        is None
    )
    assert not path.exists()
    first = record_final_manifest_usage(
        path,
        run_id="dev-run",
        manifest=final,
        checkpoint_bundle_id="bundle-a",
        command="first",
    )
    second = record_final_manifest_usage(
        path,
        run_id="dev-run",
        manifest=final,
        checkpoint_bundle_id="bundle-b",
        command="second",
    )
    assert first is None
    assert second is not None and "FINAL MANIFEST AUDIT WARNING" in second
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [item["prior_consumptions_for_run_and_manifest"] for item in records] == [0, 1]


def test_frozen_case_replay_uses_manifest_values_shared_runtime_and_no_curriculum_update() -> None:
    config = _config(episode_steps=1)
    manifest = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=505, case_count=1)
    case = manifest.cases[0]
    policy = _policy(config)
    with ReachThenCloseTask(
        config,
        seed=case.seed,
        reset_overrides=TaskResetOverrides(),
        curriculum_updates_enabled=False,
    ) as task:
        before = task.curriculum.state()
        result = replay_case(task, policy, case, training_seed=42)
        repeated = replay_case(task, policy, case, training_seed=42)
        after = task.curriculum.state()
    assert result.case_id == case.case_id
    assert result.evaluation_seed == case.seed
    assert result.target_x_m == case.target_xy_m[0]
    assert result.mass_kg == case.mass_kg
    assert repeated == result
    assert after.window == before.window
    assert after.completed_windows == before.completed_windows


def test_reports_keep_every_episode_and_exact_csv_schema(tmp_path: Path) -> None:
    config = _config()
    manifest = generate_manifest(config, kind=ManifestKind.VALIDATION, generation_seed=606, case_count=2)
    episodes = tuple(_episode(case.case_id, unsafe=index == 1) for index, case in enumerate(manifest.cases))
    summary = aggregate_episodes(episodes, manifest=manifest, training_seed=42)

    class _Checkpoint:
        def __init__(self) -> None:
            self.sidecar = {"payload_sha256": "a" * 64, "bundle_id": "b" * 64}

    paths = write_evaluation_reports(
        tmp_path,
        command="python -m tienkung_rl.evaluate --checkpoint c.pt --manifest m.json",
        checkpoint=cast(ValidatedCheckpoint, _Checkpoint()),
        checkpoint_path="c.pt",
        manifest=manifest,
        manifest_path=write_manifest(tmp_path / "manifest.json", manifest, config),
        episodes=episodes,
        summary=summary,
    )
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert len(payload["episodes"]) == 2
    assert payload["formal_gate_status"] == "not_evaluated_requires_three_complete_training_seeds"
    with paths[1].open(encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    assert len(rows) == 2
    assert "auxiliary_reasons" not in rows[0]
    assert json.loads(rows[0]["auxiliary_reasons_json"]) == []
    assert "unsafe=True" in paths[2].read_text(encoding="utf-8")


def test_frozen_repository_manifests_are_present_valid_and_disjoint() -> None:
    config = _config()
    root = Path(__file__).resolve().parents[1] / "evaluation" / "manifests"
    validation = load_manifest(root / "validation.json", config)
    standard = load_manifest(root / "final_standard_256.json", config)
    randomized = load_manifest(root / "final_randomized_256.json", config)
    assert validation.kind is ManifestKind.VALIDATION
    assert standard.kind is ManifestKind.FINAL_STANDARD
    assert randomized.kind is ManifestKind.FINAL_RANDOMIZED
    assert len(standard.cases) == len(randomized.cases) == 256
    case_groups: Sequence[set[str]] = tuple(
        {case.case_id for case in item.cases} for item in (validation, standard, randomized)
    )
    assert case_groups[0].isdisjoint(case_groups[1])
    assert case_groups[0].isdisjoint(case_groups[2])
    assert case_groups[1].isdisjoint(case_groups[2])
    validation_physics = {(case.target_xy_m, case.radius_m, case.height_m, case.mass_kg) for case in validation.cases}
    standard_physics = {(case.target_xy_m, case.radius_m, case.height_m, case.mass_kg) for case in standard.cases}
    randomized_physics = {(case.target_xy_m, case.radius_m, case.height_m, case.mass_kg) for case in randomized.cases}
    assert validation_physics.isdisjoint(standard_physics)
    assert validation_physics.isdisjoint(randomized_physics)
    for kind, seed, count, repository_path in (
        (ManifestKind.VALIDATION, 160016, 64, root / "validation.json"),
        (ManifestKind.FINAL_STANDARD, 160031, 256, root / "final_standard_256.json"),
        (ManifestKind.FINAL_RANDOMIZED, 160047, 256, root / "final_randomized_256.json"),
    ):
        bundled = bundled_manifest_path(kind)
        assert bundled.read_bytes() == repository_path.read_bytes()
        assert load_manifest(bundled, config).kind is kind
        regenerated = generate_manifest(config, kind=kind, generation_seed=seed, case_count=count)
        # Frozen cases stay reproducible across edits to informational metadata.
        assert [case._payload() for case in load_manifest(repository_path, config).cases] == [
            case._payload() for case in regenerated.cases
        ]
    setup_source = (Path(__file__).resolve().parents[1] / "setup.py").read_text(encoding="utf-8")
    assert 'Path("tienkung_rl/learning/manifest_data")' in setup_source
    assert 'glob("*.json")' in setup_source


def test_wheel_package_data_discovers_all_bundled_frozen_manifests() -> None:
    root = Path(__file__).resolve().parents[1]
    with mock.patch("setuptools.setup") as setup_call:
        runpy.run_path(str(root / "setup.py"), run_name="m16_setup_probe")
    keyword_arguments = setup_call.call_args.kwargs
    package_data = keyword_arguments["package_data"]["tienkung_rl"]
    assert {
        "learning/manifest_data/validation.json",
        "learning/manifest_data/final_standard_256.json",
        "learning/manifest_data/final_randomized_256.json",
    }.issubset(package_data)
