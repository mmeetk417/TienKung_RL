"""M11 S0--S3 sampling, complete-window promotion, and recovery tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tienkung_rl.config.load import load_run_config
from tienkung_rl.config.models import CurriculumConfig, ResolvedRunConfig
from tienkung_rl.task.curriculum import (
    CurriculumCheckpointState,
    CurriculumError,
    CurriculumManager,
    CurriculumStage,
    EpisodeCurriculumResult,
    ForcedTargetAuthority,
    WindowCounts,
    sample_curriculum_target,
)


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config()


class SequenceSource:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = iter(values)

    def random(self) -> float:
        return next(self.values)


def _offset(placement_xy: tuple[float, float], curriculum: CurriculumConfig) -> tuple[float, float]:
    return (
        placement_xy[0] - curriculum.cylinder_origin_xy_m[0],
        placement_xy[1] - curriculum.cylinder_origin_xy_m[1],
    )


@pytest.mark.parametrize(
    ("x_unit", "y_unit", "expected"),
    (
        (0.0, 0.0, (0.005, 0.005)),
        (0.0, 1.0, (0.005, 0.008)),
        (1.0, 0.0, (0.008, 0.005)),
        (1.0, 1.0, (0.008, 0.008)),
    ),
)
def test_s0_four_closed_boundaries_are_deterministic(
    config: ResolvedRunConfig,
    x_unit: float,
    y_unit: float,
    expected: tuple[float, float],
) -> None:
    placement = sample_curriculum_target(
        config.curriculum,
        CurriculumStage.S0,
        SequenceSource((0.5, x_unit, y_unit)),
    )
    assert placement.stage_at_reset == "S0"
    assert placement.component_label == "current_range"
    assert _offset(placement.target_xy_m, config.curriculum) == pytest.approx(expected, abs=1e-15)


def test_closed_unit_component_endpoint_selects_last_configured_component(config: ResolvedRunConfig) -> None:
    s0 = sample_curriculum_target(config.curriculum, "S0", SequenceSource((1.0, 0.0, 0.0)))
    s1 = sample_curriculum_target(config.curriculum, "S1", SequenceSource((1.0, 0.0, 0.0)))
    assert s0.component_label == "current_range"
    assert s1.component_label == "replay_s0_range"


@pytest.mark.parametrize(
    ("stage", "expected_current", "expected_replay"),
    (
        (CurriculumStage.S1, ((0.0, 0.035), (0.0, 0.050)), ((0.005, 0.008), (0.005, 0.008))),
        (CurriculumStage.S2, ((0.0, 0.045), (0.0, 0.070)), ((0.0, 0.035), (0.0, 0.050))),
    ),
)
def test_s1_s2_component_ratio_and_component_bounds(
    config: ResolvedRunConfig,
    stage: CurriculumStage,
    expected_current: tuple[tuple[float, float], tuple[float, float]],
    expected_replay: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    source = np.random.Generator(np.random.PCG64(100 + list(CurriculumStage).index(stage)))
    counts = {"current": 0, "replay": 0}
    for _ in range(20_000):
        placement = sample_curriculum_target(config.curriculum, stage, source)
        x, y = _offset(placement.target_xy_m, config.curriculum)
        if placement.component_label == "current_range":
            counts["current"] += 1
            bounds = expected_current
        else:
            replay_stage = CurriculumStage.S0 if stage == CurriculumStage.S1 else CurriculumStage.S1
            assert placement.component_label == f"replay_{replay_stage.value.lower()}_range"
            counts["replay"] += 1
            bounds = expected_replay
        assert bounds[0][0] <= x <= bounds[0][1]
        assert bounds[1][0] <= y <= bounds[1][1]
    assert counts["current"] / 20_000 == pytest.approx(0.8, abs=0.015)
    assert counts["replay"] / 20_000 == pytest.approx(0.2, abs=0.015)


def test_s3_ratio_boundary_labels_jitter_and_global_bounds(config: ResolvedRunConfig) -> None:
    source = np.random.Generator(np.random.PCG64(203))
    current = 0
    boundary = 0
    for _ in range(30_000):
        placement = sample_curriculum_target(config.curriculum, CurriculumStage.S3, source)
        x, y = _offset(placement.target_xy_m, config.curriculum)
        assert -1e-15 <= x <= 0.045 + 1e-15
        assert -1e-15 <= y <= 0.070 + 1e-15
        if placement.component_label == "current_range":
            current += 1
        else:
            boundary += 1
            index = int(placement.component_label.removeprefix("boundary_corner_"))
            corner = config.curriculum.boundary_corners_xy_offset_m[index]
            assert abs(x - corner[0]) <= 0.004 + 1e-15
            assert abs(y - corner[1]) <= 0.006 + 1e-15
    assert current / 30_000 == pytest.approx(0.7, abs=0.012)
    assert boundary / 30_000 == pytest.approx(0.3, abs=0.012)


def test_forced_xy_is_authorized_finite_in_range_and_applied_after_sampling(config: ResolvedRunConfig) -> None:
    origin = config.curriculum.cylinder_origin_xy_m
    forced = (origin[0] + 0.020, origin[1] + 0.030)
    baseline_source = np.random.Generator(np.random.PCG64(77))
    forced_source = np.random.Generator(np.random.PCG64(77))
    baseline = sample_curriculum_target(config.curriculum, "S3", baseline_source)
    placement = sample_curriculum_target(
        config.curriculum,
        "S3",
        forced_source,
        forced_xy_m=forced,
        authority=ForcedTargetAuthority.FROZEN_MANIFEST,
    )
    assert placement.target_xy_m == forced
    assert placement.component_label == f"forced_frozen_manifest_after_{baseline.component_label}"
    # Matching next samples prove the ordinary draw was consumed before override.
    assert sample_curriculum_target(config.curriculum, "S3", baseline_source) == sample_curriculum_target(
        config.curriculum, "S3", forced_source
    )
    with pytest.raises(CurriculumError, match="authority"):
        sample_curriculum_target(config.curriculum, "S3", np.random.default_rng(1), forced_xy_m=forced)
    with pytest.raises(CurriculumError, match="finite"):
        sample_curriculum_target(
            config.curriculum,
            "S3",
            np.random.default_rng(1),
            forced_xy_m=(math.nan, forced[1]),
            authority=ForcedTargetAuthority.VIEWER,
        )
    with pytest.raises(CurriculumError, match="outside"):
        sample_curriculum_target(
            config.curriculum,
            "S3",
            np.random.default_rng(1),
            forced_xy_m=(origin[0] + 0.046, origin[1]),
            authority=ForcedTargetAuthority.ASSET_DIAGNOSTIC,
        )


def test_forced_xy_accepts_world_space_closed_endpoint_roundoff(config: ResolvedRunConfig) -> None:
    origin = config.curriculum.cylinder_origin_xy_m
    s3 = config.curriculum.stages[3]
    forced = (origin[0] + s3.x_offset_range_m[1], origin[1] + s3.y_offset_range_m[1])
    assert forced[0] - origin[0] > s3.x_offset_range_m[1]

    placement = sample_curriculum_target(
        config.curriculum,
        "S3",
        np.random.default_rng(1),
        forced_xy_m=forced,
        authority=ForcedTargetAuthority.FROZEN_MANIFEST,
    )

    assert placement.target_xy_m == forced
    with pytest.raises(CurriculumError, match="outside"):
        sample_curriculum_target(
            config.curriculum,
            "S3",
            np.random.default_rng(1),
            forced_xy_m=(forced[0] + 1.0e-9, forced[1]),
            authority=ForcedTargetAuthority.FROZEN_MANIFEST,
        )


def _result(
    stage: CurriculumStage,
    index: int,
    *,
    total: int,
    accepted: float,
    lifted: float,
    success: float,
    dangerous: float,
) -> EpisodeCurriculumResult:
    return EpisodeCurriculumResult(
        stage_at_reset=stage,
        option_accepted=index < math.ceil(total * accepted),
        lifted_5cm=index < math.ceil(total * lifted),
        complete_success=index < math.ceil(total * success),
        dangerous=index < math.floor(total * dangerous),
    )


def _fill_window(
    manager: CurriculumManager,
    *,
    accepted: float,
    lifted: float,
    success: float,
    dangerous: float,
) -> None:
    gate_size = manager.diagnostics().target_episodes
    stage = manager.stage
    for index in range(gate_size):
        diagnostic = manager.record_completed_episode(
            _result(
                stage,
                index,
                total=gate_size,
                accepted=accepted,
                lifted=lifted,
                success=success,
                dangerous=dangerous,
            )
        )
        assert diagnostic.complete_window is (index == gate_size - 1)


def test_window_is_evaluated_only_when_exactly_full_and_ready_is_excluded(config: ResolvedRunConfig) -> None:
    manager = CurriculumManager(config, seed=1)
    for _ in range(5000):
        manager.record_strict_ready()
    assert manager.window == WindowCounts()
    assert manager.strict_ready_events == 5000
    for index in range(2047):
        diagnostic = manager.record_completed_episode(EpisodeCurriculumResult("S0", True, True, True, False))
        assert not diagnostic.complete_window
        assert manager.consecutive_passing_windows == 0
        assert diagnostic.progress == pytest.approx((index + 1) / 2048)
    final = manager.record_completed_episode(EpisodeCurriculumResult("S0", True, True, True, False))
    assert final.complete_window and final.passed and not final.stalled
    assert final.rates.option_accepted == 1.0
    assert manager.window == WindowCounts()
    assert manager.consecutive_passing_windows == 1


@pytest.mark.parametrize(
    ("stage", "window", "accepted", "lifted", "success"),
    (
        (CurriculumStage.S0, 2048, 0.70, 0.60, 0.50),
        (CurriculumStage.S1, 4096, 0.80, 0.70, 0.60),
        (CurriculumStage.S2, 8192, 0.85, 0.80, 0.70),
    ),
)
def test_all_threshold_boundaries_pass_and_two_windows_promote(
    config: ResolvedRunConfig,
    stage: CurriculumStage,
    window: int,
    accepted: float,
    lifted: float,
    success: float,
) -> None:
    manager = CurriculumManager(config, seed=2)
    if stage != CurriculumStage.S0:
        state = manager.state()
        manager.restore(replace(state, stage=stage))
    assert manager.diagnostics().target_episodes == window
    _fill_window(manager, accepted=accepted, lifted=lifted, success=success, dangerous=0.10)
    assert manager.stage == stage
    assert manager.consecutive_passing_windows == 1
    _fill_window(manager, accepted=accepted, lifted=lifted, success=success, dangerous=0.10)
    next_stage = {
        CurriculumStage.S0: CurriculumStage.S1,
        CurriculumStage.S1: CurriculumStage.S2,
        CurriculumStage.S2: CurriculumStage.S3,
    }[stage]
    assert manager.stage == next_stage
    assert manager.window == WindowCounts()
    assert manager.consecutive_passing_windows == 0


def test_failed_window_resets_streak_reports_gap_and_stall(config: ResolvedRunConfig) -> None:
    manager = CurriculumManager(config, seed=3)
    _fill_window(manager, accepted=1.0, lifted=1.0, success=1.0, dangerous=0.0)
    assert manager.consecutive_passing_windows == 1
    diagnostic = None
    for index in range(2048):
        diagnostic = manager.record_completed_episode(
            _result(
                CurriculumStage.S0,
                index,
                total=2048,
                accepted=0.69,
                lifted=1.0,
                success=1.0,
                dangerous=0.11,
            )
        )
    assert diagnostic is not None
    assert diagnostic.complete_window and not diagnostic.passed and diagnostic.stalled
    assert diagnostic.threshold_gaps.option_accepted > 0.0
    assert diagnostic.threshold_gaps.dangerous > 0.0
    assert manager.consecutive_passing_windows == 0
    assert manager.stage == CurriculumStage.S0


def test_stage_at_reset_owns_episode_and_old_inflight_result_does_not_pollute_new_stage(
    config: ResolvedRunConfig,
) -> None:
    manager = CurriculumManager(config, seed=4)
    old_episode = manager.sample_target()
    _fill_window(manager, accepted=1.0, lifted=1.0, success=1.0, dangerous=0.0)
    _fill_window(manager, accepted=1.0, lifted=1.0, success=1.0, dangerous=0.0)
    assert manager.stage == CurriculumStage.S1
    ignored = manager.record_completed_episode(
        EpisodeCurriculumResult(old_episode.stage_at_reset, True, True, True, False)
    )
    assert ignored.ignored_old_stage and not ignored.counted
    assert manager.window == WindowCounts()
    with pytest.raises(CurriculumError, match="ahead"):
        manager.record_completed_episode(EpisodeCurriculumResult("S2", True, True, True, False))


def test_s3_is_terminal_and_never_automatically_downgrades(config: ResolvedRunConfig) -> None:
    manager = CurriculumManager(config, seed=5)
    manager.restore(replace(manager.state(), stage=CurriculumStage.S3))
    for _ in range(10_000):
        diagnostic = manager.record_completed_episode(EpisodeCurriculumResult("S3", False, False, False, True))
        assert not diagnostic.counted and not diagnostic.promoted
    assert manager.stage == CurriculumStage.S3
    lower = replace(manager.state(), stage=CurriculumStage.S2)
    with pytest.raises(CurriculumError, match="downgrade"):
        manager.restore(lower)


def test_partial_window_and_rng_round_trip_resume_exactly(config: ResolvedRunConfig) -> None:
    baseline = CurriculumManager(config, seed=1234)
    for _ in range(17):
        baseline.sample_target()
    for _ in range(23):
        baseline.record_completed_episode(EpisodeCurriculumResult("S0", True, False, False, False))
    baseline.record_strict_ready()
    state = CurriculumCheckpointState.from_dict(baseline.state().to_dict())
    assert state.window.completed_episodes == 23

    expected = tuple(baseline.sample_target() for _ in range(25))
    restored = CurriculumManager(config, seed=0)
    restored.restore(state.to_dict())
    assert tuple(restored.sample_target() for _ in range(25)) == expected
    assert restored.window == state.window
    assert restored.strict_ready_events == 1
    # Sampling/open episodes never enter the promotion window by themselves.
    assert restored.window.completed_episodes == 23


def test_restore_rejects_fingerprint_version_corruption_and_impossible_window(config: ResolvedRunConfig) -> None:
    manager = CurriculumManager(config, seed=6)
    state = manager.state()
    with pytest.raises(CurriculumError, match="sampler version mismatch"):
        manager.restore(replace(state, sampler_version="changed"))
    with pytest.raises(CurriculumError, match="must be incomplete"):
        manager.restore(replace(state, window=WindowCounts(completed_episodes=2048)))
    malformed = state.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(CurriculumError, match="incomplete or unknown"):
        manager.restore(malformed)
