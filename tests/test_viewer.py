"""M17 native-viewer orchestration, diagnostics, and headless parity tests."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch

from tienkung_rl.config.load import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.learning.checkpoint import CheckpointError
from tienkung_rl.sim.mujoco_backend import MujocoEnvironmentBackend
from tienkung_rl.task.curriculum import ForcedTargetAuthority
from tienkung_rl.task.environment import ReachThenCloseTask, TaskResetOverrides
from tienkung_rl.task.randomization import ForcedCylinderParameters, ForcedParameterAuthority
from tienkung_rl.visualization.debug_geometry import (
    apply_native_debug_geometry,
    build_debug_geometry,
    separate_robot_collision_render_group,
)
from tienkung_rl.visualization.overlays import (
    OverlaySection,
    apply_native_overlays,
    build_overlay_sections,
    clear_native_overlays,
)
from tienkung_rl.visualization.viewer import (
    RealtimeClock,
    ViewerController,
    ViewerError,
    apply_viewer_brightness,
    fixed_scripted_reach_action,
    preflight_viewer_assets,
    run_case_trace,
)


@pytest.fixture
def config() -> ResolvedRunConfig:
    return load_run_config(overrides={"platform.num_envs": 1, "platform.worker_threads": 1})


def _forced(config: ResolvedRunConfig) -> TaskResetOverrides:
    origin = config.curriculum.cylinder_origin_xy_m
    return TaskResetOverrides(
        target_xy_m=(origin[0] + 0.006, origin[1] + 0.006),
        cylinder=ForcedCylinderParameters(
            radius_m=0.015,
            height_m=0.100,
            mass_kg=0.115,
            hand_sliding_friction=0.9,
            table_sliding_friction=0.9,
        ),
        target_authority=ForcedTargetAuthority.VIEWER,
        cylinder_authority=ForcedParameterAuthority.VIEWER,
    )


def test_forced_viewer_case_is_persistent_validated_and_reseedable(config: ResolvedRunConfig) -> None:
    overrides = _forced(config)
    with ReachThenCloseTask(config, seed=12, reset_overrides=overrides, curriculum_updates_enabled=False) as task:
        task.reset()
        first = task.snapshot()
        for _ in range(config.task.scene.maximum_episode_steps):
            result = task.step(torch.zeros(7, dtype=torch.float32), auto_reset=False)
            if result.transition.done:
                break
        assert result.transition.done
        task.reseed_reset_streams(99)
        task.reset()
        second = task.snapshot()
    assert first.cylinder_reset_position_m == second.cylinder_reset_position_m
    assert first.randomization["forced_authority"] == "viewer"
    cylinder = second.randomization["cylinder"]
    assert isinstance(cylinder, Mapping)
    assert cylinder["mass_kg"] == pytest.approx(0.115)


def test_viewer_asset_and_headless_runtime_make_identical_task_decisions(config: ResolvedRunConfig) -> None:
    actions = [torch.zeros(7, dtype=torch.float32) for _ in range(4)]
    expected = run_case_trace(config, actions, seed=21, reset_overrides=_forced(config), viewer_asset=False)
    actual = run_case_trace(config, actions, seed=21, reset_overrides=_forced(config), viewer_asset=True)
    assert actual == expected


def test_overlay_and_debug_plans_are_read_only_and_render_to_user_scene(config: ResolvedRunConfig) -> None:
    backend = MujocoEnvironmentBackend(config, mjcf_path=config.asset.viewer_mjcf_path)
    task = ReachThenCloseTask(config, backend=backend, seed=3, reset_overrides=_forced(config))
    try:
        task.reset()
        for _ in range(2):
            task.step(torch.zeros(7, dtype=torch.float32), auto_reset=False)
        before_curriculum = task.curriculum.state().to_dict()
        before_randomization = task.environment.randomizer.state().to_dict()
        snapshot = task.snapshot()
        sections = build_overlay_sections(snapshot, mode="zero", realtime_factor=0.42)
        plan = build_debug_geometry(
            snapshot,
            backend.read_state(),
            backend.read_contacts(),
            admission_position_error_m=config.task.admission.position_error_m,
        )
        model, _ = backend.native_viewer_handles()
        separated = separate_robot_collision_render_group(model)
        assert separated > 0
        robot_collision_mask = np.asarray(model.geom_contype) == 1
        task_collision_mask = np.isin(np.asarray(model.geom_contype), (2, 4))
        assert np.all(np.asarray(model.geom_group)[robot_collision_mask] == 5)
        assert np.all(np.asarray(model.geom_group)[task_collision_mask] == 3)
        native = SimpleNamespace(user_scn=mujoco.MjvScene(model, maxgeom=128), opt=mujoco.MjvOption())
        apply_native_debug_geometry(native, plan)
        assert native.user_scn.ngeom == len(plan.primitives)
        assert any(item.label.startswith("contact_normal_") for item in plan.primitives)
        assert any(item.label.startswith("contact_force_") for item in plan.primitives)
        assert int(native.opt.geomgroup[3]) == 1
        assert int(native.opt.geomgroup[5]) == 0
        assert "realtime=0.420x" in sections[0].content
        assert "admission_failures=" in sections[1].content
        assert "primary=" in sections[2].content
        assert "nonfinite=" in sections[3].content
        assert task.curriculum.state().to_dict() == before_curriculum
        assert task.environment.randomizer.state().to_dict() == before_randomization
    finally:
        backend.close()


def test_native_overlay_uses_mujoco_311_passive_handle_text_api() -> None:
    class TextTarget:
        def __init__(self) -> None:
            self.texts: object | None = None

        def set_texts(self, texts: object) -> None:
            self.texts = texts

    target = TextTarget()
    sections = (OverlaySection("status", "running"), OverlaySection("task", "admissible=no"))

    apply_native_overlays(target, sections)

    assert target.texts == (
        mujoco.mjtFontScale.mjFONTSCALE_100,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "status\n\ntask",
        "running\n\nadmissible=no",
    )

    clear_native_overlays(target)
    assert target.texts == (
        mujoco.mjtFontScale.mjFONTSCALE_100,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "",
        "",
    )


def test_viewer_clean_presentation_is_default_and_collision_diagnostics_are_opt_in() -> None:
    from tienkung_rl import viewer

    defaults = viewer.build_parser().parse_args(["--mode", "zero"])
    assert not defaults.show_collisions
    assert defaults.brightness == 1.5

    args = viewer.build_parser().parse_args(
        [
            "--mode",
            "zero",
            "--show-collisions",
            "--hide-debug-geometry",
            "--hide-overlays",
            "--pause-on-terminal",
        ]
    )

    assert args.show_collisions
    assert args.hide_debug_geometry
    assert args.hide_overlays
    assert args.pause_on_terminal


def test_viewer_brightness_scales_only_headlight_channels() -> None:
    headlight = SimpleNamespace(
        ambient=np.array([0.1, 0.2, 0.3]),
        diffuse=np.array([0.4, 0.5, 0.6]),
        specular=np.array([0.7, 0.8, 0.9]),
    )
    model = SimpleNamespace(vis=SimpleNamespace(headlight=headlight))
    apply_viewer_brightness(model, 1.5)
    np.testing.assert_allclose(headlight.ambient, [0.15, 0.30, 0.45])
    np.testing.assert_allclose(headlight.diffuse, [0.60, 0.75, 0.90])
    np.testing.assert_allclose(headlight.specular, [1.05, 1.20, 1.35])
    with pytest.raises(ViewerError, match="between 0 and 10"):
        apply_viewer_brightness(model, 11.0)


def test_scripted_controller_emits_only_the_real_reach_action(config: ResolvedRunConfig) -> None:
    backend = MujocoEnvironmentBackend(config, mjcf_path=config.asset.viewer_mjcf_path)
    task = ReachThenCloseTask(config, backend=backend, seed=5, reset_overrides=_forced(config))
    try:
        observation = task.reset().values

        def source(snapshot):  # type: ignore[no-untyped-def]
            return fixed_scripted_reach_action(config, backend, snapshot)

        controller = ViewerController(
            "scripted",
            seed=5,
            scripted_arm_source=source,
        )
        baseline = task.snapshot()
        before_gate = controller.action(observation, replace(baseline, episode_step=24, option_admissible=True))
        rejected_gate = controller.action(observation, replace(baseline, episode_step=25, option_admissible=False))
        accepted_gate = controller.action(observation, replace(baseline, episode_step=25, option_admissible=True))
        torch.testing.assert_close(before_gate, rejected_gate)
        torch.testing.assert_close(rejected_gate, accepted_gate)
        assert accepted_gate.shape == (7,)
        assert torch.isfinite(accepted_gate).all()
    finally:
        backend.close()


def test_realtime_clock_never_drops_steps_and_restarts_per_episode() -> None:
    wall = [10.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return wall[0]

    def sleep(duration: float) -> None:
        sleeps.append(duration)
        wall[0] += duration

    clock = RealtimeClock(2.0, wall[0], monotonic, sleep)
    clock.pace(2.02)
    assert sleeps == pytest.approx([0.02])
    wall[0] += 0.04
    clock.pace(2.04)
    assert len(sleeps) == 1  # late execution does not skip or fake a physics step
    restarted = RealtimeClock(0.0, wall[0], monotonic, sleep)
    wall[0] += 0.01
    assert restarted.factor(0.005) == pytest.approx(0.5)


def test_no_display_fails_before_native_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    from tienkung_rl.visualization import viewer

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(ViewerError, match="--headless-steps"):
        viewer._require_display()


def test_checkpoint_preflight_fails_before_environment_construction(
    config: ResolvedRunConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tienkung_rl.visualization import viewer

    constructed = False

    def reject(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise CheckpointError("incompatible checkpoint")

    def construct(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal constructed
        constructed = True
        raise AssertionError("environment construction must not be reached")

    monkeypatch.setattr(viewer, "preflight_viewer_assets", lambda _: None)
    monkeypatch.setattr(viewer, "_load_policy", reject)
    monkeypatch.setattr(viewer, "MujocoEnvironmentBackend", construct)
    args = argparse.Namespace(mode="policy", checkpoint=Path("bad.pt"))
    with pytest.raises(CheckpointError, match="incompatible"):
        viewer.run(config, args)
    assert not constructed


def test_asset_preflight_accepts_edits_without_hash_gate(config: ResolvedRunConfig) -> None:
    changed = replace(config, asset=replace(config.asset, viewer_mjcf_sha256="0" * 64))
    preflight_viewer_assets(changed)
