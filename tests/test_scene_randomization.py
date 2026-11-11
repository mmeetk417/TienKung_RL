"""M06 scene/randomization contract and real-MuJoCo integration tests."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import MujocoEnvironmentBackend, PhysicsState
from tienkung_rl.task.randomization import (
    CurriculumPlacement,
    CylinderRandomizer,
    EpisodeRandomizationParameters,
    EpisodeResetOrchestrator,
    EpisodeResetResult,
    ForcedCylinderParameters,
    ForcedParameterAuthority,
    RandomizationError,
    sample_closed_uniform,
)
from tienkung_rl.task.scene import (
    CONTACT_RULES_PATH,
    SITE_DECLARATION_PATH,
    SceneContractError,
    bind_scene,
    create_scene_description,
)


@pytest.fixture(scope="module")
def config() -> ResolvedRunConfig:
    return load_run_config()


@pytest.fixture
def backend(config: ResolvedRunConfig) -> Iterator[MujocoEnvironmentBackend]:
    instance = MujocoEnvironmentBackend(config)
    try:
        yield instance
    finally:
        instance.close()


class _EndpointSource:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


def test_scene_description_is_typed_complete_and_binds_real_runtime(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
) -> None:
    scene = create_scene_description(config)
    bound = bind_scene(scene, config, backend)

    assert scene.robot.keyframe_name == "robot_and_scene_home"
    assert scene.robot.fixed_base_body_name == "pelvis"
    assert scene.robot.environment_origin_m == (0.0, 0.0, 0.0)
    assert scene.table.center_m == config.task.scene.table_center_m
    assert scene.table.size_m == config.task.scene.table_size_m
    assert scene.table.leg_geom_names == tuple(f"table_leg_collision_{index}" for index in range(4))
    assert scene.cylinder.origin_xy_m == config.task.scene.cylinder_origin_xy_m
    assert {item.name for item in scene.task_frames} == {"right_tcp_frame", "target_tcp_frame"}
    assert bound.runtime_mjcf_path == backend.runtime_path.resolve()
    assert bound.bound_object_ids["keys"][scene.robot.keyframe_name] == backend.name_cache.home_key_id


def _mutated_runtime(config: ResolvedRunConfig, tmp_path: Path, mutation: str) -> Path:
    tree = ET.parse(config.asset.runtime_mjcf_path)
    root = tree.getroot()
    if mutation == "missing":
        geom = next(item for item in root.iter("geom") if item.get("name") == "table_top_collision")
        geom.set("name", "renamed_table_top")
    elif mutation == "duplicate":
        worldbody = root.find("worldbody")
        assert worldbody is not None
        ET.SubElement(worldbody, "geom", {"name": "table_top_collision", "type": "sphere", "size": "0.01"})
    elif mutation == "coordinate":
        body = next(item for item in root.iter("body") if item.get("name") == "cylinder")
        body.set("pos", "0 0 0")
    elif mutation == "frame":
        frame = next(item for item in root.iter("frame") if item.get("name") == "target_tcp_frame")
        frame.set("pos", "0 0 0.1")
    elif mutation == "physics":
        option = root.find("option")
        assert option is not None
        option.set("iterations", "9")
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    path = tmp_path / f"{mutation}.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "coordinate", "frame", "physics"])
def test_scene_binding_fails_closed_on_name_coordinate_or_physics_drift(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    tmp_path: Path,
    mutation: str,
) -> None:
    with pytest.raises(SceneContractError):
        bind_scene(
            create_scene_description(config),
            config,
            backend,
            runtime_mjcf_path=_mutated_runtime(config, tmp_path, mutation),
        )


def test_every_runtime_material_contact_and_solver_value_comes_from_physics_config(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
) -> None:
    scene = create_scene_description(config)
    bind_scene(scene, config, backend)

    changed_contact = replace(config.physics.contact, margin_m=config.physics.contact.margin_m + 0.001)
    changed_physics = replace(config.physics, contact=changed_contact)
    changed_config = replace(config, physics=changed_physics)
    changed_config.validate()
    with pytest.raises(SceneContractError, match="margin differs"):
        bind_scene(scene, changed_config, backend)

    changed_solver = replace(
        config.physics.solver,
        position_iterations=config.physics.solver.position_iterations + 1,
    )
    changed_config = replace(config, physics=replace(config.physics, solver=changed_solver))
    changed_config.validate()
    with pytest.raises(SceneContractError, match="iterations differs"):
        bind_scene(scene, changed_config, backend)

    changed_solver = replace(
        config.physics.solver,
        no_slip_iterations=config.physics.solver.no_slip_iterations + 1,
    )
    changed_config = replace(config, physics=replace(config.physics, solver=changed_solver))
    changed_config.validate()
    with pytest.raises(SceneContractError, match="noslip_iterations differs"):
        bind_scene(scene, changed_config, backend)


def test_versioned_reach_and_script_contact_rules_resolve_all_names(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    tmp_path: Path,
) -> None:
    scene = create_scene_description(config)
    assert scene.contact_rules.version == "tienkung2_pro_contact_groups_v1"
    assert scene.contact_rules.reach.allowed_group_pairs == ()
    assert scene.contact_rules.script.allowed_group_pairs == (
        ("thumb_side", "cylinder"),
        ("four_finger_opposing", "cylinder"),
    )
    bind_scene(scene, config, backend)

    raw = yaml.safe_load(CONTACT_RULES_PATH.read_text(encoding="utf-8"))
    raw["phase_rules"]["script"]["allowed_pairs"][0][0] = "unknown_group"
    invalid = tmp_path / "contact_groups.yaml"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(SceneContractError, match="unknown group"):
        create_scene_description(config, contact_rules_path=invalid, site_declaration_path=SITE_DECLARATION_PATH)


def test_closed_uniform_endpoint_fixture_covers_both_bounds(config: ResolvedRunConfig) -> None:
    assert sample_closed_uniform(_EndpointSource(0.0), config.randomization.radius) == 0.010
    assert sample_closed_uniform(_EndpointSource(1.0), config.randomization.radius) == 0.015
    with pytest.raises(RandomizationError, match=r"outside \[0, 1\]"):
        sample_closed_uniform(_EndpointSource(math.nextafter(1.0, math.inf)), config.randomization.radius)


def test_sample_contains_five_values_fixed_friction_inertia_version_and_position(
    config: ResolvedRunConfig,
) -> None:
    randomizer = CylinderRandomizer(config.randomization, seed=17)
    first = randomizer.sample()
    second = randomizer.sample()
    assert first.rng_position == 0
    assert second.rng_position == 1
    assert first.distribution_version == config.randomization.version
    assert first.torsional_friction == config.randomization.fixed_torsional_friction
    assert first.rolling_friction == config.randomization.fixed_rolling_friction
    assert first.to_cylinder_parameters().solid_inertia_kg_m2 == pytest.approx(first.solid_inertia_kg_m2)

    heavier = replace(first, mass_kg=first.mass_kg * 1.1)
    assert heavier.solid_inertia_kg_m2 != first.solid_inertia_kg_m2
    assert np.asarray(heavier.solid_inertia_kg_m2) == pytest.approx(np.asarray(first.solid_inertia_kg_m2) * 1.1)


def test_fixed_frictions_are_not_sampled_but_change_randomization_fingerprint(
    config: ResolvedRunConfig,
) -> None:
    samples = [CylinderRandomizer(config.randomization, seed=index).sample() for index in range(20)]
    assert {item.torsional_friction for item in samples} == {config.randomization.fixed_torsional_friction}
    assert {item.rolling_friction for item in samples} == {config.randomization.fixed_rolling_friction}


@pytest.mark.parametrize("authority", list(ForcedParameterAuthority))
def test_forced_parameters_require_explicit_authorized_workflow_and_remain_validated(
    config: ResolvedRunConfig,
    authority: ForcedParameterAuthority,
) -> None:
    forced = ForcedCylinderParameters(
        radius_m=config.randomization.radius.minimum,
        height_m=config.randomization.height.maximum,
        mass_kg=config.randomization.mass.minimum,
        hand_sliding_friction=config.randomization.hand_sliding_friction.maximum,
        table_sliding_friction=config.randomization.table_sliding_friction.minimum,
    )
    sample = CylinderRandomizer(config.randomization, seed=3).sample(forced=forced, authority=authority)
    assert sample.radius_m == forced.radius_m
    assert sample.height_m == forced.height_m
    assert sample.mass_kg == forced.mass_kg

    with pytest.raises(RandomizationError, match="require viewer"):
        CylinderRandomizer(config.randomization, seed=3).sample(forced=forced)
    with pytest.raises(RandomizationError, match="outside"):
        invalid_randomizer = CylinderRandomizer(config.randomization, seed=3)
        before = invalid_randomizer.state()
        invalid_randomizer.sample(
            forced=ForcedCylinderParameters(radius_m=config.randomization.radius.minimum - 0.001),
            authority=authority,
        )
    assert invalid_randomizer.state() == before
    with pytest.raises(RandomizationError, match="outside"):
        CylinderRandomizer(config.randomization, seed=3).sample(
            forced=ForcedCylinderParameters(mass_kg=math.nan),
            authority=authority,
        )


def test_rng_state_restore_reproduces_all_subsequent_samples(config: ResolvedRunConfig) -> None:
    first = CylinderRandomizer(config.randomization, seed=90210)
    for _ in range(7):
        first.sample()
    checkpoint = first.state()
    expected = tuple(first.sample() for _ in range(10))

    restored = CylinderRandomizer(config.randomization, seed=0)
    restored.restore(checkpoint)
    assert restored.state() == checkpoint
    assert tuple(restored.sample() for _ in range(10)) == expected

    wrong_version = replace(checkpoint, distribution_version="wrong")
    with pytest.raises(RandomizationError, match="version differs"):
        restored.restore(wrong_version)


def _forced_extreme(config: ResolvedRunConfig, *, maximum: bool) -> ForcedCylinderParameters:
    select = (lambda distribution: distribution.maximum) if maximum else (lambda distribution: distribution.minimum)
    return ForcedCylinderParameters(
        radius_m=select(config.randomization.radius),
        height_m=select(config.randomization.height),
        mass_kg=select(config.randomization.mass),
        hand_sliding_friction=select(config.randomization.hand_sliding_friction),
        table_sliding_friction=select(config.randomization.table_sliding_friction),
    )


def _reset_with_extreme(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    *,
    maximum: bool,
) -> EpisodeResetResult[tuple[float, ...]]:
    orchestrator = EpisodeResetOrchestrator(
        config,
        backend,
        CylinderRandomizer(config.randomization, seed=11),
    )
    result = orchestrator.reset(
        lambda: CurriculumPlacement("S0", config.task.scene.cylinder_origin_xy_m, "test"),
        lambda: None,
        lambda state, parameters: tuple(state.cylinder_position_m),
        forced=_forced_extreme(config, maximum=maximum),
        authority=ForcedParameterAuthority.SCRIPT_DIAGNOSTIC,
    )
    return result


def test_sample_is_applied_to_real_independent_models_with_physical_inertia_and_friction(
    config: ResolvedRunConfig,
) -> None:
    first = MujocoEnvironmentBackend(config)
    second = MujocoEnvironmentBackend(config)
    try:
        second_before = second.read_state().cylinder_parameters
        result = _reset_with_extreme(config, first, maximum=True)
        sample = result.parameters.cylinder
        assert second.read_state().cylinder_parameters == second_before
        assert first.read_state().cylinder_parameters != second.read_state().cylinder_parameters

        model = first._model
        assert model is not None
        geom_id = first.name_cache.cylinder_geom_id
        body_id = first.name_cache.cylinder_body_id
        half_side = sample.radius_m / np.sqrt(2.0)
        assert tuple(model.geom_size[geom_id, :3]) == pytest.approx((half_side, half_side, 0.5 * sample.height_m))
        assert float(model.body_mass[body_id]) == pytest.approx(sample.mass_kg)
        assert tuple(model.body_inertia[body_id]) == pytest.approx(sample.solid_inertia_kg_m2)
        assert tuple(model.geom_friction[geom_id]) == pytest.approx(
            (
                min(sample.hand_sliding_friction, sample.table_sliding_friction),
                sample.torsional_friction,
                sample.rolling_friction,
            )
        )
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("maximum", [False, True])
def test_target_height_uses_actual_randomized_geometric_center_and_reset_order(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
    maximum: bool,
) -> None:
    events: list[str] = []
    randomizer = CylinderRandomizer(config.randomization, seed=22)
    orchestrator = EpisodeResetOrchestrator(config, backend, randomizer)

    def curriculum() -> CurriculumPlacement:
        events.append("curriculum")
        return CurriculumPlacement("S0", config.task.scene.cylinder_origin_xy_m, "bounds")

    def clear() -> None:
        events.append("clear")
        assert backend.read_state().cylinder_parameters.height_m == (
            config.randomization.height.maximum if maximum else config.randomization.height.minimum
        )

    def observe(state: PhysicsState, parameters: EpisodeRandomizationParameters) -> np.ndarray:
        events.append("observation")
        assert events == ["curriculum", "clear", "observation"]
        return np.array(state.cylinder_position_m, copy=True)

    result = orchestrator.reset(
        curriculum,
        clear,
        observe,
        forced=_forced_extreme(config, maximum=maximum),
        authority=ForcedParameterAuthority.VIEWER,
    )
    sample = result.parameters.cylinder
    expected_z = (
        config.task.scene.table_center_m[2]
        + 0.5 * config.task.scene.table_size_m[2]
        + config.task.scene.cylinder_table_clearance_m
        + 0.5 * sample.height_m
    )
    assert result.parameters.cylinder_position_m[2] == pytest.approx(expected_z)
    assert result.parameters.grasp_target_position_m == result.parameters.cylinder_position_m
    assert tuple(result.observation) == pytest.approx(result.parameters.grasp_target_position_m)
    assert result.parameters.reset_sequence == (
        "curriculum_xy",
        "five_parameter_sample",
        "inertia_and_friction_application",
        "physics_reset",
        "transient_clear",
        "first_observation",
    )
    assert result.state.episode_physics_steps == 0
    assert backend.read_contacts().contacts == ()


def test_statistical_ranges_uniformity_and_independence(config: ResolvedRunConfig) -> None:
    """Use N=30k; 10-bin deviations <=8%, mean <=1%, and |correlation| <=0.03.

    These tolerances are deliberately several standard errors wide: each bin
    expects 3000 values (binomial sigma about 52), and independent correlations
    have standard error about 0.0058.  The test detects biased/shared streams
    without becoming flaky for the fixed seed.
    """

    count = 30_000
    randomizer = CylinderRandomizer(config.randomization, seed=8675309)
    matrix = np.asarray(
        [
            (
                sample.radius_m,
                sample.height_m,
                sample.mass_kg,
                sample.hand_sliding_friction,
                sample.table_sliding_friction,
            )
            for sample in (randomizer.sample() for _ in range(count))
        ],
        dtype=np.float64,
    )
    distributions = (
        config.randomization.radius,
        config.randomization.height,
        config.randomization.mass,
        config.randomization.hand_sliding_friction,
        config.randomization.table_sliding_friction,
    )
    for column, distribution in enumerate(distributions):
        values = matrix[:, column]
        assert float(values.min()) >= distribution.minimum
        assert float(values.max()) <= distribution.maximum
        midpoint = 0.5 * (distribution.minimum + distribution.maximum)
        width = distribution.maximum - distribution.minimum
        assert abs(float(values.mean()) - midpoint) <= 0.01 * width
        counts, _ = np.histogram(values, bins=10, range=(distribution.minimum, distribution.maximum))
        assert float(np.max(np.abs(counts - count / 10))) <= 0.08 * (count / 10)
    correlations = np.corrcoef(matrix, rowvar=False)
    assert isinstance(correlations, np.ndarray)
    off_diagonal = correlations[np.triu_indices(5, k=1)]
    assert float(np.max(np.abs(off_diagonal))) <= 0.03


def test_episode_parameter_structure_is_json_reusable_without_reinference(
    config: ResolvedRunConfig,
    backend: MujocoEnvironmentBackend,
) -> None:
    orchestrator = EpisodeResetOrchestrator(
        config,
        backend,
        CylinderRandomizer(config.randomization, seed=123),
    )
    result = orchestrator.reset(
        lambda: CurriculumPlacement("S0", config.task.scene.cylinder_origin_xy_m, "current"),
        lambda: None,
        lambda _state, parameters: parameters.to_dict(),
    )
    payload = result.parameters.to_dict()
    assert result.observation == payload
    cylinder_payload = cast(dict[str, object], payload["cylinder"])
    rng_payload = cast(dict[str, object], payload["rng_state_after"])
    assert cylinder_payload["solid_inertia_kg_m2"] == list(result.parameters.cylinder.solid_inertia_kg_m2)
    assert rng_payload["samples_drawn"] == 1
    assert json.loads(json.dumps(payload, allow_nan=False))["cylinder"]["distribution_version"] == (
        config.randomization.version
    )
