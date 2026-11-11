"""Real-MuJoCo integration tests for the named single-environment boundary."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tienkung_rl.config import load_run_config
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.sim import (
    BackendClosedError,
    BackendContractError,
    CylinderParameters,
    EnvironmentBackend,
    MujocoEnvironmentBackend,
    NonFinitePhysicsError,
    PhysicsSubstepTrace,
    ResetSpec,
    is_illegal_contact,
    mujoco_quaternion_to_wxyz,
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


def _current_targets(backend: MujocoEnvironmentBackend) -> np.ndarray:
    return np.array(backend.read_state().actuator_targets_rad, dtype=np.float64, order="C")


def test_backend_satisfies_protocol_and_headless_lifecycle(
    config: ResolvedRunConfig,
) -> None:
    first = MujocoEnvironmentBackend(config)
    second = MujocoEnvironmentBackend(config)
    try:
        assert isinstance(first, EnvironmentBackend)
        assert first.runtime_path.name == "reach_then_close.xml"
        assert first.render_context_created is False
        assert first.is_closed is False

        # Each backend owns independent mutable state even though it loads the
        # same committed runtime asset.
        first.apply_targets(_current_targets(first))
        first.step_substeps(4)
        assert first.read_state().simulation_time_s == pytest.approx(0.020)
        assert second.read_state().simulation_time_s == 0.0
    finally:
        first.close()
        first.close()
        second.close()
    with pytest.raises(BackendClosedError, match="closed"):
        first.read_state()
    assert first.is_closed is True


def test_name_cache_strictly_matches_model_and_rejects_drift(
    config: ResolvedRunConfig,
    tmp_path: Path,
) -> None:
    with MujocoEnvironmentBackend(config) as instance:
        names = instance.name_cache
        assert len(names.object_ids["bodies"]) == 66
        assert len(names.object_ids["joints"]) == 55
        # The training runtime intentionally omits the 63 visual-only STL
        # geoms; both products retain the display-only checkerboard floor.
        assert len(names.object_ids["geoms"]) == 38
        assert "floor" in names.object_ids["geoms"]
        assert len(names.object_ids["sites"]) == 5
        assert len(names.object_ids["actuators"]) == 13
        assert names.actuator_names == tuple(
            f"{name}_position"
            for name in (
                *config.physics.actuator_calibration.arm.joint_names,
                *config.physics.actuator_calibration.hand.joint_names,
            )
        )

    source = Path(config.asset.runtime_mjcf_path).with_name("name_index.json")
    invalid = json.loads(source.read_text(encoding="utf-8"))
    invalid["indices"]["sites"].pop("grasp_center_site")
    invalid_path = tmp_path / "name_index.json"
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(BackendContractError, match="runtime sites do not match"):
        MujocoEnvironmentBackend(config, name_index_path=invalid_path)


def test_state_is_finite_named_float64_readonly_and_detached(
    backend: MujocoEnvironmentBackend,
) -> None:
    state = backend.reset(ResetSpec())
    expected_shapes = {
        "joint_positions_rad": (19,),
        "joint_velocities_rad_s": (19,),
        "body_positions_m": (66, 3),
        "body_quaternions_wxyz": (66, 4),
        "site_positions_m": (5, 3),
        "site_quaternions_wxyz": (5, 4),
        "site_linear_velocities_m_s": (5, 3),
        "site_angular_velocities_rad_s": (5, 3),
        "actuator_targets_rad": (13,),
        "actuator_forces_n_m": (13,),
    }
    for field_name, shape in expected_shapes.items():
        array = getattr(state, field_name)
        assert array.shape == shape
        assert array.dtype == np.float64
        assert np.isfinite(array).all()
        assert array.flags.c_contiguous
        assert array.flags.writeable is False
    config_joint_names = backend.name_cache.observed_joint_names
    assert state.joint_names == config_joint_names
    assert config_joint_names[:7] == (
        "shoulder_pitch_r_joint",
        "shoulder_roll_r_joint",
        "shoulder_yaw_r_joint",
        "elbow_pitch_r_joint",
        "elbow_yaw_r_joint",
        "wrist_pitch_r_joint",
        "wrist_roll_r_joint",
    )
    assert state.body_names[state.body_index("cylinder")] == "cylinder"
    assert state.site_names[state.site_index("right_tcp_site")] == "right_tcp_site"
    with pytest.raises(ValueError, match="read-only"):
        state.joint_positions_rad[0] = 1.0
    detached = state.joint_positions_rad.copy()
    backend.apply_targets(_current_targets(backend))
    backend.step_substeps(4)
    np.testing.assert_array_equal(state.joint_positions_rad, detached)


def test_reset_restores_state_clears_accumulators_and_keeps_lifetime_counter(
    backend: MujocoEnvironmentBackend,
) -> None:
    initial = backend.reset(ResetSpec())
    targets = _current_targets(backend)
    targets[0] += 0.1
    backend.apply_targets(targets)
    step = backend.step_substeps(4)
    assert step.contacts.substeps_collected == 4
    assert step.state.total_physics_steps >= 4

    reset_state = backend.reset(ResetSpec())
    assert reset_state.episode_physics_steps == 0
    assert reset_state.total_physics_steps == step.state.total_physics_steps
    assert reset_state.simulation_time_s == 0.0
    assert reset_state.total_simulation_time_s == pytest.approx(step.state.total_simulation_time_s)
    np.testing.assert_allclose(reset_state.joint_positions_rad, initial.joint_positions_rad, atol=1e-12)
    contacts = backend.read_contacts()
    assert contacts.contacts == ()
    assert contacts.substeps_collected == 0
    assert contacts.maximum_contacts_seen == 0
    assert backend.read_warnings().warnings == ()


def test_reset_named_joint_override_updates_matching_position_target(
    backend: MujocoEnvironmentBackend,
) -> None:
    shoulder_position = -0.25
    state = backend.reset(ResetSpec(robot_joint_positions_rad={"shoulder_pitch_r_joint": shoulder_position}))
    assert state.joint_positions_rad[state.joint_index("shoulder_pitch_r_joint")] == pytest.approx(shoulder_position)
    assert state.actuator_targets_rad[0] == pytest.approx(shoulder_position)


def test_invalid_reset_fails_before_mutating_cylinder(
    backend: MujocoEnvironmentBackend, config: ResolvedRunConfig
) -> None:
    original = backend.reset(ResetSpec()).cylinder_parameters
    changed = CylinderParameters(radius_m=0.018)
    with pytest.raises(BackendContractError, match="unknown robot joint"):
        backend.reset(
            ResetSpec(
                robot_joint_positions_rad={"not_a_joint": 0.0},
                cylinder=changed,
            )
        )
    assert backend.read_state().cylinder_parameters == original
    with pytest.raises(BackendContractError, match="outside configured"):
        backend.reset(ResetSpec(cylinder=CylinderParameters(radius_m=config.randomization.radius.minimum - 0.001)))
    assert backend.read_state().cylinder_parameters == original


@pytest.mark.parametrize(
    ("targets", "error", "message"),
    [
        (np.zeros(12, dtype=np.float64), ValueError, r"shape \(13,\)"),
        (np.zeros(13, dtype=np.float32), TypeError, "dtype float64"),
        (np.full(13, np.nan, dtype=np.float64), ValueError, "non-finite"),
        (np.zeros(26, dtype=np.float64)[::2], ValueError, "C-contiguous"),
        ([0.0] * 13, TypeError, "CPU numpy.ndarray"),
    ],
)
def test_targets_reject_wrong_shape_dtype_device_layout_and_nonfinite(
    backend: MujocoEnvironmentBackend,
    targets: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        backend.apply_targets(targets)  # type: ignore[arg-type]  # Deliberate invalid boundary input.


def test_targets_are_exactly_seven_arm_then_six_hand_and_range_checked(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    targets = _current_targets(backend)
    targets[:7] += np.asarray((0.01, -0.01, 0.01, 0.01, -0.01, 0.01, -0.01))
    targets[7:] += np.asarray((0.01, 0.01, 0.01, 0.01, 0.01, 0.01))
    backend.apply_targets(targets)
    np.testing.assert_array_equal(backend.read_state().actuator_targets_rad, targets)
    invalid = targets.copy()
    invalid[0] = 100.0
    with pytest.raises(ValueError, match="shoulder_pitch_r_joint_position"):
        backend.apply_targets(invalid)


def test_policy_step_is_exactly_four_substeps_at_five_milliseconds(
    backend: MujocoEnvironmentBackend,
) -> None:
    state = backend.reset(ResetSpec())
    backend.apply_targets(np.array(state.actuator_targets_rad, copy=True))
    with pytest.raises(ValueError, match="exactly 4"):
        backend.step_substeps(3)
    first = backend.step_substeps(4)
    second = backend.step_substeps(4)
    assert first.substeps_executed == second.substeps_executed == 4
    assert first.state.episode_physics_steps == 4
    assert second.state.episode_physics_steps == 8
    assert first.state.simulation_time_s == pytest.approx(0.020, abs=1e-12)
    assert second.state.simulation_time_s == pytest.approx(0.040, abs=1e-12)
    assert second.state.total_simulation_time_s == pytest.approx(
        second.state.total_physics_steps * 0.005,
        abs=1e-12,
    )


def test_zero_physical_targets_remain_finite_without_hidden_warning(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    backend.apply_targets(np.zeros(13, dtype=np.float64))
    for _ in range(25):
        result = backend.step_substeps(4)
    assert np.isfinite(result.state.joint_positions_rad).all()
    assert np.isfinite(result.state.joint_velocities_rad_s).all()
    assert result.state.simulation_time_s == pytest.approx(0.5, abs=1e-12)
    assert result.diagnostics.healthy


def test_single_joint_target_moves_towards_setpoint(backend: MujocoEnvironmentBackend) -> None:
    initial = backend.reset(ResetSpec())
    targets = np.array(initial.actuator_targets_rad, copy=True)
    initial_position = float(initial.joint_positions_rad[0])
    targets[0] = initial_position + 0.1
    initial_error = abs(initial_position - targets[0])
    backend.apply_targets(targets)
    for _ in range(25):
        result = backend.step_substeps(4)
    final_position = float(result.state.joint_positions_rad[0])
    assert final_position > initial_position
    assert abs(final_position - targets[0]) < initial_error * 0.15
    assert result.diagnostics.healthy


def test_quaternion_conversion_and_known_world_rotation(
    backend: MujocoEnvironmentBackend,
) -> None:
    half_sqrt = math.sqrt(0.5)
    expected = np.asarray((half_sqrt, 0.0, 0.0, half_sqrt), dtype=np.float64)
    np.testing.assert_allclose(mujoco_quaternion_to_wxyz(expected), expected, atol=1e-15)
    np.testing.assert_allclose(mujoco_quaternion_to_wxyz(-expected), expected, atol=1e-15)
    with pytest.raises(TypeError, match="float64"):
        mujoco_quaternion_to_wxyz(expected.astype(np.float32))

    state = backend.reset(ResetSpec(cylinder_quaternion_wxyz=tuple(expected)))
    np.testing.assert_allclose(state.cylinder_quaternion_wxyz, expected, atol=1e-12)
    assert state.cylinder_quaternion_wxyz[0] >= 0.0


def test_contacts_are_collected_for_each_substep_with_force_impulse_and_groups(
    backend: MujocoEnvironmentBackend,
) -> None:
    initial = backend.reset(ResetSpec())
    backend.apply_targets(np.array(initial.actuator_targets_rad, copy=True))
    frame = backend.step_substeps(4).contacts
    assert frame.substeps_collected == 4
    assert frame.maximum_contacts_seen > 0
    assert len(frame.contacts) >= frame.maximum_contacts_seen
    # The reset clearance can leave the first substep without contact.
    assert {sample.substep_index for sample in frame.contacts} <= {0, 1, 2, 3}
    assert frame.contacts
    assert any("cylinder_collision" in (sample.geom1, sample.geom2) for sample in frame.contacts)
    for sample in frame.contacts:
        assert sample.physics_step > 0
        assert np.linalg.norm(sample.normal_from_geom1_to_geom2) == pytest.approx(1.0, abs=1e-12)
        assert sample.impulse_n_s == pytest.approx(
            np.linalg.norm(sample.force_on_geom2_world_n) * 0.005,
            rel=1e-12,
            abs=1e-15,
        )
        assert sample.penetration_m == pytest.approx(max(0.0, -sample.signed_distance_m))
        assert sample.geom1_groups == tuple(sorted(sample.geom1_groups))
        assert sample.geom2_groups == tuple(sorted(sample.geom2_groups))
    assert frame.maximum_contact_force_n == pytest.approx(
        max(np.linalg.norm(sample.force_on_geom2_world_n) for sample in frame.contacts)
    )
    assert frame.maximum_contact_impulse_n_s == pytest.approx(max(sample.impulse_n_s for sample in frame.contacts))


def test_right_arm_table_contact_force_limit_is_inclusive(
    config: ResolvedRunConfig,
) -> None:
    limit = config.task.termination.right_arm_table_contact_force_limit_n
    assert not is_illegal_contact(
        ("right_arm",),
        ("table",),
        limit,
        right_arm_table_force_limit_n=limit,
    )
    assert is_illegal_contact(
        ("right_arm",),
        ("table",),
        np.nextafter(limit, np.inf),
        right_arm_table_force_limit_n=limit,
    )
    assert is_illegal_contact(
        ("right_arm",),
        ("cylinder",),
        0.1,
        right_arm_table_force_limit_n=limit,
    )
    assert is_illegal_contact(
        ("tcp",),
        ("table",),
        0.1,
        right_arm_table_force_limit_n=limit,
    )
    assert is_illegal_contact(
        ("right_arm", "non_task_body"),
        ("table",),
        0.1,
        right_arm_table_force_limit_n=limit,
    )


def test_hand_force_norms_and_illegal_impulse_are_reproducible_aggregations(
    backend: MujocoEnvironmentBackend,
    config: ResolvedRunConfig,
) -> None:
    state = backend.reset(ResetSpec())
    backend.apply_targets(np.array(state.actuator_targets_rad, copy=True))
    frame = backend.step_substeps(4).contacts
    assert frame.hand_link_names == (
        "right_thumb_1",
        "right_thumb_2",
        "right_thumb_3",
        "right_thumb_4",
        "right_index_1",
        "right_index_2",
        "right_middle_1",
        "right_middle_2",
        "right_ring_1",
        "right_ring_2",
        "right_little_1",
        "right_little_2",
    )
    reconstructed = np.zeros((4, 12, 3), dtype=np.float64)
    hand_indices = {name: index for index, name in enumerate(frame.hand_link_names)}
    for sample in frame.contacts:
        if sample.body1 in hand_indices:
            reconstructed[sample.substep_index, hand_indices[sample.body1]] -= sample.force_on_geom2_world_n
        if sample.body2 in hand_indices:
            reconstructed[sample.substep_index, hand_indices[sample.body2]] += sample.force_on_geom2_world_n
    expected_hand_norms = np.linalg.norm(reconstructed, axis=2).max(axis=0)
    np.testing.assert_allclose(frame.hand_contact_force_norms_n, expected_hand_norms, atol=1e-12)

    illegal_impulses = [
        sample.impulse_n_s
        for sample in frame.contacts
        if is_illegal_contact(
            sample.geom1_groups,
            sample.geom2_groups,
            float(np.linalg.norm(sample.force_on_geom2_world_n)),
            right_arm_table_force_limit_n=config.task.termination.right_arm_table_contact_force_limit_n,
        )
    ]
    # A collision-calibrated reset need not start in illegal contact.  The
    # diagnostic contract is the maximum illegal impulse, or zero when none
    # occurred during this step group.
    assert frame.maximum_illegal_contact_impulse_n_s == pytest.approx(max(illegal_impulses, default=0.0))


def test_warning_and_contact_capacity_diagnostics_are_structured_and_reset(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    data = backend._data
    assert data is not None
    bad_velocity = 4
    contact_full = 1
    data.warning[bad_velocity].lastinfo = 12
    data.warning[bad_velocity].number = 2
    data.warning[contact_full].lastinfo = 4096
    data.warning[contact_full].number = 1
    diagnostics = backend.read_warnings()
    assert diagnostics.solver_anomaly
    assert diagnostics.contact_capacity_exceeded
    assert diagnostics.nonfinite_state is False
    assert len(diagnostics.warnings) == 2
    by_code = {warning.code: warning for warning in diagnostics.warnings}
    assert by_code[bad_velocity].occurrence_count == 2
    assert "QVEL" in by_code[bad_velocity].message
    assert by_code[bad_velocity].physics_step == diagnostics.physics_step
    assert by_code[bad_velocity].config_fingerprint == backend.config_fingerprint
    assert backend.read_warnings().warnings == diagnostics.warnings

    backend.reset(ResetSpec())
    assert backend.read_warnings().healthy
    assert backend.read_warnings().warnings == ()


def test_substep_trace_preserves_solver_anomaly_without_masking_diagnostics(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    data = backend._data
    assert data is not None
    bad_velocity = 4
    data.warning[bad_velocity].lastinfo = 12
    data.warning[bad_velocity].number = 1
    assert backend.read_warnings().solver_anomaly

    result = backend.step_substeps(4)
    assert result.diagnostics.solver_anomaly
    assert not result.substep_trace.finite.any()
    assert np.all(np.diff(result.substep_trace.simulation_times_s) > 0.0)


def test_substep_trace_time_validation_distinguishes_healthy_and_anomalous() -> None:
    common = {
        "cylinder_positions_m": np.zeros((4, 3), dtype=np.float64),
        "cylinder_quaternions_wxyz": np.repeat(
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            4,
            axis=0,
        ),
        "cylinder_linear_velocities_m_s": np.zeros((4, 3), dtype=np.float64),
        "cylinder_angular_velocities_rad_s": np.zeros((4, 3), dtype=np.float64),
        "simulation_times_s": np.asarray((0.005, 0.010, 0.010, 0.010), dtype=np.float64),
    }
    anomalous = PhysicsSubstepTrace(
        **common,
        finite=np.asarray((True, False, False, False), dtype=np.bool_),
    )
    np.testing.assert_array_equal(anomalous.simulation_times_s, common["simulation_times_s"])
    assert not anomalous.simulation_times_s.flags.writeable

    with pytest.raises(ValueError, match=r"healthy.*strictly increasing"):
        PhysicsSubstepTrace(**common, finite=np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="non-decreasing"):
        PhysicsSubstepTrace(
            **{**common, "simulation_times_s": np.asarray((0.005, 0.010, 0.009, 0.010))},
            finite=np.asarray((True, False, False, False), dtype=np.bool_),
        )


def test_nonfinite_live_state_fails_with_diagnostics(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    data = backend._data
    assert data is not None
    data.qpos[0] = np.nan
    with pytest.raises(NonFinitePhysicsError, match="physical step") as captured:
        backend.read_state()
    assert captured.value.diagnostics.nonfinite_state
    assert captured.value.diagnostics.config_fingerprint == backend.config_fingerprint


@pytest.mark.parametrize(
    "parameters",
    [
        CylinderParameters(
            radius_m=0.012,
            height_m=0.080,
            mass_kg=0.100,
            hand_sliding_friction=0.7,
            table_sliding_friction=1.1,
            torsional_friction=0.005,
            rolling_friction=0.0001,
        ),
        CylinderParameters(
            radius_m=0.015,
            height_m=0.120,
            mass_kg=0.130,
            hand_sliding_friction=1.1,
            table_sliding_friction=0.7,
            torsional_friction=0.005,
            rolling_friction=0.0001,
        ),
    ],
)
def test_cylinder_geometry_mass_solid_inertia_and_friction_update(
    backend: MujocoEnvironmentBackend,
    parameters: CylinderParameters,
) -> None:
    z = 0.94 + 0.002 + parameters.height_m / 2.0
    state = backend.reset(
        ResetSpec(
            cylinder_position_m=(0.5252919, -0.26195379, z),
            cylinder=parameters,
        )
    )
    assert state.cylinder_parameters == parameters
    np.testing.assert_allclose(state.cylinder_position_m, (0.5252919, -0.26195379, z))
    model = backend._model
    assert model is not None
    geom_id = backend.name_cache.cylinder_geom_id
    body_id = backend.name_cache.cylinder_body_id
    half_side = parameters.radius_m / np.sqrt(2.0)
    np.testing.assert_allclose(model.geom_size[geom_id, :3], (half_side, half_side, parameters.height_m / 2))
    assert model.body_mass[body_id] == pytest.approx(parameters.mass_kg)
    np.testing.assert_allclose(model.body_inertia[body_id], parameters.solid_inertia_kg_m2, rtol=1e-14)
    # ``body_invweight0`` is derived by ``mj_setConst``; its translational
    # component proves the runtime update refreshed more than source arrays.
    assert model.body_invweight0[body_id, 0] == pytest.approx(1.0 / parameters.mass_kg)
    assert model.geom_friction[geom_id, 0] == pytest.approx(
        min(parameters.hand_sliding_friction, parameters.table_sliding_friction)
    )
    for geom_id in backend._contact_groups.hand_geom_ids:
        np.testing.assert_allclose(
            model.geom_friction[geom_id],
            (
                parameters.hand_sliding_friction,
                parameters.torsional_friction,
                parameters.rolling_friction,
            ),
        )
    for geom_id in backend._contact_groups.table_geom_ids:
        np.testing.assert_allclose(
            model.geom_friction[geom_id],
            (
                parameters.table_sliding_friction,
                parameters.torsional_friction,
                parameters.rolling_friction,
            ),
        )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("radius_m", 0.0),
        ("height_m", math.inf),
        ("mass_kg", -1.0),
        ("hand_sliding_friction", -0.1),
    ],
)
def test_invalid_cylinder_parameter_construction_fails(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match=keyword):
        CylinderParameters(**{keyword: value})


def test_named_actuator_ranges_and_site_hinge_kinematics_are_readonly(
    backend: MujocoEnvironmentBackend,
) -> None:
    backend.reset(ResetSpec())
    ranges = backend.read_actuator_control_ranges()
    assert ranges.actuator_names == backend.read_state().actuator_names
    assert ranges.ranges_rad.shape == (13, 2)
    assert ranges.ranges_rad.dtype == np.float64
    assert ranges.ranges_rad.flags.c_contiguous
    assert not ranges.ranges_rad.flags.writeable
    assert np.isfinite(ranges.ranges_rad).all()
    assert np.all(ranges.lower_rad < ranges.upper_rad)
    with pytest.raises(ValueError, match="read-only"):
        ranges.ranges_rad[0, 0] = 0.0

    kinematics = backend.read_site_hinge_kinematics("right_tcp_site", "elbow_pitch_r_joint")
    assert kinematics.site_jacobian_6d.shape == (6,)
    assert kinematics.site_jacobian_6d.dtype == np.float64
    assert kinematics.site_jacobian_6d.flags.c_contiguous
    assert not kinematics.site_jacobian_6d.flags.writeable
    assert np.isfinite(kinematics.site_jacobian_6d).all()
    assert kinematics.joint_lower_rad <= kinematics.joint_position_rad <= kinematics.joint_upper_rad
    with pytest.raises(KeyError, match="unknown site"):
        backend.read_site_hinge_kinematics("missing_site", "elbow_pitch_r_joint")
    with pytest.raises(BackendContractError, match="not a scalar hinge"):
        backend.read_site_hinge_kinematics("right_tcp_site", "cylinder_free_joint")
