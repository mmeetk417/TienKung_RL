"""M03 real-MuJoCo asset-validation acceptance and failure tests."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import ClassVar

from tienkung_rl.assets import pipeline, validation
from tienkung_rl.config.load import DEFAULT_CONFIG_PATH, load_run_config
from tienkung_rl.config.models import ResolvedRunConfig

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "tienkung_rl" / "assets" / "tienkung2_pro"


class StructuralFailureFixtureTests(unittest.TestCase):
    inventory: ClassVar[pipeline.SourceInventory]
    runtime_root: ClassVar[ET.Element]

    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = pipeline.inspect_source_assets()
        cls.runtime_root = ET.parse(pipeline.MJCF_DIR / pipeline.RUNTIME_MJCF_NAME).getroot()

    def test_joint_hierarchy_and_type_contract_passes_for_committed_asset(self) -> None:
        result = validation.compare_joint_structure(self.inventory, self.runtime_root)
        self.assertTrue(result.passed, result.to_dict())
        self.assertEqual(result.measurements["source_movable_joint_count"], 54)
        self.assertEqual(result.measurements["runtime_robot_joint_count"], 54)

    def test_axis_flip_is_reported_by_name_and_in_explicit_units(self) -> None:
        root = copy.deepcopy(self.runtime_root)
        element = root.find(".//joint[@name='shoulder_pitch_r_joint']")
        self.assertIsNotNone(element)
        assert element is not None
        element.set("axis", "0 -1 0")
        result = validation.compare_joint_axes_and_limits(self.inventory, root)
        issues = [issue for issue in result.issues if issue.code == "joint.axis"]
        self.assertFalse(result.passed)
        self.assertEqual([issue.object_name for issue in issues], ["shoulder_pitch_r_joint"])
        self.assertEqual(issues[0].unit, "absolute axis component")
        self.assertEqual(result.measurements["limit_unit"], "rad")

    def test_missing_limit_fails_closed(self) -> None:
        root = copy.deepcopy(self.runtime_root)
        element = root.find(".//joint[@name='wrist_roll_r_joint']")
        self.assertIsNotNone(element)
        assert element is not None
        element.attrib.pop("range")
        element.set("limited", "false")
        result = validation.compare_joint_axes_and_limits(self.inventory, root)
        issues = [issue for issue in result.issues if issue.code == "joint.limit_missing"]
        self.assertFalse(result.passed)
        self.assertEqual([issue.object_name for issue in issues], ["wrist_roll_r_joint"])

    def test_tracking_overshoot_is_relative_to_settled_joint_for_both_directions(self) -> None:
        positive_peak, positive_overshoot = validation.tracking_response_metrics(
            (1.20, 1.21, 1.225, 1.219),
            settled_position_rad=1.20,
            target_position_rad=1.22,
            direction=1.0,
        )
        negative_peak, negative_overshoot = validation.tracking_response_metrics(
            (-0.40, -0.41, -0.427, -0.42),
            settled_position_rad=-0.40,
            target_position_rad=-0.42,
            direction=-1.0,
        )
        self.assertAlmostEqual(positive_peak, 0.025)
        self.assertAlmostEqual(positive_overshoot, 0.005)
        self.assertAlmostEqual(negative_peak, 0.027)
        self.assertAlmostEqual(negative_overshoot, 0.007)


class ValidationReportTests(unittest.TestCase):
    config: ClassVar[ResolvedRunConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_run_config(DEFAULT_CONFIG_PATH)

    def test_full_validator_and_cli_report_pass_without_unexplained_warnings(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m03-report-") as temporary:
            report_path = Path(temporary) / "asset-validation.json"
            arguments = argparse.Namespace(report=report_path)
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                exit_code = validation.run(self.config, arguments)

            self.assertEqual(exit_code, 0, terminal.getvalue())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], validation.REPORT_SCHEMA_VERSION)
            self.assertEqual(payload["validator_version"], validation.VALIDATOR_VERSION)
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["error_count"], 0)
            self.assertEqual(payload["unexplained_warning_count"], 0)
            check_ids = [check["check_id"] for check in payload["checks"]]
            self.assertEqual(
                check_ids,
                [
                    "joint_structure",
                    "joint_axes_limits",
                    "actuators",
                    "hand_mimic",
                    "named_objects",
                    "runtime_variants",
                    "stl_visual_only",
                    "inertials",
                    "initial_pose",
                    "collision_coverage",
                    "initial_penetration",
                    "zero_action_settle",
                    "arm_tracking",
                    "hand_motion",
                    "contact_probes",
                ],
            )
            coverage_path = report_path.with_name("asset-validation.collision_coverage.svg")
            self.assertEqual(payload["artifacts"]["collision_coverage_svg"], coverage_path.name)
            self.assertTrue(coverage_path.is_file())
            self.assertIn("simplified-collision coverage", coverage_path.read_text(encoding="utf-8"))
            checks = {check["check_id"]: check for check in payload["checks"]}
            actuator_measurements = checks["actuators"]["measurements"]
            self.assertEqual(actuator_measurements["physics_version"], "mujoco_cpu_physics_v3")
            self.assertEqual(actuator_measurements["physics_calibration_status"], "starting_point_unvalidated")
            self.assertEqual(
                actuator_measurements["actuator_calibration_version"],
                "mujoco_position_actuator_calibration_v2",
            )
            self.assertEqual(
                actuator_measurements["actuator_calibration_status"],
                "starting_point_unvalidated",
            )
            variant_measurements = checks["runtime_variants"]["measurements"]
            self.assertEqual(variant_measurements["headless_mesh_count"], 13)
            self.assertEqual(variant_measurements["viewer_mesh_count"], 63)
            self.assertIn("[PASS] asset validation", terminal.getvalue())

    def test_runtime_drift_is_machine_readable_and_does_not_modify_input(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m03-drift-") as temporary:
            repository_root = Path(temporary)
            copied_asset = repository_root / "tienkung_rl" / "assets" / "tienkung2_pro"
            shutil.copytree(ASSET_ROOT, copied_asset)
            runtime = copied_asset / "mjcf" / pipeline.RUNTIME_MJCF_NAME
            runtime.write_bytes(runtime.read_bytes() + b"\n<!-- deliberate M03 drift -->\n")
            before = runtime.read_bytes()

            report = validation.validate_assets(
                self.config,
                repository_root=repository_root,
                runtime_mjcf=runtime,
            )

            baseline = validation.validate_assets(self.config)
            self.assertEqual(report.passed, baseline.passed)
            self.assertEqual(
                [(check.check_id, check.passed) for check in report.checks],
                [(check.check_id, check.passed) for check in baseline.checks],
            )
            self.assertEqual(runtime.read_bytes(), before)
            self.assertNotIn("hash_integrity", {check.check_id for check in report.checks})

    def test_preflight_failure_has_stable_nonzero_report_shape(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m03-missing-") as temporary:
            report = validation.validate_assets(self.config, repository_root=Path(temporary))
        self.assertFalse(report.passed)
        self.assertEqual(len(report.checks), 1)
        self.assertEqual(report.checks[0].check_id, "preflight")
        self.assertEqual(report.checks[0].issues[0].code, "preflight.load")
        self.assertEqual(report.runtime_mjcf_sha256, "unavailable")
        self.assertTrue(report.terminal_lines()[-1].startswith("[FAIL] asset validation:"))


if __name__ == "__main__":
    unittest.main()
