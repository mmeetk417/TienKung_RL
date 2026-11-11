"""M02 deterministic MuJoCo asset-pipeline acceptance tests."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import mujoco  # type: ignore[import-untyped]
import numpy as np

from tienkung_rl.assets import pipeline
from tienkung_rl.config.load import DEFAULT_CONFIG_PATH, load_run_config
from tienkung_rl.config.models import ResolvedRunConfig

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "tienkung_rl" / "assets" / "tienkung2_pro"
BASE_GOLDEN_SHA256 = "686b720e33bf9372425fff79e1091c68d24da2f41ec7f20612c3acf112cffafb"


class SourceInventoryTests(unittest.TestCase):
    def test_source_is_byte_exact_traceable_and_complete(self) -> None:
        inventory = pipeline.inspect_source_assets()
        self.assertEqual(len(inventory.joints), 54)
        self.assertEqual(len(inventory.links), 63)
        self.assertEqual(len(inventory.mesh_paths), 63)
        self.assertEqual(len(inventory.mesh_references), 126)
        self.assertEqual(len(inventory.mimic), 12)
        self.assertIn("right_tcp_link", inventory.links)
        self.assertEqual((pipeline.SOURCE_DIR / "LICENSE").read_bytes(), (ROOT / "LICENSE").read_bytes())

        manifest = json.loads((pipeline.SOURCE_DIR / pipeline.SOURCE_MANIFEST_NAME).read_text())
        self.assertEqual(manifest["counts"]["movable_joints"], 54)
        self.assertEqual(manifest["counts"]["distinct_meshes"], 63)
        for item in [manifest["urdf"], *manifest["meshes"], *manifest["notices"]]:
            path = ASSET / item["path"]
            self.assertEqual(item["size_bytes"], path.stat().st_size)
            self.assertEqual(item["sha256"], pipeline.sha256_file(path))

    def test_missing_and_case_mismatched_mesh_references_fail(self) -> None:
        source_root = ET.parse(pipeline.SOURCE_URDF).getroot()
        first_mesh = source_root.find(".//mesh")
        self.assertIsNotNone(first_mesh)
        assert first_mesh is not None
        original = first_mesh.get("filename")
        self.assertIsNotNone(original)
        assert original is not None
        with tempfile.TemporaryDirectory(prefix="m02-source-negative-") as temporary:
            urdf = Path(temporary) / "probe.urdf"
            for replacement in (original.replace("pelvis.STL", "Pelvis.STL"), "../meshes/missing.STL"):
                with self.subTest(reference=replacement):
                    first_mesh.set("filename", replacement)
                    ET.ElementTree(source_root).write(urdf, encoding="utf-8", xml_declaration=True)
                    with self.assertRaisesRegex(pipeline.SourceAssetError, "missing or case-mismatched"):
                        pipeline.inspect_source_assets(urdf, pipeline.VISUAL_MESH_DIR)
            first_mesh.set("filename", original)


class PatchContractTests(unittest.TestCase):
    def test_bundle_schema_and_source_derived_mimic_are_strict(self) -> None:
        bundle = pipeline.load_patch_bundle()
        inventory = pipeline.inspect_source_assets()
        self.assertEqual(bundle.main["application_order"][0], "fixed_base")
        self.assertEqual(bundle.main["application_order"][-1], "metadata")
        self.assertEqual(bundle.main["patch_version"], "tienkung2_pro_mjcf_patch_v5")
        self.assertEqual(bundle.main["asset_contract_version"], "tienkung2_pro_mujoco_asset_v3")
        self.assertEqual(bundle.main["gravity_compensation"]["root_body"], "shoulder_pitch_r_link")
        self.assertEqual(bundle.main["gravity_compensation"]["factor"], 1.0)
        self.assertEqual(bundle.collisions["calibration"]["version"], "simplified_collision_calibration_v2")
        self.assertEqual(bundle.main["mimic"]["source"], "urdf.joint.mimic")
        self.assertEqual(
            bundle.main["mimic"]["constraint_calibration_version"],
            "right_hand_mimic_equality_v2",
        )
        self.assertEqual(bundle.main["mimic"]["solref_direct"], [-10000000.0, -20000.0])
        right_mimic = {name: value for name, value in inventory.mimic.items() if name.startswith("right_")}
        self.assertEqual(len(right_mimic), 6)
        self.assertEqual(right_mimic["right_thumb_3_joint"], ("right_thumb_2_joint", 1.1425, 0.0))

    def test_duplicates_overwrites_and_unknown_targets_fail(self) -> None:
        for namespace in ("actuator", "site"):
            with self.subTest(namespace=namespace):
                tracker = pipeline.PatchTracker()
                tracker.claim(namespace, "duplicate", "first", 1)
                with self.assertRaises(pipeline.PatchConflictError):
                    tracker.claim(namespace, "duplicate", "second", 1)
        tracker = pipeline.PatchTracker()
        element = ET.Element("body")
        tracker.set_attribute(element, "pos", "0 0 0", source="first", target="body:probe")
        with self.assertRaises(pipeline.PatchConflictError):
            tracker.set_attribute(element, "pos", "1 0 0", source="second", target="body:probe")
        with self.assertRaises(pipeline.UnknownPatchTargetError):
            pipeline._find_body(ET.fromstring("<mujoco><worldbody/></mujoco>"), "absent", "test")

    def test_duplicate_yaml_keys_fail(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m02-yaml-negative-") as temporary:
            path = Path(temporary) / "duplicate.yaml"
            path.write_text("schema_version: one\nschema_version: two\n", encoding="utf-8")
            with self.assertRaisesRegex(pipeline.PatchSchemaError, "duplicate YAML key"):
                pipeline._load_yaml(path)


class GeneratedAssetTests(unittest.TestCase):
    config: ClassVar[ResolvedRunConfig]
    runtime_path: ClassVar[Path]
    runtime_root: ClassVar[ET.Element]
    model: ClassVar[Any]
    manifest: ClassVar[dict[str, Any]]
    name_index: ClassVar[dict[str, Any]]
    hierarchy: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_run_config(DEFAULT_CONFIG_PATH)
        cls.runtime_path = ASSET / "mjcf" / pipeline.RUNTIME_MJCF_NAME
        cls.runtime_root = ET.parse(cls.runtime_path).getroot()
        cls.model = mujoco.MjModel.from_xml_path(str(cls.runtime_path))
        cls.manifest = json.loads((ASSET / pipeline.GENERATION_MANIFEST_NAME).read_text())
        cls.name_index = json.loads((ASSET / "mjcf" / pipeline.NAME_INDEX_NAME).read_text())
        cls.hierarchy = json.loads((ASSET / "mjcf" / pipeline.JOINT_HIERARCHY_NAME).read_text())

    def test_golden_hashes_and_manifest_are_current(self) -> None:
        base = ASSET / "mjcf" / pipeline.BASE_MJCF_NAME
        self.assertEqual(pipeline.sha256_file(base), BASE_GOLDEN_SHA256)
        self.assertEqual(self.manifest["tool"]["version"], "3.11.0")
        self.assertEqual(mujoco.__version__, "3.11.0")
        for relative, digest in self.manifest["output_hashes"].items():
            self.assertEqual(pipeline.sha256_file(ASSET / relative), digest)
        for filename, digest in self.manifest["patch_hashes"].items():
            self.assertEqual(pipeline.sha256_file(ASSET / "patches" / filename), digest)

    def test_two_clean_generations_are_byte_identical(self) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="m02-generation-a-") as first,
            tempfile.TemporaryDirectory(prefix="m02-generation-b-") as second,
        ):
            result_a = pipeline.generate_assets(self.config, output_root=Path(first))
            result_b = pipeline.generate_assets(self.config, output_root=Path(second))
            self.assertEqual(result_a.hashes, result_b.hashes)
            for relative in pipeline.GENERATED_RELATIVE_PATHS:
                self.assertEqual((Path(first) / relative).read_bytes(), (Path(second) / relative).read_bytes())

    def test_runtime_preserves_joint_contract_and_has_7_plus_6_actuators(self) -> None:
        source_names = [entry["name"] for entry in self.hierarchy["joints"]]
        runtime_joint_names = set(self.name_index["indices"]["joints"])
        self.assertEqual(self.hierarchy["movable_joint_count"], 54)
        self.assertTrue(set(source_names).issubset(runtime_joint_names))
        self.assertEqual(
            Counter(entry["runtime_role"] for entry in self.hierarchy["joints"]),
            {
                "active_arm": 7,
                "active_hand": 6,
                "mimic_follower": 6,
                "fixed_non_task": 35,
            },
        )
        self.assertEqual(self.model.njnt, 55)
        self.assertEqual(self.model.nu, 13)
        actuator_elements = self.runtime_root.findall("./actuator/position")
        self.assertEqual(len(actuator_elements), 13)
        expected_joints = (
            *self.config.physics.actuator_calibration.arm.joint_names,
            *self.config.physics.actuator_calibration.hand.joint_names,
        )
        self.assertEqual(tuple(element.get("joint") for element in actuator_elements), expected_joints)
        for element in actuator_elements:
            for attribute in ("kp", "kv", "ctrlrange", "forcerange", "user"):
                self.assertIn(attribute, element.attrib)

    def test_mimic_sites_frames_and_name_index_are_complete(self) -> None:
        source_mimic = {
            entry["name"]: (
                entry["mimic_joint"],
                entry["mimic_multiplier"],
                entry["mimic_offset"],
            )
            for entry in self.hierarchy["joints"]
            if entry["mimic_joint"] is not None and entry["name"].startswith("right_")
        }
        equalities = {
            element.get("joint1"): element
            for element in self.runtime_root.findall("./equality/joint")
            if element.get("name", "").startswith("mimic__")
        }
        self.assertEqual(set(equalities), set(source_mimic))
        for follower, (leader, multiplier, offset) in source_mimic.items():
            equality = equalities[follower]
            polycoef = tuple(float(item) for item in equality.get("polycoef", "").split())
            self.assertEqual(equality.get("joint2"), leader)
            self.assertEqual(polycoef[:2], (offset, multiplier))
        required_sites = {
            "right_tcp_site",
            "grasp_center_site",
            "approach_direction_site",
            "target_tcp_site",
            "cylinder_axis_site",
        }
        self.assertTrue(required_sites.issubset(self.name_index["indices"]["sites"]))
        self.assertEqual(set(self.name_index["declared_frames"]), {"right_tcp_frame", "target_tcp_frame"})

    def test_right_hand_mesh_collisions_and_contact_groups_cover_task(self) -> None:
        viewer_root = ET.parse(ASSET / "mjcf" / pipeline.VIEWER_MJCF_NAME).getroot()
        mesh_geoms = [geom for geom in viewer_root.findall(".//geom") if geom.get("type") == "mesh"]
        self.assertEqual(len(mesh_geoms), 76)
        visual_meshes = [geom for geom in mesh_geoms if geom.get("contype") == "0" and geom.get("conaffinity") == "0"]
        physical_meshes = [geom for geom in mesh_geoms if geom.get("contype") != "0" or geom.get("conaffinity") != "0"]
        self.assertEqual(len(visual_meshes), 63)
        self.assertEqual(len(physical_meshes), 13)
        self.assertTrue(all(geom.get("group") == "1" for geom in visual_meshes))
        self.assertTrue(all(geom.get("group") == "3" for geom in physical_meshes))
        collision_geoms = [geom for geom in self.runtime_root.findall(".//geom") if geom.get("group") == "3"]
        self.assertTrue(collision_geoms)
        self.assertTrue(any(geom.get("type") == "mesh" for geom in collision_geoms))
        self.assertEqual(sum(geom.get("type") == "mesh" for geom in collision_geoms), 13)
        self.assertEqual(self.model.nmesh, 13)
        runtime_products = self.manifest["runtime_products"]
        self.assertFalse(runtime_products["headless"]["visual_meshes_compiled"])
        self.assertTrue(runtime_products["viewer"]["visual_meshes_compiled"])
        self.assertEqual(runtime_products["headless"]["collision_meshes_compiled"], 13)
        self.assertEqual(runtime_products["headless"]["model_summary"]["nmesh"], 13)
        self.assertEqual(runtime_products["viewer"]["model_summary"]["nmesh"], 63)
        groups = self.name_index["contact_groups"]
        self.assertEqual(len(groups["right_hand_observation_links"]["members"]), 12)
        for name in (
            "thumb_side",
            "four_finger_opposing",
            "right_arm",
            "tcp",
            "reach_diagnostic",
            "robot",
            "table",
            "cylinder",
            "non_task_body",
        ):
            self.assertIn(name, groups)

    def test_scene_and_solver_values_are_injected_from_m01(self) -> None:
        option = self.runtime_root.find("option")
        self.assertIsNotNone(option)
        assert option is not None
        self.assertEqual(float(option.get("timestep", "nan")), self.config.physics.physics_dt_s)
        self.assertEqual(
            tuple(float(item) for item in option.get("gravity", "").split()),
            self.config.physics.gravity_m_s2,
        )
        self.assertEqual(int(option.get("iterations", "-1")), self.config.physics.solver.position_iterations)
        for name in ("table", "cylinder"):
            self.assertIn(name, self.name_index["indices"]["bodies"])
        for name in ("table_top_collision", "table_leg_collision_0", "cylinder_collision"):
            self.assertIn(name, self.name_index["indices"]["geoms"])
        self.assertEqual(self.manifest["scene"]["cylinder_nominal_mass_kg"], 0.115)

        floor = next(geom for geom in self.runtime_root.findall(".//geom") if geom.get("name") == "floor")
        self.assertEqual(
            {floor.get("type"), floor.get("material"), floor.get("group")},
            {"plane", "groundplane", "1"},
        )
        self.assertEqual((floor.get("contype"), floor.get("conaffinity")), ("0", "0"))
        texture = self.runtime_root.find("./asset/texture[@name='groundplane']")
        self.assertIsNotNone(texture)
        assert texture is not None
        self.assertEqual((texture.get("builtin"), texture.get("mark")), ("checker", "edge"))
        self.assertEqual((texture.get("width"), texture.get("height")), ("300", "300"))
        self.assertEqual(self.manifest["scene"]["floor"]["source"], "rmm-sim/mj_assets/stanford_tidybot2/scene.xml")

    def test_home_keyframe_is_finite_stable_and_not_penetrating(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        mujoco.mj_forward(self.model, data)
        penetrations = [data.contact[index].dist for index in range(data.ncon) if data.contact[index].dist < -1.0e-8]
        self.assertEqual(penetrations, [])
        initial_qpos = data.qpos.copy()
        for _ in range(100):
            mujoco.mj_step(self.model, data)
        self.assertTrue(np.isfinite(data.qpos).all())
        self.assertTrue(np.isfinite(data.qvel).all())
        roles = {entry["name"]: entry["runtime_role"] for entry in self.hierarchy["joints"]}
        fixed_ids = [
            self.name_index["indices"]["joints"][name] for name, role in roles.items() if role == "fixed_non_task"
        ]
        self.assertLess(float(np.max(np.abs(data.qpos[fixed_ids] - initial_qpos[fixed_ids]))), 0.01)
        self.assertFalse(np.any(data.warning.number))

    def test_active_right_arm_is_gravity_compensated_with_safe_home_clearance(self) -> None:
        root_id = self.name_index["indices"]["bodies"]["shoulder_pitch_r_link"]
        compensated_ids = {root_id}
        frontier = [root_id]
        while frontier:
            parent = frontier.pop()
            children = np.flatnonzero(self.model.body_parentid == parent)
            compensated_ids.update(int(child) for child in children)
            frontier.extend(int(child) for child in children)
        np.testing.assert_allclose(self.model.body_gravcomp[sorted(compensated_ids)], 1.0)

        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, 0)
        mujoco.mj_forward(self.model, data)
        clearance = pipeline._home_robot_table_clearance(self.model, self.name_index["contact_groups"])
        self.assertGreaterEqual(clearance, 0.03)
        initial_qpos = data.qpos.copy()
        illegal_groups = {"right_arm", "tcp", "non_task_body"}
        for _ in range(400):
            mujoco.mj_step(self.model, data)
            for index in range(data.ncon):
                first = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(data.contact[index].geom1),
                )
                second = mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(data.contact[index].geom2),
                )
                groups = {
                    name
                    for name, spec in self.name_index["contact_groups"].items()
                    if first in spec["members"] or second in spec["members"]
                }
                self.assertTrue(illegal_groups.isdisjoint(groups), (first, second, groups))
        joint_ids = self.model.actuator_trnid[:7, 0]
        qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.assertLess(
            float(np.max(np.abs(data.qpos[qpos_addresses] - initial_qpos[qpos_addresses]))),
            1.0e-8,
        )

    def test_open_half_and_full_close_track_mimic_and_active_endpoints(self) -> None:
        pairs = [
            (
                entry["name"],
                entry["mimic_joint"],
                float(entry["mimic_multiplier"]),
                float(entry["mimic_offset"]),
            )
            for entry in self.hierarchy["joints"]
            if entry["name"].startswith("right_") and entry["mimic_joint"] is not None
        ]
        self.assertEqual(len(pairs), 6)
        joint_ids = self.name_index["indices"]["joints"]
        measured: dict[float, float] = {}
        active_joint_ids = self.model.actuator_trnid[7:, 0]
        for close_fraction in (0.0, 0.5, 1.0):
            data = mujoco.MjData(self.model)
            mujoco.mj_resetDataKeyframe(self.model, data, 0)
            mujoco.mj_forward(self.model, data)
            start = data.ctrl.copy()
            target = start.copy()
            lower = self.model.actuator_ctrlrange[7:, 0]
            upper = self.model.actuator_ctrlrange[7:, 1]
            target[7:] = lower + close_fraction * (upper - lower)
            for step in range(160):  # 0.8 s at the M01 0.005 s physics step
                alpha = (step + 1) / 160.0
                data.ctrl[:] = start + alpha * (target - start)
                mujoco.mj_step(self.model, data)
            for _ in range(200):  # one-second loaded hold
                mujoco.mj_step(self.model, data)
            errors = [
                abs(
                    data.qpos[self.model.jnt_qposadr[joint_ids[follower]]]
                    - (multiplier * data.qpos[self.model.jnt_qposadr[joint_ids[leader]]] + offset)
                )
                for follower, leader, multiplier, offset in pairs
            ]
            measured[close_fraction] = max(errors)
            self.assertLess(measured[close_fraction], 0.01)
            active_positions = np.asarray(
                [data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in active_joint_ids]
            )
            active_ranges = self.model.jnt_range[active_joint_ids]
            self.assertTrue(np.all(active_positions >= active_ranges[:, 0] - 0.01))
            self.assertTrue(np.all(active_positions <= active_ranges[:, 1] + 0.01))
            self.assertLess(float(np.max(np.abs(active_positions - data.ctrl[7:]))), 0.01)
            self.assertTrue(np.isfinite(data.qpos).all())
            self.assertFalse(np.any(data.warning.number))
        self.assertLess(measured[0.0], 0.01)
        self.assertLess(measured[0.5], 0.01)
        self.assertLess(measured[1.0], 0.01)

    def test_check_detects_drift_without_overwriting_it(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m02-check-drift-") as temporary:
            committed = Path(temporary)
            for relative in pipeline.GENERATED_RELATIVE_PATHS:
                destination = committed / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ASSET / relative, destination)
            drifted = committed / "mjcf" / pipeline.RUNTIME_MJCF_NAME
            marker = b"\n<!-- deliberate test drift -->\n"
            drifted.write_bytes(drifted.read_bytes() + marker)
            before = drifted.read_bytes()
            with self.assertRaisesRegex(pipeline.AssetPipelineError, "content drift"):
                pipeline.check_generated_assets(self.config, committed_root=committed)
            self.assertEqual(drifted.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
