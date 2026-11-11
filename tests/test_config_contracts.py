"""Pure-standard-library tests for M01 configuration contracts."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import ClassVar

from tienkung_rl.config.load import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    DuplicateKeyError,
    load_run_config,
    write_config_snapshot,
)
from tienkung_rl.config.models import ResolvedRunConfig
from tienkung_rl.config.serialization import canonical_json


class ConfigurationContractTests(unittest.TestCase):
    default_text: ClassVar[str]
    config: ClassVar[ResolvedRunConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.default_text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        cls.config = load_run_config(DEFAULT_CONFIG_PATH)

    def _load_text(
        self,
        text: str,
        *,
        overrides: tuple[str, ...] = (),
    ) -> ResolvedRunConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(text, encoding="utf-8")
            return load_run_config(path, overrides=overrides)

    def test_default_is_frozen_and_satisfies_dimensions_and_timing(self) -> None:
        config = self.config
        self.assertEqual(config.task.observation.dimension, 106)
        self.assertEqual(config.task.action.dimension, 7)
        self.assertTrue(config.task.option.automatic_trigger_on_admission_dwell)
        self.assertEqual(config.task.admission.dwell_time_s, 0.30)
        self.assertEqual(config.task.termination.right_arm_table_contact_force_limit_n, 5.0)
        self.assertEqual(config.physics.policy_dt_s, 0.020)
        self.assertEqual(config.physics.physics_dt_s * config.physics.decimation, 0.020)
        self.assertEqual(
            sum(
                round(value / config.physics.policy_dt_s)
                for value in (
                    config.task.option.settle_time_s,
                    config.task.option.closing_time_s,
                    config.task.option.hold_time_s,
                    config.task.option.lift_time_s,
                    config.task.option.lift_hold_time_s,
                )
            ),
            600,
        )
        with self.assertRaises(FrozenInstanceError):
            config.run.seed = 7  # type: ignore[misc]

    def test_unknown_duplicate_missing_and_wrong_type_are_rejected(self) -> None:
        unknown = self.default_text.replace("  seed: 42\n", "  seed: 42\n  surprise: true\n", 1)
        with self.assertRaisesRegex(ConfigError, "unknown field"):
            self._load_text(unknown)

        duplicate = self.default_text.replace("  seed: 42\n", "  seed: 42\n  seed: 43\n", 1)
        with self.assertRaises(DuplicateKeyError):
            self._load_text(duplicate)

        missing = self.default_text.replace("  worker_threads: 8\n", "", 1)
        with self.assertRaisesRegex(ConfigError, "missing required field.*worker_threads"):
            self._load_text(missing)

        wrong_type = self.default_text.replace("  num_envs: 128\n", "  num_envs: '128'\n", 1)
        with self.assertRaisesRegex(ConfigError, "expected int"):
            self._load_text(wrong_type)

    def test_cli_override_is_last_and_revalidated(self) -> None:
        overridden = self._load_text(
            self.default_text.replace("  seed: 42\n", "  seed: 11\n", 1),
            overrides=(
                "run.seed=99",
                "platform.num_envs=4",
                "platform.worker_threads=2",
            ),
        )
        self.assertEqual(overridden.run.seed, 99)
        self.assertEqual(overridden.platform.num_envs, 4)
        self.assertEqual(overridden.platform.worker_threads, 2)
        with self.assertRaisesRegex(ConfigError, "cannot exceed"):
            self._load_text(
                self.default_text,
                overrides=("platform.num_envs=2", "platform.worker_threads=3"),
            )
        with self.assertRaisesRegex(ConfigError, "unknown override path"):
            self._load_text(self.default_text, overrides=("run.unknown=1",))
        with self.assertRaisesRegex(ConfigError, "duplicate command-line override"):
            self._load_text(self.default_text, overrides=("run.seed=1", "run.seed=2"))

    def test_nonfinite_unit_range_and_cross_field_errors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigError, "finite"):
            self._load_text(self.default_text, overrides=("physics.physics_dt_s=.nan",))
        with self.assertRaisesRegex(ConfigError, "unit must be m"):
            self._load_text(self.default_text, overrides=("randomization.radius.unit=mm",))
        with self.assertRaisesRegex(ConfigError, "minimum must be below maximum"):
            self._load_text(
                self.default_text,
                overrides=("randomization.radius.minimum=0.020",),
            )
        with self.assertRaisesRegex(ConfigError, "policy_dt_s must equal"):
            self._load_text(self.default_text, overrides=("physics.policy_dt_s=0.021",))

    def test_canonical_json_has_fixed_golden_and_normalizes_negative_zero(self) -> None:
        value = {"z": -0.0, "a": [1.0, 0.0], "unicode": "米"}
        golden = '{"a":[1.0,0.0],"unicode":"米","z":0.0}'
        self.assertEqual(canonical_json(value), golden)
        self.assertEqual(
            hashlib.sha256(golden.encode("utf-8")).hexdigest(),
            "ff4260231863c9619e6f549f550606ead5506ea31be34f271dad2b4533345817",
        )
        self.assertEqual(canonical_json({"b": 1, "a": 2}), canonical_json({"a": 2, "b": 1}))

    def test_asset_hashes_are_optional_historical_metadata(self) -> None:
        self.assertEqual(self.config.asset.runtime_mjcf_sha256, "")
        changed = self._load_text(
            self.default_text,
            overrides=("asset.asset_version=local_model",),
        )
        self.assertEqual(changed.asset.asset_version, "local_model")
        legacy = self.default_text.replace("asset:\n", "asset:\n  runtime_mjcf_sha256: historical-value\n", 1)
        self.assertEqual(self._load_text(legacy).asset.runtime_mjcf_sha256, "historical-value")

    def test_snapshot_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_config_snapshot(self.config, first)
            write_config_snapshot(self.config, second)
            expected_names = {
                "resolved.yaml",
                "canonical.json",
                "dependencies.txt",
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected_names)
            for name in expected_names:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
