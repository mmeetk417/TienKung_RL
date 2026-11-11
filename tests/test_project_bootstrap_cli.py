"""Package, CLI, and simulator isolation tests."""

from __future__ import annotations

import ast
import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from tienkung_rl import _cli

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_COMMANDS = ("train", "evaluate", "viewer", "validate_asset", "smoke")


class PackageAndEnvironmentTests(unittest.TestCase):
    def test_package_discovery_has_only_active_roots(self) -> None:
        packages = [
            ".".join(path.parent.relative_to(ROOT).parts) for path in (ROOT / "tienkung_rl").rglob("__init__.py")
        ]
        packages += [
            ".".join(path.parent.relative_to(ROOT / "rsl_rl").parts)
            for path in (ROOT / "rsl_rl" / "rsl_rl").rglob("__init__.py")
        ]
        self.assertEqual(
            {package.partition(".")[0] for package in packages},
            {"tienkung_rl", "rsl_rl"},
        )
        setup_source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('include=("tienkung_rl", "tienkung_rl.*")', setup_source)
        self.assertIn('include=("rsl_rl", "rsl_rl.*")', setup_source)
        self.assertIn('package_data={"tienkung_rl": ASSET_PACKAGE_DATA}', setup_source)
        self.assertEqual(
            (ROOT / "MANIFEST.in").read_text(encoding="utf-8").strip(),
            "recursive-include tienkung_rl/assets/tienkung2_pro *",
        )
        self.assertNotIn("find_packages()", setup_source)

    def test_locked_environment_is_cpu_only_and_versioned(self) -> None:
        source = (ROOT / "environment.lock.yml").read_text(encoding="utf-8")
        dependency_lines = [line.strip()[2:] for line in source.splitlines() if line.strip().startswith("- ")]
        dependencies = {
            line.split("==", 1)[0].split("=", 1)[0].casefold() for line in dependency_lines if not line.startswith("--")
        }
        self.assertIn("python=3.11.13", source)
        self.assertIn("mujoco==3.11.0", source)
        self.assertIn("torch==2.13.0+cpu", source)
        self.assertTrue({"mujoco", "torch", "numpy"}.issubset(dependencies))
        forbidden = {
            "isaaclab",
            "isaacsim",
            "omniverse",
            "jax",
            "jaxlib",
            "cuda",
            "warp",
            "mjx",
        }
        self.assertFalse(dependencies & forbidden)
        for line in dependency_lines:
            if line.startswith(("--", "conda-forge")) or line == "pip:":
                continue
            if line.startswith(("python=", "pip=")):
                continue
            self.assertIn("==", line, msg=f"unlocked dependency: {line}")

    def test_install_script_is_single_user_fail_fast_flow(self) -> None:
        path = ROOT / "install.sh"
        source = path.read_text(encoding="utf-8")
        self.assertTrue(os.access(path, os.X_OK))
        self.assertIn("set -Eeuo pipefail", source)
        self.assertIn("conda env create", source)
        self.assertIn("python -m pip install --no-deps --editable", source)
        self.assertIn("python -m tienkung_rl.validation.installation", source)
        self.assertNotIn("sudo", source)

    def test_vendored_rsl_rl_imports_from_repository_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "from rsl_rl.algorithms import PPO; print(PPO.__name__)"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "PPO")


class PublicCliTests(unittest.TestCase):
    def test_every_help_is_lazy_and_successful(self) -> None:
        for command in PUBLIC_COMMANDS:
            code = (
                "import runpy,sys; "
                f"sys.argv=['tienkung_rl.{command}','--help']; "
                f"runpy.run_module('tienkung_rl.{command}',run_name='__main__')"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("--config", result.stdout)
            self.assertIn("--worker-threads", result.stdout)

    def test_common_arguments_use_declared_units_and_names(self) -> None:
        parser = _cli.create_parser("test", "test parser")
        args = parser.parse_args(
            [
                "--config",
                "run.yaml",
                "--seed",
                "7",
                "--output-dir",
                "out",
                "--num-envs",
                "4",
                "--worker-threads",
                "2",
            ]
        )
        self.assertEqual(args.config, Path("run.yaml"))
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.output_dir, Path("out"))
        self.assertEqual(args.num_envs, 4)
        self.assertEqual(args.worker_threads, 2)

    def test_invalid_duplicate_and_conflicting_overrides_fail(self) -> None:
        parser = _cli.create_parser("test", "test parser")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--num-envs", "0"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--seed", "1", "--seed", "2"])
        args = parser.parse_args(["--seed", "1", "--set", "run.seed=2"])
        with self.assertRaisesRegex(_cli.CliUsageError, "both"):
            _cli.config_overrides(args)
        args = parser.parse_args(["--set", "run.seed=1", "--set", "run.seed=2"])
        with self.assertRaisesRegex(_cli.CliUsageError, "twice"):
            _cli.config_overrides(args)

    def test_help_does_not_import_mujoco_or_load_configuration(self) -> None:
        from tienkung_rl import viewer

        mujoco_modules_before = {name for name in sys.modules if name == "mujoco" or name.startswith("mujoco.")}
        with (
            mock.patch.object(_cli, "load_config", side_effect=AssertionError("must remain lazy")),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(SystemExit, "0"),
        ):
            viewer.main(["--help"])
        mujoco_modules_after = {name for name in sys.modules if name == "mujoco" or name.startswith("mujoco.")}
        self.assertEqual(mujoco_modules_after, mujoco_modules_before)

    def test_unknown_and_command_conflicts_return_nonzero(self) -> None:
        commands = (
            ("train", ["--unknown-option"]),
            ("evaluate", ["--checkpoint", "new.pt", "--random-policy"]),
            ("viewer", ["--mode", "policy"]),
        )
        for command, arguments in commands:
            result = subprocess.run(
                [sys.executable, "-m", f"tienkung_rl.{command}", *arguments],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("error", result.stderr.casefold())

    def test_shared_cli_loads_m01_config_with_typed_overrides(self) -> None:
        parser = _cli.create_parser("test", "test parser")
        args = parser.parse_args(
            [
                "--seed",
                "13",
                "--output-dir",
                "outputs/probe",
                "--num-envs",
                "4",
                "--worker-threads",
                "2",
            ]
        )
        config = _cli.load_config(parser, args)
        self.assertEqual(config.run.seed, 13)
        self.assertEqual(config.run.output_dir, "outputs/probe")
        self.assertEqual(config.platform.num_envs, 4)
        self.assertEqual(config.platform.worker_threads, 2)

    def test_every_command_reports_strict_config_errors_before_runtime(self) -> None:
        for command in PUBLIC_COMMANDS:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    f"tienkung_rl.{command}",
                    "--set",
                    "platform.unknown=1",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, msg=result.stderr)
            self.assertIn("configuration error", result.stderr)
            self.assertIn("unknown override path", result.stderr)

    def test_conflicting_cli_override_returns_nonzero(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "tienkung_rl.train",
                "--seed",
                "1",
                "--set",
                "run.seed=2",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("set by both --seed and --set", result.stderr)


class SimulatorIsolationTests(unittest.TestCase):
    def test_active_imports_never_reference_archived_simulator(self) -> None:
        files = list((ROOT / "tienkung_rl").rglob("*.py"))
        files += list((ROOT / "tests").rglob("*.py"))
        files += list((ROOT / "tools").rglob("*.py"))
        forbidden_roots = {"isaaclab", "omni", "legacy"}
        failures: list[str] = []
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.split(".", 1)[0].casefold() in forbidden_roots:
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{name}")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
