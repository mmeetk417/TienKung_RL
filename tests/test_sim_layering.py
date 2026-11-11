"""Static architecture guards for the simulation layer."""

from __future__ import annotations

import ast
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1] / "tienkung_rl" / "sim"
FORBIDDEN_IMPORT_PARTS = {
    "admission",
    "curriculum",
    "evaluate",
    "evaluation",
    "option",
    "reward",
    "termination",
    "telemetry",
    "viewer",
    "visualization",
}


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return tuple(modules)


def test_sim_layer_has_no_task_reward_curriculum_evaluation_or_viewer_imports() -> None:
    for path in SIM_ROOT.glob("*.py"):
        for module in _imports(path):
            parts = set(module.split("."))
            assert not parts.intersection(FORBIDDEN_IMPORT_PARTS), (path, module)


def test_headless_backend_does_not_construct_rendering_objects() -> None:
    backend_path = SIM_ROOT / "mujoco_backend.py"
    source = backend_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(backend_path))
    imported = _imports(backend_path)
    assert "mujoco.viewer" not in imported
    assert "Renderer" not in source
    assert "GLContext" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"launch", "launch_passive"}
