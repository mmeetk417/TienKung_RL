#!/usr/bin/env python3
"""Check that active TienKung and local RSL-RL imports resolve in this project."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ROOTS = {"tienkung_rl", "rsl_rl"}


def module_exists(module: str) -> bool:
    parts = module.split(".")
    path = ROOT / "rsl_rl" / "rsl_rl" / Path(*parts[1:]) if parts[0] == "rsl_rl" else ROOT / Path(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[0] == "rsl_rl":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_relative(current: str, is_package: bool, level: int, imported: str | None) -> str:
    package = current.split(".") if is_package else current.split(".")[:-1]
    if level > len(package) + 1:
        return ""
    base = package[: len(package) - level + 1]
    if imported:
        base.extend(imported.split("."))
    return ".".join(base)


class ImportCollector(ast.NodeVisitor):
    def __init__(self, current_module: str, is_package: bool):
        self.current_module = current_module
        self.is_package = is_package
        self.modules: list[tuple[int, str]] = []
        self.type_checking_imports = 0

    def visit_If(self, node: ast.If) -> None:
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            self.type_checking_imports += sum(
                isinstance(child, (ast.Import, ast.ImportFrom)) for child in ast.walk(node)
            )
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.modules.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            imported = resolve_relative(
                self.current_module,
                self.is_package,
                node.level,
                node.module,
            )
        else:
            imported = node.module or ""
        self.modules.append((node.lineno, imported))


def main() -> int:
    files = sorted((ROOT / "tienkung_rl").rglob("*.py"))
    files.extend(sorted((ROOT / "rsl_rl" / "rsl_rl").rglob("*.py")))
    files.extend(sorted((ROOT / "tests").rglob("*.py")))
    files.extend(sorted((ROOT / "tools").rglob("*.py")))
    failures: list[dict[str, object]] = []
    checked_imports = 0
    skipped_type_checking = 0

    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        collector = ImportCollector(module_name(path), path.name == "__init__.py")
        collector.visit(tree)
        skipped_type_checking += collector.type_checking_imports
        for line, imported in collector.modules:
            root = imported.split(".", 1)[0]
            if root not in LOCAL_ROOTS:
                continue
            checked_imports += 1
            if not module_exists(imported):
                failures.append(
                    {
                        "file": str(path.relative_to(ROOT)),
                        "line": line,
                        "module": imported,
                    }
                )

    result = {
        "python_files": len(files),
        "local_imports_checked": checked_imports,
        "type_checking_imports_skipped": skipped_type_checking,
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
