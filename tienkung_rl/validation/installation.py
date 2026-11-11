"""Read-only imports used by the one-command installation self-check."""

from __future__ import annotations

import importlib
import json
import platform


def main() -> int:
    modules = {
        name: importlib.import_module(name) for name in ("mujoco", "mujoco.viewer", "torch", "rsl_rl", "tienkung_rl")
    }
    versions = {name: getattr(module, "__version__", "imported") for name, module in modules.items()}
    print(
        json.dumps(
            {
                "status": "PASS",
                "python": platform.python_version(),
                "imports": versions,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
