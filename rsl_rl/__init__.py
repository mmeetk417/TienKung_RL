"""Source-tree import bridge for the vendored :mod:`rsl_rl` package.

An installed wheel maps ``rsl_rl/rsl_rl`` directly to the top-level package.
When commands run from the repository root, Python otherwise treats the outer
directory as a namespace package and hides that editable-install mapping.
Extending this package path keeps source-tree and installed imports identical.
"""

from pathlib import Path

_VENDORED_PACKAGE = Path(__file__).resolve().parent / "rsl_rl"
__path__.append(str(_VENDORED_PACKAGE))
