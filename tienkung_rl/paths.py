"""Shared project paths, independent of simulation and asset-generation code."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ASSET_ROOT = PACKAGE_ROOT / "assets" / "tienkung2_pro"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reach_then_close.yaml"
