"""Native-viewer overlays and simulator-independent debug geometry."""

from .debug_geometry import (
    DebugGeometryPlan,
    DebugPrimitive,
    apply_native_debug_geometry,
    build_debug_geometry,
)
from .overlays import OverlaySection, apply_native_overlays, build_overlay_sections

__all__ = [
    "DebugGeometryPlan",
    "DebugPrimitive",
    "OverlaySection",
    "apply_native_debug_geometry",
    "apply_native_overlays",
    "build_debug_geometry",
    "build_overlay_sections",
]
