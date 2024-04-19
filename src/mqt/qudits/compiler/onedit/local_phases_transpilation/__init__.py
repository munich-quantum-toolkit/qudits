from __future__ import annotations

from .propagate_virtrz import ZPropagationPass
from .remove_phase_rotations import ZRemovalPass

__all__ = [
    "ZPropagationPass",
    "ZRemovalPass",
]
