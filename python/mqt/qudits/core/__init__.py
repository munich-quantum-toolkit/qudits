"""Core structure used in the package."""

from __future__ import annotations

from .dfs_tree import NAryTree, Node
from .level_graph import LevelGraph

__all__ = [
    "LevelGraph",
    "NAryTree",
    "Node",
]
