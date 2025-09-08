from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlData:
    indices: list[int]
    ctrl_states: list[int]
