from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlData:
    indices: list[int] | int
    ctrl_states: list[int] | int
