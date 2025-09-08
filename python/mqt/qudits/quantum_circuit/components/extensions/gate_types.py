from __future__ import annotations

import enum


class GateTypes(enum.Enum):
    """Enumeration for gate types."""

    SINGLE: str = "Single Qudit Gate"
    TWO: str = "Two Qudit Gate"
    MULTI: str = "Multi Qudit Gate"


CORE_GATE_TYPES: tuple[GateTypes, GateTypes, GateTypes] = (GateTypes.SINGLE, GateTypes.TWO, GateTypes.MULTI)
