from __future__ import annotations

import enum


class GateTypes(enum.Enum):
    """Enumeration for gate types."""

    SINGLE = "Single Qudit Gate"
    TWO = "Two Qudit Gate"
    MULTI = "Multi Qudit Gate"


CORE_GATE_TYPES: tuple[GateTypes, GateTypes, GateTypes] = (GateTypes.SINGLE, GateTypes.TWO, GateTypes.MULTI)
