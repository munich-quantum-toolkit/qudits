from __future__ import annotations

import enum


class GateTypes(enum.Enum):
    """Enumeration for job status."""

    SINGLE = "Single Qudit Gate"
    TWO = "Two Qudit Gate"
    MULTI = "Multi Qudit Gate"


CORE_GATE_TYPES = (GateTypes.SINGLE, GateTypes.TWO, GateTypes.MULTI)
