# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import enum


class GateTypes(enum.Enum):
    """Enumeration for gate types."""

    SINGLE = "Single Qudit Gate"
    TWO = "Two Qudit Gate"
    MULTI = "Multi Qudit Gate"


CORE_GATE_TYPES: tuple[GateTypes, GateTypes, GateTypes] = (GateTypes.SINGLE, GateTypes.TWO, GateTypes.MULTI)
