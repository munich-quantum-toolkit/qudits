from __future__ import annotations

import enum
from abc import ABC
from typing import TYPE_CHECKING

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate

if TYPE_CHECKING:
    import numpy as np

    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class CustomUnitary(Gate, ABC):
    def __init__(
            self,
            circuit: QuantumCircuit,
            name: str,
            gate_type: enum,
            target_qudits: list[int] | int,
            dimensions: list[int] | int,
            parameters: list | None = None,
            control_set=None,
            label: str | None = None
    ):
        pass

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        pass

    def validate_parameter(self, parameter):
        pass

    def __qasm__(self) -> str:
        raise NotImplemented

    def __str__(self):
        pass
