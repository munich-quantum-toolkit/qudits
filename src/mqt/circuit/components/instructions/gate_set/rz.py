from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.circuit.components.instructions.gate import Gate

if TYPE_CHECKING:
    import numpy as np

    from mqt.circuit.components.instructions.gate_extensions.controls import ControlData


class Rz(Gate):
    def __qasm__(self) -> str:
        pass

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        pass

    def validate_parameter(self, parameter):
        pass

    def __init__(self, name: str, num_qudits: int, params: list, controls: ControlData | None = None):
        super().__init__(name, num_qudits, params)

    def __str__(self):
        pass
