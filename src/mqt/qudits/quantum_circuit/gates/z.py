from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class Z(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        self.qasm_tag = "z"

    def __array__(self) -> np.ndarray:
        matrix = np.zeros((self._dimensions, self._dimensions), dtype="complex")
        for i in range(self._dimensions):
            omega = np.mod(2 * i / self._dimensions, 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega
            array = np.zeros(self._dimensions, dtype="complex")
            array[i] = 1
            result = omega * np.outer(array, array)
            matrix += result

        return matrix

    def validate_parameter(self, parameter=None) -> bool:
        return True

    def __str__(self) -> str:
        # TODO
        pass
