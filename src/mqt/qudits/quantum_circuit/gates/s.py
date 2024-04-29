from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class S(Gate):
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
        self.qasm_tag = "s"

    def __array__(self) -> np.ndarray:
        if self._dimensions == 2:
            return np.array([[1, 0], [0, 1j]])
        matrix = np.zeros((self._dimensions, self._dimensions), dtype="complex")
        for i in range(self._dimensions):
            omega = np.e ** (2 * np.pi * 1j / self._dimensions)
            omega **= np.mod(i * (i + 1) / 2, self._dimensions)
            array = np.zeros(self._dimensions, dtype="complex")
            array[i] = 1
            result = omega * np.outer(array, array)
            matrix += result
        return matrix

    def validate_parameter(self, parameter=None) -> bool:
        if np.mod(self._dimensions, 2) == 0 and self._dimensions > 2:
            msg = "S can be applied to prime dimensional qudits"
            raise Exception(msg)
        return True

    def __str__(self) -> str:
        # TODO
        pass
