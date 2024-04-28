from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class H(Gate):
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
        self.qasm_tag = "h"

    def __array__(self) -> np.ndarray:
        matrix = np.zeros((self._dimensions, self._dimensions), dtype="complex")
        for e0, e1 in itertools.product(range(self._dimensions), repeat=2):
            omega = np.mod(2 / self._dimensions * (e0 * e1), 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega
            array0 = np.zeros(self._dimensions, dtype="complex")
            array1 = np.zeros(self._dimensions, dtype="complex")
            array0[e0] = 1
            array1[e1] = 1
            matrix += omega * np.outer(array0, array1)
        return matrix * (1 / np.sqrt(self._dimensions))

    def validate_parameter(self, parameter=None) -> bool:
        return True

    def __str__(self) -> str:
        # TODO
        pass
