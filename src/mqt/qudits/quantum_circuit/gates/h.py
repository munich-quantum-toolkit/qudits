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
        basis_states_projectors = [list(range(self._dimensions)), list(range(self._dimensions))]

        matrix_array = np.outer([0 for x in range(self._dimensions)], [0 for x in range(self._dimensions)])
        matrix = None

        for e0, e1 in itertools.product(*basis_states_projectors):
            omega = np.mod(2 / self._dimensions * (e0 * e1), 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega

            l1 = [0 for x in range(self._dimensions)]
            l2 = [0 for x in range(self._dimensions)]
            l1[e0] = 1
            l2[e1] = 1

            array1 = np.array(l1, dtype="complex")
            array2 = np.array(l2, dtype="complex")

            result = omega * np.outer(array1, array2)

            matrix_array = matrix_array + result
            matrix = (1 / np.sqrt(self._dimensions)) * matrix_array

        return matrix

    def validate_parameter(self, parameter=None) -> bool:
        return True

    def __str__(self) -> str:
        # TODO
        pass
