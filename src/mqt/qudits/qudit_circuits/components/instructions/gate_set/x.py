from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class X(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ):
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        basis_states_list = list(range(self._dimensions))

        matrix = np.outer([0 for x in range(self._dimensions)], [0 for x in range(self._dimensions)])

        for i in basis_states_list:
            i_plus_1 = np.mod(i + 1, self._dimensions)

            l1 = [0 for x in range(self._dimensions)]
            l2 = [0 for x in range(self._dimensions)]
            l1[i_plus_1] = 1
            l2[i] = 1

            array1 = np.array(l1, dtype="complex")
            array2 = np.array(l2, dtype="complex")

            result = np.outer(array1, array2)
            matrix = matrix + result

        return matrix

    def validate_parameter(self, parameter=None):
        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
