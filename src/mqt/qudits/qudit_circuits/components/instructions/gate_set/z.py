from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class Z(Gate):
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
        self.qasm_tag = "z"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension = self._dimensions
        levels_list = list(range(dimension))

        matrix = np.outer([0 for x in range(dimension)], [0 for x in range(dimension)])

        for level in levels_list:
            omega = np.mod(2 * level / dimension, 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[level] = 1
            l2[level] = 1

            array1 = np.array(l1, dtype="complex")
            array2 = np.array(l2, dtype="complex")
            proj = np.outer(array1, array2)
            result = omega * proj

            matrix = matrix + result

        return matrix

    def validate_parameter(self, parameter=None):
        return True

    def __str__(self):
        # TODO
        pass
