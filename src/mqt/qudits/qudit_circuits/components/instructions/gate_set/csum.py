from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.x import X
from mqt.qudits.qudit_circuits.components.instructions.mini_tools.matrix_factory_tools import from_dirac_to_basis

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class CSum(Gate):
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
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        self.qasm_tag = "csum"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        ctrl_size = self.parent_circuit.dimensions[self._target_qudits[0]]
        target_size = self.parent_circuit.dimensions[self._target_qudits[1]]

        matrix = np.zeros(ctrl_size * target_size, dtype="complex")

        x_gate = X(self.parent_circuit, None, self._target_qudits[1], target_size, None)

        for i in range(ctrl_size):
            temp = np.zeros(ctrl_size, dtype="complex")
            mapmat = temp + np.outer(
                np.array(from_dirac_to_basis([i], ctrl_size)), np.array(from_dirac_to_basis([i], ctrl_size))
            )

            Xmat = x_gate.to_matrix(identities=0)
            Xmat_i = np.linalg.matrix_power(Xmat, i)

            if self._target_qudits[0] < self._target_qudits[1]:
                matrix = matrix + (np.kron(mapmat, Xmat_i))
            else:
                matrix = matrix + (np.kron(Xmat_i, mapmat))

        return matrix

    def validate_parameter(self, parameter=None):
        return True

    def __str__(self):
        # TODO
        pass
