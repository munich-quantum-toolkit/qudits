from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .x import X

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class CSum(Gate):
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
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        self.qasm_tag = "csum"

    def __array__(self) -> np.ndarray:
        ctrl_size = self.parent_circuit.dimensions[self._target_qudits[0]]
        target_size = self.parent_circuit.dimensions[self._target_qudits[1]]

        x_gate = X(self.parent_circuit, "x", self._target_qudits[1], target_size, None)
        x_mat = x_gate.to_matrix(identities=0)
        matrix = np.zeros((ctrl_size * target_size, ctrl_size * target_size), dtype="complex")
        for i in range(ctrl_size):
            basis = np.array(from_dirac_to_basis([i], ctrl_size), dtype="complex")
            mapmat = np.outer(basis, basis)
            x_mat_i = np.linalg.matrix_power(x_mat, i)
            if self._target_qudits[0] < self._target_qudits[1]:
                matrix += np.kron(mapmat, x_mat_i)
            else:
                matrix += np.kron(x_mat_i, mapmat)
        return matrix

    def validate_parameter(self, parameter=None) -> bool:
        return True

    def __str__(self) -> str:
        # TODO
        pass
