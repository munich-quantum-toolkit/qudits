from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .x import X

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class CSum(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        dimensions: list[int],
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="csum",
        )

    def __array__(self) -> NDArray[np.complex128, np.complex128]:  # noqa: PLW3201
        qudits_targeted = cast("list[int]", self.target_qudits)
        qudit_targeted_0: int = qudits_targeted[0]
        qudit_targeted_1: int = qudits_targeted[1]
        ctrl_size = self.parent_circuit.dimensions[qudit_targeted_0]
        target_size = self.parent_circuit.dimensions[qudit_targeted_1]

        x_gate = X(self.parent_circuit, "x", qudit_targeted_1, target_size, None)
        x_mat = x_gate.to_matrix(identities=0)
        matrix = np.zeros((ctrl_size * target_size, ctrl_size * target_size), dtype="complex")
        for i in range(ctrl_size):
            basis = np.array(from_dirac_to_basis([i], ctrl_size), dtype="complex")
            mapmat = np.outer(basis, basis)
            x_mat_i = np.linalg.matrix_power(x_mat, i)
            if qudit_targeted_0 < qudit_targeted_1:
                matrix += np.kron(mapmat, x_mat_i)
            else:
                matrix += np.kron(x_mat_i, mapmat)

        if self.dagger:
            return matrix.conj().T

        return matrix

    def to_qasm(self) -> str:
        string_description = self.__qasm__()
        if self.dagger:
            return "inv @ " + string_description
        return string_description
