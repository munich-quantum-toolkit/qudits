from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class GellMann(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: list,
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
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b, self.type_m = parameters

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        d = self._dimensions
        matrix = np.zeros((d, d), dtype=complex)

        if self.type_m == "s":
            matrix[self.lev_a, self.lev_b] = 1
            matrix[self.lev_b, self.lev_a] = 1
        elif self.type_m == "a":
            matrix[self.lev_a, self.lev_b] -= 1j
            matrix[self.lev_b, self.lev_a] += 1j
        else:
            # lev_a is l in this case
            E = np.zeros((d, d), dtype=complex)

            for j_ind in range(0, self.lev_b):
                E[j_ind, j_ind] += 1

            E[self.lev_b, self.lev_b] -= self.lev_b

            coeff = np.sqrt(2 / (self.lev_b * (self.lev_b + 1)))

            E = coeff * E

            matrix = E

        return matrix

    def validate_parameter(self, parameter):
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], int)
        assert (
            0 <= parameter[0] < parameter[1]
        ), f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
        assert isinstance(parameter[2], str), "type parameter should be a string"

        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
