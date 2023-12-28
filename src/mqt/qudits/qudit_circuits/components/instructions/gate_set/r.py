from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.gellman import GellMann

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class R(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: list | None,
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
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b, self.theta, self.phi = parameters

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension = self._dimensions
        theta = self.theta
        phi = self.phi
        matrix = np.identity(dimension, dtype="complex")

        matrix[self.lev_a, self.lev_a] = np.cos(theta / 2) * matrix[self.lev_a, self.lev_a]
        matrix[self.lev_b, self.lev_b] = np.cos(theta / 2) * matrix[self.lev_b, self.lev_b]

        cosine_matrix = matrix

        return cosine_matrix - 1j * np.sin(theta / 2) * (
            np.sin(phi)
            * GellMann(
                self.parent_circuit,
                "Gellman_a",
                self._target_qudits,
                [self.lev_a, self.lev_b, "a"],
                self._dimensions,
                None,
            ).to_matrix()
            + np.cos(phi)
            * GellMann(
                self.parent_circuit,
                "Gellman_s",
                self._target_qudits,
                [self.lev_a, self.lev_b, "s"],
                self._dimensions,
                None,
            ).to_matrix()
        )

    def validate_parameter(self, parameter):
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], int)
        assert (
            0 <= parameter[0] < parameter[1]
        ), f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
        assert 0 <= parameter[2] <= 2 * np.pi, f"Angle theta should be in the range [0, 2*pi]: {parameter[2]}"
        assert 0 <= parameter[3] <= 2 * np.pi, f"Angle phi should be in the range [0, 2*pi]: {parameter[3]}"

        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
