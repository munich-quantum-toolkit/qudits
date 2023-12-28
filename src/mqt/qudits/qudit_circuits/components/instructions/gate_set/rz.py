from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.r import R

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class Rz(Gate):
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
            self.lev_a, self.lev_b, self.phi = parameters

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension = self._dimensions
        phi = self.phi

        pi_there = R(
            self.parent_circuit, "R", self._target_qudits, [self.lev_a, self.lev_b, -np.pi / 2, 0.0], dimension
        ).to_matrix()
        rotate = R(
            self.parent_circuit, "R", self._target_qudits, [self.lev_a, self.lev_b, phi, np.pi / 2], dimension
        ).to_matrix()
        pi_back = R(
            self.parent_circuit, "R", self._target_qudits, [self.lev_a, self.lev_b, np.pi / 2, 0.0], dimension
        ).to_matrix()

        return pi_back @ rotate @ pi_there

    def validate_parameter(self, parameter):
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], int)
        assert (
            0 <= parameter[0] < parameter[1]
        ), f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
        assert 0 <= parameter[2] <= 2 * np.pi, f"Angle phi should be in the range [0, 2*pi]: {parameter[2]}"

        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
