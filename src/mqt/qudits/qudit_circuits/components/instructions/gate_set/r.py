from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.compiler.onedit.local_rotation_tools.local_compilation_minitools import regulate_theta, theta_cost
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
                gate_type=GateTypes.SINGLE,
                target_qudits=target_qudits,
                dimensions=dimensions,
                control_set=controls,
        )
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b, self.theta, self.phi = parameters
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self.theta = regulate_theta(self.theta)
            self._params = parameters
        self.qasm_tag = "rxy"

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

    def levels_setter(self, la, lb):
        if la < lb:
            return la, lb
        else:
            return lb, la

    def validate_parameter(self, parameter):
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], float)
        assert isinstance(parameter[3], float)
        assert parameter[0] >= 0 and parameter[0] <= self._dimensions
        assert parameter[1] >= 0 and parameter[1] <= self._dimensions
        assert parameter[0] != parameter[1]
        # Useful to remember direction of the rotation
        self.original_lev_a = parameter[0]
        self.original_lev_b = parameter[1]

        return True

    def __str__(self):
        # TODO
        pass

    @property
    def cost(self):
        return theta_cost(self.theta)
