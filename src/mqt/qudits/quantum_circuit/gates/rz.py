from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...compiler.compilation_minitools.local_compilation_minitools import regulate_theta
from ..gate import ControlData, Gate, GateTypes
from .r import R

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit


class Rz(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: list | None,
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
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b, self.phi = parameters
            self.phi = regulate_theta(self.phi)
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self._params = parameters
        self.qasm_tag = "rz"

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

    def levels_setter(self, la, lb):
        if la < lb:
            return la, lb
        return lb, la

    def validate_parameter(self, parameter) -> bool:
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], float)

        assert parameter[0] >= 0
        assert parameter[0] <= self._dimensions
        assert parameter[1] >= 0
        assert parameter[1] <= self._dimensions
        assert parameter[0] != parameter[1]
        # Useful to remember direction of the rotation
        self.original_lev_a = parameter[0]
        self.original_lev_b = parameter[1]

        return True

    def __str__(self) -> str:
        # TODO
        pass
