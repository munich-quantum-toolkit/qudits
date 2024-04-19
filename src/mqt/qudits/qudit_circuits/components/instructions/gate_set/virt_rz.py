from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mqt.qudits.compiler.compilation_minitools.local_compilation_minitools import phi_cost, regulate_theta
from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class VirtRz(Gate):
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
            self.lev_a, self.phi = parameters
            self.phi = regulate_theta(self.phi)
            self._params = parameters
        self.qasm_tag = "virtrz"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension = self._dimensions
        theta = self.phi
        matrix = np.identity(dimension, dtype="complex")
        matrix[self.lev_a, self.lev_a] = np.exp(-1j * theta) * matrix[self.lev_a, self.lev_a]

        return matrix

    def validate_parameter(self, parameter):
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], float)
        assert 0 <= parameter[0] <= self._dimensions
        return True

    def __str__(self):
        # TODO
        pass

    @property
    def cost(self):
        return phi_cost(self.phi)
