from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class Perm(Gate):
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
            gate_type=GateTypes.MULTI,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        if self.validate_parameter(parameters):
            self.perm_data = parameters

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        return np.eye(reduce(lambda x, y: x * y, self._dimensions))[:, self.perm_data]

    def validate_parameter(self, parameter):
        assert isinstance(parameter, list), "Input is not a list"
        assert all(
            0 <= num < len(parameter) for num in parameter
        ), "Numbers are not within the range of the list length"
        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
