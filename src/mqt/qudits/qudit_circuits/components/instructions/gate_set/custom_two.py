from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.abstract_custom import CustomUnitary

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class CustomTwo(CustomUnitary):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: np.ndarray,
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ):
        super().__init__(
            circuit=circuit,
            name=name,
            parameters=parameters,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        if self.validate_parameter(parameters):
            self.__array_storage = parameters

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        return self.__array_storage

    def validate_parameter(self, parameter=None):
        return isinstance(parameter, np.ndarray)

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
