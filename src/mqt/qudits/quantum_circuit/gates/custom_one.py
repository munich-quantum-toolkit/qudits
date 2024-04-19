from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gate import ControlData, Gate, GateTypes

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit


class CustomOne(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: np.ndarray,
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
            self.__array_storage = parameters
            self._params = "--"
        self.qasm_tag = "cuone"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        return self.__array_storage

    def validate_parameter(self, parameter=None):
        return isinstance(parameter, np.ndarray)

    def __str__(self) -> str:
        # TODO
        pass
