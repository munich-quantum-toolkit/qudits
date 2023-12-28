from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.gellman import GellMann

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class MS(Gate):
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
            self.theta = parameters[0]

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        theta = self.theta
        dimension_0 = self._dimensions[0]
        dimension_1 = self._dimensions[1]

        return expm(
            -1j
            * theta
            * (
                (
                    np.outer(np.identity(dimension_0, dtype="complex"), GellMann(0, 1, "s", dimension_1).to_matrix())
                    + np.outer(GellMann(0, 1, "s", dimension_0).to_matrix(), np.identity(dimension_1, dtype="complex"))
                )
                @ (
                    np.outer(np.identity(dimension_0, dtype="complex"), GellMann(0, 1, "s", dimension_1).to_matrix())
                    + np.outer(GellMann(0, 1, "s", dimension_0).to_matrix(), np.identity(dimension_1, dtype="complex"))
                )
            )
            / 4
        )

    def validate_parameter(self, parameter):
        assert 0 <= parameter[0] <= 2 * np.pi, f"Angle should be in the range [0, 2*pi]: {parameter[0]}"
        return True

    def __qasm__(self) -> str:
        # TODO
        pass

    def __str__(self):
        # TODO
        pass
