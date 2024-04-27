from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class S(Gate):
    def __init__(
            self,
            circuit: QuantumCircuit,
            name: str,
            target_qudits: list[int] | int,
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
        self.qasm_tag = "s"

    def __array__(self) -> np.ndarray:
        dimension = self._dimensions
        levels_list = list(range(dimension))
        if dimension == 2:
            return np.array([[1, 0], [0, 1j]])
        matrix = np.outer([0 for x in range(dimension)], [0 for x in range(dimension)])

        for lev in levels_list:
            omega = np.e ** (2 * np.pi * 1j / dimension)
            omega **= np.mod(lev * (lev + 1) / 2, dimension)

            l1 = [0 for _ in range(dimension)]
            l2 = [0 for _ in range(dimension)]
            l1[lev] = 1
            l2[lev] = 1

            array1 = np.array(l1, dtype="complex")
            array2 = np.array(l2, dtype="complex")

            result = omega * np.outer(array1, array2)

            matrix = matrix + result

        return matrix

    def validate_parameter(self, parameter=None) -> bool:
        if np.mod(self._dimensions, 2) == 0 and self._dimensions > 2:
            msg = "S can be applied to prime dimensional qudits"
            raise Exception(msg)
        return True

    def __str__(self) -> str:
        # TODO
        pass
