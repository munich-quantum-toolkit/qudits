from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..gate import ControlData, Gate, GateTypes

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit


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

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension = self._dimensions
        levels_list = list(range(dimension))

        matrix = np.outer([0 for x in range(dimension)], [0 for x in range(dimension)])

        for lev in levels_list:
            omega = np.mod(2 / dimension * lev * (lev + 1) / 2, 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega

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
        return True

    def __str__(self) -> str:
        # TODO
        pass
