from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class X(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        dimensions: int,
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="x",
        )

    def __array__(self) -> NDArray:  # noqa: PLW3201
        matrix = np.zeros((self.dimensions, self.dimensions), dtype="complex")
        for i in range(self.dimensions):
            i_plus_1 = np.mod(i + 1, self.dimensions)
            array1 = np.zeros(self.dimensions, dtype="complex")
            array2 = np.zeros(self.dimensions, dtype="complex")
            array1[i_plus_1] = 1
            array2[i] = 1
            matrix += np.outer(array1, array2)
        return matrix

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions
