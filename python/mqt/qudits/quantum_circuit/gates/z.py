# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class Z(Gate):
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
            qasm_tag="z",
        )

    def __array__(self) -> NDArray[np.complex128]:  # noqa: PLW3201
        matrix = np.zeros((self.dimensions, self.dimensions), dtype=np.complex128)
        for i in range(self.dimensions):
            omega = np.mod(2 * i / self.dimensions, 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega
            array = np.zeros(self.dimensions, dtype=np.complex128)
            array[i] = 1
            result = omega * np.outer(array, array)
            matrix += result

        return matrix

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer in Z gate"
        return self._dimensions
