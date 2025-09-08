# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class H(Gate):
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
            qasm_tag="h",
        )

    def __array__(self) -> NDArray[np.complex128]:  # noqa: PLW3201
        matrix = np.zeros((self.dimensions, self.dimensions), dtype=np.complex128)
        for e0, e1 in itertools.product(range(self.dimensions), repeat=2):
            omega = np.mod(2 / self.dimensions * (e0 * e1), 2)
            omega = omega * np.pi * 1j
            omega = np.e**omega
            array0 = np.zeros(self.dimensions, dtype=np.complex128)
            array1 = np.zeros(self.dimensions, dtype=np.complex128)
            array0[e0] = 1
            array1[e1] = 1
            matrix += omega * np.outer(array0, array1)
        return np.asarray(matrix * (1 / np.sqrt(self.dimensions)), dtype=np.complex128)

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions
