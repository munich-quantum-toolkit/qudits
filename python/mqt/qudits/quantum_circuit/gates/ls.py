# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import expm

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class LS(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        parameters: list[float],
        dimensions: list[int],
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="ls",
            params=parameters,
            theta=parameters[0],
        )
        if self.validate_parameter(parameters):
            self.theta = parameters[0]
            self._params = parameters

    def __array__(self) -> NDArray[np.complex128]:  # noqa: PLW3201
        dimension_0 = self.dimensions[0]
        dimension_1 = self.dimensions[1]

        exp_matrix = np.zeros((dimension_0 * dimension_1, dimension_0 * dimension_1), dtype=np.complex128)
        d_min = min(dimension_0, dimension_1)
        for i in range(d_min):
            exp_matrix += np.outer(
                np.array(from_dirac_to_basis([i, i], self.dimensions)),
                np.array(from_dirac_to_basis([i, i], self.dimensions)),
            )

        return np.asarray(expm(-1j * self.theta * exp_matrix), dtype=np.complex128)

    @staticmethod
    def validate_parameter(parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            assert 0 <= cast("float", parameter[0]) <= 2 * np.pi, (
                f"Angle should be in the range [0, 2*pi]: {parameter[0]}"
            )
            return True

        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast("list[int]", self._dimensions)
