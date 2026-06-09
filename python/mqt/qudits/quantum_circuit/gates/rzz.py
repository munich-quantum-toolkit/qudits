# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...compiler.compilation_minitools.local_compilation_minitools import regulate_theta
from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class Rzz(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit | None,
        name: str,
        target_qudits: list[int],
        parameters: list[int | float],
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ) -> None:
        if isinstance(dimensions, int):
            dimensions = [dimensions, dimensions]

        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="rzz",
        )

        # Nimm einfach den letzten Wert der Liste also phi
        if isinstance(parameters, list):
            self.phi = regulate_theta(float(parameters[-1]))
        else:
            self.phi = regulate_theta(float(parameters))

        self._params = parameters

    def __array__(self) -> NDArray[np.complex128]:
        dim_ctrl, dim_target = self.dimensions
        dim_total = dim_ctrl * dim_target

        result = np.zeros((dim_total, dim_total), dtype=np.complex128)

        for i in range(dim_ctrl):
            # Z-Wert für das erste Qudit: Level 0 ist 1, Rest ist -1
            z_ctrl = 1 if i == 0 else -1

            for j in range(dim_target):
                # Z-Wert für das zweite Qudit: Level 0 ist 1, Rest ist -1
                z_target = 1 if j == 0 else -1

                # Phase berechnen: exp(-i * phi * z1 * z2 / 2)
                phase = np.exp(-1j * self.phi * z_ctrl * z_target / 2)

                index = i * dim_target + j
                result[index, index] = phase

        return result

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False
        if isinstance(parameter, list):
            return len(parameter) > 0
        return isinstance(parameter, (int, float))

    @property
    def dimensions(self) -> list[int]:
        return cast("list[int]", self._dimensions)
