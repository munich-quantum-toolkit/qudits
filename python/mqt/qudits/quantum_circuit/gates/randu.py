# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.stats import unitary_group

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class RandU(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        dimensions: list[int],
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.MULTI,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="rdu",
        )

    def __array__(self) -> NDArray[np.complex128]:  # noqa: PLW3201
        dim = reduce(operator.mul, self.dimensions)
        return np.asarray(unitary_group.rvs(dim), dtype=np.complex128)

    @property
    def dimensions(self) -> list[int]:
        return cast("list[int]", self._dimensions)
