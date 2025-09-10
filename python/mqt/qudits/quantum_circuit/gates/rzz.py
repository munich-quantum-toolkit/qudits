from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import expm  # type: ignore[import-not-found]

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
            qasm_tag="rzz",
            params=parameters,
            theta=parameters[0],
        )
        if self.validate_parameter(parameters):
            self.theta = parameters[0]
            self._params = parameters

    @staticmethod
    def zab(dimension: int, ps: list[int | str]) -> NDArray:
        m = np.zeros((dimension, dimension))
        a, b, _typ = ps
        m[int(a), int(a)] = 1
        m[int(b), int(b)] = -1
        return m

    def __array__(self) -> NDArray:  # noqa: PLW3201
        theta = self.theta
        dimension_0 = self.dimensions[0]
        dimension_1 = self.dimensions[1]
        ps: list[int | str] = [0, 1, "a"]

        m0 = Rzz.zab(dimension_0, ps)
        m1 = Rzz.zab(dimension_1, ps)
        gate_part_1 = np.kron(m0, np.identity(dimension_1, dtype="complex"))
        gate_part_2 = np.kron(np.identity(dimension_0, dtype="complex"), m1)
        return expm(-1j * theta / 2 * gate_part_1 @ gate_part_2)

    @staticmethod
    def validate_parameter(parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            return True
        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast("list[int]", self._dimensions)
