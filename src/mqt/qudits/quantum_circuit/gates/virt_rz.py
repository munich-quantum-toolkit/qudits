from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...compiler.compilation_minitools.local_compilation_minitools import phi_cost, regulate_theta
from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class VirtRz(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: list[int | float],
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
            qasm_tag="virtrz",
        )
        if self.validate_parameter(parameters):
            self.lev_a: int = cast("int", parameters[0])
            self.phi: float = cast("float", parameters[1])
            self.phi = regulate_theta(self.phi)
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        dimension = self.dimensions
        theta = self.phi
        matrix = np.identity(dimension, dtype="complex")
        matrix[self.lev_a, self.lev_a] = np.exp(-1j * theta) * matrix[self.lev_a, self.lev_a]

        return matrix

    def _dagger_properties(self) -> None:
        self.phi *= -1
        self.update_params([self.lev_a, self.phi])

    def validate_parameter(self, param: Parameter) -> bool:
        if param is None:
            return False

        if isinstance(param, list):
            if len(param) != 2:
                return False
            if not (isinstance(param[0], int) and isinstance(param[1], float)):
                return False
            return 0 <= param[0] < self.dimensions

        if isinstance(param, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def cost(self) -> float:
        return phi_cost(self.phi)

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions
