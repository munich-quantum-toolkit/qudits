from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...compiler.compilation_minitools.local_compilation_minitools import phi_cost, regulate_theta
from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .r import R

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class Rz(Gate):
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
            qasm_tag="rz",
        )
        if self.validate_parameter(parameters):
            self.original_lev_a: int = cast("int", parameters[0])
            self.original_lev_b: int = cast("int", parameters[1])
            self.phi: float = cast("float", parameters[2])
            self.phi = regulate_theta(self.phi)
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self._params = parameters

    def __array__(self) -> NDArray[np.complex128, np.complex128]:  # noqa: PLW3201
        dimension = self.dimensions
        phi = self.phi
        qudit_targeted: int = cast("int", self.target_qudits)

        pi_there = R(
            self.parent_circuit, "R", qudit_targeted, [self.lev_a, self.lev_b, -np.pi / 2, 0.0], dimension
        ).to_matrix()
        rotate = R(
            self.parent_circuit, "R", qudit_targeted, [self.lev_a, self.lev_b, phi, np.pi / 2], dimension
        ).to_matrix()
        pi_back = R(
            self.parent_circuit, "R", qudit_targeted, [self.lev_a, self.lev_b, np.pi / 2, 0.0], dimension
        ).to_matrix()

        return np.matmul(np.matmul(pi_back, rotate), pi_there)  # pi_back @ rotate @ pi_there

    def _dagger_properties(self) -> None:
        self.phi *= -1
        self.update_params([self.lev_a, self.lev_b, self.phi])

    @staticmethod
    def levels_setter(la: int, lb: int) -> tuple[int, int]:
        if la < lb:
            return la, lb
        return lb, la

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            assert isinstance(parameter[0], int)
            assert isinstance(parameter[1], int)
            assert isinstance(parameter[2], float)

            assert parameter[0] >= 0
            assert parameter[0] < self.dimensions
            assert parameter[1] >= 0
            assert parameter[1] < self.dimensions
            assert parameter[0] != parameter[1]
            # Useful to remember direction of the rotation
            self.original_lev_a = parameter[0]
            self.original_lev_b = parameter[1]

            return True
        if isinstance(parameter, np.ndarray):
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
