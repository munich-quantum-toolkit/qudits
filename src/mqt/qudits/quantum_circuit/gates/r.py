from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...compiler.compilation_minitools.local_compilation_minitools import regulate_theta, theta_cost
from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .gellmann import GellMann

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class R(Gate):
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
            qasm_tag="rxy",
        )

        if self.validate_parameter(parameters):
            self.original_lev_a: int = cast(int, parameters[0])
            self.original_lev_b: int = cast(int, parameters[1])
            self.theta: float = cast(float, parameters[2])
            self.phi: float = cast(float, parameters[3])
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self.theta = regulate_theta(self.theta)
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        dimension = self.dimensions
        theta = self.theta
        phi = self.phi
        matrix = np.identity(dimension, dtype="complex")

        matrix[self.lev_a, self.lev_a] = np.cos(theta / 2) * matrix[self.lev_a, self.lev_a]
        matrix[self.lev_b, self.lev_b] = np.cos(theta / 2) * matrix[self.lev_b, self.lev_b]

        cosine_matrix = matrix
        pa: list[int | str] = [self.lev_a, self.lev_b, "a"]
        ps: list[int | str] = [self.lev_a, self.lev_b, "s"]
        qudit_targeted = cast(int, self.target_qudits)

        return cosine_matrix - 1j * np.sin(theta / 2) * (
            np.sin(phi)
            * GellMann(self.parent_circuit, "Gellman_a", qudit_targeted, pa, self.dimensions, None).to_matrix()
            + np.cos(phi)
            * GellMann(self.parent_circuit, "Gellman_s", qudit_targeted, ps, self.dimensions, None).to_matrix()
        )

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
            assert isinstance(parameter[3], float)
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
        return theta_cost(self.theta)

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions
