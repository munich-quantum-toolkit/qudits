from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .r import R

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class Rh(Gate):
    """SU2 Hadamard."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: list[int],
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
            qasm_tag="rh",
        )
        if self.validate_parameter(parameters):
            self.original_lev_a = parameters[0]
            self.original_lev_b = parameters[1]
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        # (R(-np.pi, 0, l1, l2, dim) * R(np.pi / 2, np.pi / 2, l1, l2, dim))
        dimension = self.dimensions
        qudit_targeted: int = cast(int, self.target_qudits)

        pi_x = R(self.parent_circuit, "R", qudit_targeted, [self.lev_a, self.lev_b, -np.pi, 0.0], dimension).to_matrix()
        rotate = R(
            self.parent_circuit,
            "R",
            qudit_targeted,
            [self.lev_a, self.lev_b, np.pi / 2, np.pi / 2],
            dimension,
        ).to_matrix()

        return np.matmul(pi_x, rotate)

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
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions
