from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class GellMann(Gate):
    """Gate used as generator for Givens rotations."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: list[int | str],
        dimensions: int,
        controls: ControlData | None = None,
    ) -> None:
        assert self.validate_parameter(parameters), "Invalid parameters"
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="gell",
            lev_a=cast(int, parameters[0]),
            lev_b=cast(int, parameters[1]),
            params=parameters,
        )
        self.type_m = cast(str, parameters[2])

    def __array__(self) -> NDArray:  # noqa: PLW3201
        d = self._dimensions
        matrix = np.zeros((d, d), dtype=complex)

        if self.type_m == "s":
            matrix[self.lev_a, self.lev_b] = 1
            matrix[self.lev_b, self.lev_a] = 1
        elif self.type_m == "a":
            matrix[self.lev_a, self.lev_b] -= 1j
            matrix[self.lev_b, self.lev_a] += 1j
        else:
            m = np.zeros((d, d), dtype=complex)

            for j_ind in range(self.lev_b):
                m[j_ind, j_ind] += 1

            m[self.lev_b, self.lev_b] -= self.lev_b

            coeff = np.sqrt(2 / (self.lev_b * (self.lev_b + 1)))

            m = coeff * m

            matrix = m

        return matrix

    @staticmethod
    def validate_parameter(param: Parameter) -> bool:
        if param is None:
            return False

        if isinstance(param, list):
            parameter = cast(list[Union[int, str]], param)
            assert isinstance(parameter[0], int)
            assert isinstance(parameter[1], int)
            assert isinstance(parameter[2], str)
            assert 0 <= parameter[0] < parameter[1], (
                f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
            )
            assert isinstance(parameter[2], str), "type parameter should be a string"

            return True

        if isinstance(param, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False
