from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class Perm(Gate):
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
            params=parameters,
            qasm_tag="pm",
        )
        if self.validate_parameter(parameters):
            self.perm_data = parameters
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        return np.eye(self.dimensions)[:, self.perm_data]

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            """Verify that the input is a list of indices"""
            dims = self.dimensions
            p = cast(list[int], parameter)
            assert all((0 <= num < len(parameter) and num < dims) for num in p), (
                "Numbers are not within the range of the list length"
            )

            return True

        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> int:
        return cast(int, self._dimensions)
