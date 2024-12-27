from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...exceptions.circuiterror import InvalidQuditDimensionError
from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class S(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        dimensions: int,
        controls: ControlData | None = None,
    ) -> None:
        if self.is_prime(dimensions):
            super().__init__(
                circuit=circuit,
                name=name,
                gate_type=GateTypes.SINGLE,
                target_qudits=target_qudits,
                dimensions=dimensions,
                control_set=controls,
                qasm_tag="s",
            )
        else:
            msg = "S can be applied to prime dimensional qudits"
            raise InvalidQuditDimensionError(msg)

    def __array__(self) -> NDArray:  # noqa: PLW3201
        if self.dimensions == 2:
            matrix = np.array([[1, 0], [0, 1j]])
        else:
            matrix = np.zeros((self.dimensions, self.dimensions), dtype="complex")
            for i in range(self.dimensions):
                omega = np.e ** (2 * np.pi * 1j / self.dimensions)
                omega **= np.mod(i * (i + 1) / 2, self.dimensions)
                array = np.zeros(self.dimensions, dtype="complex")
                array[i] = 1
                result = omega * np.outer(array, array)
                matrix += result

        if self.dagger:
            return matrix.conj().T

        return matrix

    @staticmethod
    def is_prime(n: int) -> bool:
        """Check if a number is prime.

        Optimized for dimensions under 23.

        Args:
        n (int): The number to check for primality

        Returns:
        bool: True if the number is prime, False otherwise
        """
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        return all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions

    def to_qasm(self) -> str:
        string_description = self.__qasm__()
        if self.dagger:
            return "inv @ " + string_description
        return string_description
