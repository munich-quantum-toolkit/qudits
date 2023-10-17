from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mqt.exceptions.circuiterror import CircuitError

if TYPE_CHECKING:
    import numpy as np


class Gate(ABC):
    """Unitary gate."""

    def __init__(
        self,
        name: str,
        num_qudits: int,
        params: list,
        label: str | None = None,
        duration=None,
        unit="dt",
    ) -> None:
        """Create a new gate.

        Args:
            name: The Qobj name of the gate.
            num_qudits: The number of qudits the gate acts on.
            params: A list of parameters.
            label: An optional label for the gate.
        """
        self.definition = None
        self._name = name
        self._num_qudits = num_qudits
        self._params = params
        self.label = label
        self.duration = duration
        self.unit = unit

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    @abstractmethod
    def __array__(self, dtype: str = "complex") -> np.ndarray:
        pass

    def to_matrix(self) -> np.ndarray:
        """Return a Numpy.array for the gate unitary matrix.

        Returns:
            np.ndarray: if the Gate subclass has a matrix definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            return self.__array__
        msg = "to_matrix not defined for this "
        raise CircuitError(msg, {type(self)})

    @abstractmethod
    def control(
        self,
        num_ctrl_qudits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
    ):
        pass
        """Return controlled version of gate. See :class:`.ControlledGate` for usage.

        Args:
            num_ctrl_qudits: number of controls to add to gate (default: ``1``)
            label: optional gate label
            ctrl_state: The control state in decimal or as a bitstring
                (e.g. ``'111'``). If ``None``, use ``2**num_ctrl_qudits-1``.

        Returns:
            qiskit.circuit.ControlledGate: Controlled version of gate. This default algorithm
            uses ``num_ctrl_qudits-1`` ancilla qubits so returns a gate of size
            ``num_qudits + 2*num_ctrl_qudits - 1``.

        Raises:
            QiskitError: unrecognized mode or invalid ctrl_state
        """

    @abstractmethod
    def validate_parameter(self, parameter):
        pass
        """
        if isinstance(parameter, ParameterExpression):
            if len(parameter.parameters) > 0:
                return parameter  # expression has free parameters, we cannot validate it
            if not parameter.is_real():
                msg = f"Bound parameter expression is complex in gate {self.name}"
                raise CircuitError(msg)
            return parameter  # per default assume parameters must be real when bound
        if isinstance(parameter, (int, float)):
            return parameter
        elif isinstance(parameter, (np.integer, np.floating)):
            return parameter.item()
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for gate {self.name}.")
        """

    @abstractmethod
    def __qasm__(self) -> str:
        pass
