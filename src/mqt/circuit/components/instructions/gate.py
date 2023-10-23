from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from mqt.circuit.components.instructions.gate_extensions.controls import ControlData
from mqt.circuit.components.instructions.gate_extensions.gatetypes import GateTypes
from mqt.circuit.components.instructions.instruction import Instruction
from mqt.exceptions.circuiterror import CircuitError

if TYPE_CHECKING:
    import enum

    import numpy as np
    from numpy import ndarray

    from mqt.circuit.circuit import QuantumCircuit


class Gate(Instruction, ABC):
    """Unitary gate."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        gate_type: enum,
        target_qudits: list[int] | int,
        dimensions: list[int] | int,
        params: list | None = None,
        control_set=None,
        label: str | None = None,
        duration=None,
        unit="dt",
    ) -> None:
        self.parent_circuit = circuit
        # self.definition = None #todo unsure whether necesssary
        self._name = name
        self._target_qudits = target_qudits
        self._dimensions = dimensions
        self._params = params
        self._label = label
        self._duration = duration
        self._unit = unit
        self._controls_data = None
        if control_set:
            self.control(**vars(control_set))
        # TODO do it with inheritance one day
        self.gate_type = gate_type

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    @property
    def ref_lines(self):
        if isinstance(self._target_qudits, int):
            lines = self.get_control_lines
            lines.append(self._target_qudits)
        elif isinstance(self._target_qudits, list):
            lines = self._target_qudits + self.get_control_lines
        if len(lines) == 0:
            msg = "Gate has no target or control lines"
            raise CircuitError(msg)
        return lines

    @abstractmethod
    def __array__(self, dtype: str = "complex") -> np.ndarray:
        pass

    def to_matrix(self) -> Callable[[str], ndarray]:
        """Return a np.ndarray for the gate unitary matrix.

        Returns:
            np.ndarray: if the Gate subclass has a matrix definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            return self.__array__()
        msg = "to_matrix not defined for this "
        raise CircuitError(msg, {type(self)})

    def control(self, indices: list[int] | int, ctrl_states: list[int] | int):
        if len(indices) > self.parent_circuit.num_qudits or any(
            idx >= self.parent_circuit.num_qudits for idx in indices
        ):
            msg = "Indices or Number of Controls is beyond the Quantum Circuit Size"
            raise IndexError(msg)
        if any(idx in self._target_qudits for idx in indices):
            msg = "Controls overlap with targets"
            raise IndexError(msg)
        if any(ctrl >= self._dimensions[i] for i, ctrl in enumerate(ctrl_states)):
            msg = "Controls States beyond qudit size "
            raise IndexError(msg)
        if self.ref_lines < 2:
            self.set_gate_type_single()
        elif self.ref_lines == 2:
            self.set_gate_type_two()
        elif self.ref_lines > 2:
            self.set_gate_type_multi()
        self._controls_data = ControlData(indices, ctrl_states)

    @abstractmethod
    def validate_parameter(self, parameter):
        pass

    @abstractmethod
    def __qasm__(self) -> str:
        pass

    @abstractmethod
    def __str__(self):
        # String representation for drawing?
        pass

    def set_gate_type_single(self):
        self.gate_type = GateTypes.SINGLE

    def set_gate_type_two(self):
        self.gate_type = GateTypes.TWO

    def set_gate_type_multi(self):
        self.gate_type = GateTypes.MULTI

    @property
    def get_control_lines(self):
        if self._controls_data:
            return self._controls_data.indices
        return []
