from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

from mqt.qudits.exceptions.circuiterror import CircuitError
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.matrix_factory import MatrixFactory
from mqt.qudits.qudit_circuits.components.instructions.instruction import Instruction

if TYPE_CHECKING:
    import enum

    import numpy as np
    from numpy import ndarray

    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


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
        self.dagger = False
        self.parent_circuit = circuit
        self._name = name
        self._target_qudits = target_qudits
        self._dimensions = dimensions
        self._params = params
        self._label = label
        self._duration = duration
        self._unit = unit
        self._controls_data = None
        self.is_long_range = False
        if isinstance(target_qudits, list) and len(target_qudits) > 0:
            self.is_long_range = any((b - a) > 1 for a, b in zip(sorted(target_qudits)[:-1], sorted(target_qudits)[1:]))
        if control_set:
            self.control(**vars(control_set))
        # TODO do it with inheritance one day
        self.gate_type = gate_type

    # Set higher priority than Numpy array and parameters classes
    __array_priority__ = 20

    @property
    def reference_lines(self):
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

    def dag(self):
        self._name = self._name + "_dag"
        self.dagger = True

    def to_matrix(self, identities=0) -> Callable[[str], ndarray]:
        """Return a np.ndarray for the gate unitary parameters.

        Returns:
            np.ndarray: if the Gate subclass has a parameters definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            matrix_factory = MatrixFactory(self, identities)
            return matrix_factory.generate_matrix()
        msg = "to_matrix not defined for this "
        raise CircuitError(msg, {type(self)})

    def control(self, indices: list[int] | int, ctrl_states: list[int] | int):
        # AT THE MOMENT WE SUPPORT CONTROL OF SINGLE QUDIT GATES
        assert self.gate_type == GateTypes.SINGLE
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
        if self.reference_lines == 2:
            self.set_gate_type_two()
        elif self.reference_lines > 2:
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

    @property
    def control_info(self):
        return {
            "target": self._target_qudits,
            "dimensions_slice": self._dimensions,
            "params": self._params,
            "controls": self._controls_data,
        }
