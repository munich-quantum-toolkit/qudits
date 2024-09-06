from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import MatrixFactory

from ..exceptions import CircuitError
from .components.extensions.controls import ControlData
from .components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Parameter = Optional[Union[list[Union[int, float]], list[Union[int, str]], NDArray[np.complex128, np.complex128]]]
    from .circuit import QuantumCircuit


class Instruction(ABC):
    @abstractmethod
    def __init__(self, name: str) -> None:
        pass


class Gate(Instruction):
    """Unitary gate_matrix."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        gate_type: GateTypes,
        target_qudits: list[int] | int,
        dimensions: list[int] | int,
        params: Parameter = None,
        control_set: ControlData | None = None,
        label: str | None = None,
        lev_a: int = 0,
        lev_b: int = 0,
        theta: float = 0.0,
        phi: float = 0.0,
        qasm_tag: str = "",
    ) -> None:
        self.dagger = False
        self.parent_circuit = circuit
        self._name = name
        self.gate_type = gate_type
        self._target_qudits = target_qudits
        self._dimensions = dimensions
        self._params: Parameter = params
        self._label = label
        self._controls_data: ControlData | None = None
        if control_set:
            self.control(**vars(control_set))

        self.is_long_range = self.check_long_range()
        # inheritable parameters
        self.lev_a: int = lev_a
        self.lev_b: int = lev_b
        self.theta: float = theta
        self.phi: float = phi
        self.qasm_tag = qasm_tag

    @property
    def reference_lines(self) -> list[int]:
        lines = []
        if isinstance(self.target_qudits, int):
            lines = self.get_control_lines.copy()
            lines.append(self.target_qudits)
        elif isinstance(self.target_qudits, list):
            lines = self.get_control_lines.copy() + self.target_qudits.copy()
        if len(lines) == 0:
            msg = "Gate has no target or control lines"
            raise CircuitError(msg)
        return lines

    @abstractmethod
    def __array__(self) -> NDArray:  # noqa: PLW3201
        pass

    def dag(self) -> Gate:
        self._name += "_dag"
        self.dagger = True
        return self

    def to_matrix(self, identities: int = 0) -> NDArray:
        """Return a np.ndarray for the gate_matrix unitary parameters.

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
        raise CircuitError(msg)

    def control(self, indices: list[int], ctrl_states: list[int]) -> Gate:
        if len(indices) == 0 or len(ctrl_states) == 0:
            return self
        # AT THE MOMENT WE SUPPORT CONTROL OF SINGLE QUDIT GATES
        assert self.gate_type == GateTypes.SINGLE
        if len(indices) > self.parent_circuit.num_qudits or any(
            idx >= self.parent_circuit.num_qudits for idx in indices
        ):
            msg = "Indices or Number of Controls is beyond the Quantum Circuit Size"
            raise IndexError(msg)
        if isinstance(self.target_qudits, int):
            if self.target_qudits in indices:
                msg = "Controls overlap with targets"
                raise IndexError(msg)
        elif any(idx in list(self.target_qudits) for idx in indices):
            msg = "Controls overlap with targets"
            raise IndexError(msg)
        # if isinstance(self._dimensions, int):
        #    dimensions = [self._dimensions]
        if any(ctrl >= self.parent_circuit.dimensions[i] for i, ctrl in enumerate(ctrl_states)):
            msg = "Controls States beyond qudit size "
            raise IndexError(msg)
        self._controls_data = ControlData(indices, ctrl_states)
        # Set new type
        if len(self.reference_lines) == 2:
            self.set_gate_type_two()
        elif len(self.reference_lines) > 2:
            self.set_gate_type_multi()
        self.check_long_range()
        return self

    def validate_parameter(self, param: Parameter) -> bool:  # noqa: PLR6301 ARG002
        return False

    @property
    def dimensions(self) -> list[int] | int:
        return self._dimensions

    @property
    def target_qudits(self) -> list[int] | int:
        """Get the target qudits.

        Returns:
            Union[List[int], np.ndarray]: The current target qudits.
        """
        return self._target_qudits

    @target_qudits.setter
    def target_qudits(self, value: list[int] | int) -> None:
        self._target_qudits = value

    def __qasm__(self) -> str:  # noqa: PLW3201
        """Generate QASM for Gate export."""
        string = f"{self.qasm_tag} "
        if isinstance(self._params, np.ndarray):
            string += self.return_custom_data()
        elif self._params:
            string += "("
            for parameter in self._params:
                string += f"{parameter}, "
            string = string[:-2]
            string += ") "
        if isinstance(self.target_qudits, int):
            targets = [self.target_qudits]
        elif isinstance(self.target_qudits, list):
            targets = self.target_qudits
        for qudit in targets:
            string += (
                f"{self.parent_circuit.inverse_sitemap[qudit][0]}[{self.parent_circuit.inverse_sitemap[qudit][1]}], "
            )
        string = string[:-2]
        if self._controls_data is not None:
            string += " ctl "
            for _ctrl in self._controls_data.indices:
                string += (
                    f"{self.parent_circuit.inverse_sitemap[qudit][0]}[{self.parent_circuit.inverse_sitemap[qudit][1]}] "
                )
            string += str(self._controls_data.ctrl_states)

        return string + ";\n"

    def check_long_range(self) -> bool:
        target_qudits: list[int] = self.reference_lines
        if len(target_qudits) > 1:
            return any((b - a) > 1 for a, b in zip(sorted(target_qudits)[:-1], sorted(target_qudits)[1:]))
        return False

    def set_gate_type_single(self) -> None:
        self.gate_type = GateTypes.SINGLE

    def set_gate_type_two(self) -> None:
        self.gate_type = GateTypes.TWO

    def set_gate_type_multi(self) -> None:
        self.gate_type = GateTypes.MULTI

    @property
    def get_control_lines(self) -> list[int]:
        if self._controls_data:
            return self._controls_data.indices
        return []

    @property
    def control_info(self) -> dict[str, int | list[int] | Parameter | ControlData]:
        return {
            "target": self.target_qudits,
            "dimensions_slice": self._dimensions,
            "params": self._params,
            "controls": self._controls_data,
        }

    def return_custom_data(self) -> str:
        if not self.parent_circuit.path_save:
            return "(custom_data) "

        key = "".join(random.choice(string.ascii_letters) for _ in range(4))  # noqa: S311
        file_path = Path(self.parent_circuit.path_save) / f"{self._name}_{key}.npy"
        np.save(file_path, self._params)
        return f"({file_path}) "
