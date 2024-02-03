from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.mini_tools.matrix_factory_tools import (
    from_dirac_to_basis,
    insert_at,
)

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class CEx(Gate):
    def __init__(
            self,
            circuit: QuantumCircuit,
            name: str,
            target_qudits: list[int] | int,
            parameters: list | None,
            dimensions: list[int] | int,
            controls: ControlData | None = None,
    ):
        super().__init__(
                circuit=circuit,
                name=name,
                gate_type=GateTypes.TWO,
                target_qudits=target_qudits,
                dimensions=dimensions,
                control_set=controls,
        )
        self._params = parameters
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b, self.ctrl_lev, self.phi = parameters
            self._params = parameters
        else:
            self.lev_a, self.lev_b, self.ctrl_lev, self.phi = None, None, None, None
            self._params = None
        self.qasm_tag = "cx"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        if self._params is None:
            ang = 0
            ctrl_level = 1
            levels_swap_low = 0
            levels_swap_high = 1
        else:
            levels_swap_low, levels_swap_high, ctrl_level, ang = self._params

        dimension = reduce(lambda x, y: x * y, self._dimensions)
        dimension_ctrl, dimension_target = self._dimensions
        result = np.zeros((dimension, dimension), dtype="complex")

        for i in range(dimension_ctrl):
            temp = np.zeros((dimension_ctrl, dimension_ctrl), dtype="complex")
            mapmat = temp + np.outer(
                    np.array(from_dirac_to_basis([i], dimension_ctrl)),
                    np.array(from_dirac_to_basis([i], dimension_ctrl))
            )

            if i == ctrl_level:  # apply control on 1 rotation on levels 01
                #opmat = np.array([[0, -1j * np.cos(ang) - np.sin(ang)], [-1j * np.cos(ang) + np.sin(ang), 0]])
                embedded_op = np.identity(dimension_target, dtype="complex")
                embedded_op[levels_swap_low, levels_swap_low] = 0
                embedded_op[levels_swap_low, levels_swap_high] = -1j * np.cos(ang) - np.sin(ang)
                embedded_op[levels_swap_high, levels_swap_low] = -1j * np.cos(ang) + np.sin(ang)
                embedded_op[levels_swap_high, levels_swap_high] = 0
                #embedded_op = insert_at(embedded_op, (0, 0), opmat)
            else:
                embedded_op = np.identity(dimension_target, dtype="complex")
            if self._target_qudits[0] < self._target_qudits[1]:
                result = result + np.kron(mapmat, embedded_op)
            else:
                result = result + np.kron(embedded_op, mapmat)

        return result

    def validate_parameter(self, parameter):
        if parameter is None:
            return False
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)
        assert isinstance(parameter[2], int)
        assert isinstance(parameter[3], float)
        assert (
                0 <= parameter[0] < parameter[1]
        ), f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
        assert 0 <= parameter[3] <= 2 * np.pi, f"Angle should be in the range [0, 2*pi]: {parameter[2]}"

        return True

    def __str__(self):
        # TODO
        pass
