from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, cast

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..components.extensions.matrix_factory import from_dirac_to_basis
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class CEx(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        parameters: list[int | float] | None,
        dimensions: list[int],
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="cx",
            lev_b=1,
            params=[0, 1, 1, 0.0],
        )
        # if customized
        if parameters is not None and self.validate_parameter(parameters):
            self.lev_a: int = cast("int", parameters[0])
            self.lev_b: int = cast("int", parameters[1])
            self.ctrl_lev: int = cast("int", parameters[2])
            self.phi: float = cast("float", parameters[3])
            # self.lev_a, self.lev_b, self.ctrl_lev, self.phi = parameters
            self._params: list[int | float] = parameters
        else:
            self.ctrl_lev = 1

    def __array__(self) -> NDArray:  # noqa: PLW3201
        levels_swap_low: int = cast("int", self._params[0])
        levels_swap_high: int = cast("int", self._params[1])
        ctrl_level: int = cast("int", self._params[2])
        ang: float = cast("float", self._params[3])
        dimension = reduce(operator.mul, self.dimensions)
        dimension_ctrl, dimension_target = self.dimensions
        qudits_targeted = cast("list[int]", self.target_qudits)
        result = np.zeros((dimension, dimension), dtype="complex")

        for i in range(dimension_ctrl):
            temp = np.zeros((dimension_ctrl, dimension_ctrl), dtype="complex")
            mapmat = temp + np.outer(
                np.array(from_dirac_to_basis([i], dimension_ctrl)), np.array(from_dirac_to_basis([i], dimension_ctrl))
            )

            if i == ctrl_level:  # apply control on 1 rotation on levels 01
                # opmat = np.array([[0, -1j * np.cos(ang) - np.sin(ang)], [-1j * np.cos(ang) + np.sin(ang), 0]])
                embedded_op = np.identity(dimension_target, dtype="complex")
                embedded_op[levels_swap_low, levels_swap_low] = 0
                embedded_op[levels_swap_low, levels_swap_high] = -1j * np.cos(ang) - np.sin(ang)
                embedded_op[levels_swap_high, levels_swap_low] = -1j * np.cos(ang) + np.sin(ang)
                embedded_op[levels_swap_high, levels_swap_high] = 0
                # embedded_op = insert_at(embedded_op, (0, 0), opmat)
            else:
                embedded_op = np.identity(dimension_target, dtype="complex")
            if qudits_targeted[0] < qudits_targeted[1]:
                result += np.kron(mapmat, embedded_op)
            else:
                result += np.kron(embedded_op, mapmat)

        return result

    @staticmethod
    def validate_parameter(parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            assert isinstance(parameter[0], int)
            assert isinstance(parameter[1], int)
            assert isinstance(parameter[2], int)
            assert isinstance(parameter[3], float)
            assert 0 <= parameter[0] < parameter[1], (
                f"lev_a and lev_b are out of range or in wrong order: {parameter[0]}, {parameter[1]}"
            )
            assert 0 <= parameter[3] <= 2 * np.pi, f"Angle should be in the range [0, 2*pi]: {parameter[2]}"

            return True

        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast("list[int]", self._dimensions)
