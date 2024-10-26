from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import expm  # type: ignore[import-not-found]

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .gellmann import GellMann

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class MS(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        parameters: list[float],
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
            qasm_tag="ms",
            params=parameters,
            theta=parameters[0],
        )
        if self.validate_parameter(parameters):
            self.theta = parameters[0]
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        theta = self.theta
        dimension_0 = self.dimensions[0]
        dimension_1 = self.dimensions[1]
        ps: list[int | str] = [0, 1, "s"]
        qudits_targeted = cast(list[int], self.target_qudits)
        qudit_targeted_0: int = qudits_targeted[0]
        qudit_targeted_1: int = qudits_targeted[1]

        gate_part_1 = np.kron(
            np.identity(dimension_0, dtype="complex"),
            GellMann(self.parent_circuit, "Gellman_s", qudit_targeted_1, ps, dimension_1, None).to_matrix(),
        ) + np.kron(
            GellMann(self.parent_circuit, "Gellman_s", qudit_targeted_0, ps, dimension_0, None).to_matrix(),
            np.identity(dimension_1, dtype="complex"),
        )
        gate_part_2 = np.kron(
            np.identity(dimension_0, dtype="complex"),
            GellMann(self.parent_circuit, "Gellman_s", qudit_targeted_1, ps, dimension_1, None).to_matrix(),
        ) + np.kron(
            GellMann(self.parent_circuit, "Gellman_s", qudit_targeted_0, ps, dimension_0, None).to_matrix(),
            np.identity(dimension_1, dtype="complex"),
        )
        return expm(-1j * theta * gate_part_1 @ gate_part_2 / 4)

    @staticmethod
    def validate_parameter(parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            assert 0 <= cast(float, parameter[0]) <= 2 * np.pi, (
                f"Angle should be in the range [0, 2*pi]: {parameter[0]}"
            )
            return True
        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast(list[int], self._dimensions)
