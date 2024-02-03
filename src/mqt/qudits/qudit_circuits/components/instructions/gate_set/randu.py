from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from scipy.stats import unitary_group

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes

if TYPE_CHECKING:
    import numpy as np

    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class RandU(Gate):
    def __init__(
            self,
            circuit: QuantumCircuit,
            name: str,
            target_qudits: list[int] | int,
            dimensions: list[int] | int,
            controls: ControlData | None = None,
    ):
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.MULTI,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        self.qasm_tag = "rdu"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dim = reduce(lambda x, y: x * y, self._dimensions)
        return unitary_group.rvs(dim)

    def validate_parameter(self, parameter):
        return True

    def __str__(self):
        # TODO
        pass
