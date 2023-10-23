from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.circuit.components.instructions.gate import Gate

if TYPE_CHECKING:
    import numpy as np

    from mqt.circuit.components.instructions.gate_extensions.controls import ControlData


class R(Gate):
    def __qasm__(self) -> str:
        pass

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        pass

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

    def __init__(self, name: str, num_qudits: int, params: list, controls: ControlData | None = None):
        super().__init__(name, num_qudits, params)

    def __str__(self):
        pass
