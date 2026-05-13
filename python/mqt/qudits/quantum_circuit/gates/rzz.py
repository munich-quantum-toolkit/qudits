from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from ...compiler.compilation_minitools.local_compilation_minitools import regulate_theta

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter

class Rzz(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit | None,
        name: str,
        target_qudits: list[int],
        parameters: list[int | float],
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ) -> None:
        # Falls nur eine Dimension übergeben wird -> automatisch [d, d]
        if isinstance(dimensions, int):
            dimensions = [dimensions, dimensions]

        if not isinstance(dimensions, list) or len(dimensions) != 2:
            raise TypeError("dimensions must be int or list[int, int]")

        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="rzz",
        )

        # Parameter-Zuweisung mit Fallback für den Pytest (nur phi übergeben)
        if self.validate_parameter(parameters):
            if len(parameters) == 3:
                self.lev_a = cast(int, parameters[0])
                self.lev_b = cast(int, parameters[1])
                self.phi = regulate_theta(cast(float, parameters[2]))
            elif len(parameters) == 1:
                # Fallback: Wenn nur ein Wert übergeben wurde, ist es phi.
                # Wir nehmen Standard-Levels 0 und 1 an.
                self.lev_a = 0
                self.lev_b = 1
                self.phi = regulate_theta(cast(float, parameters[0]))
            
            self._params = parameters
        else:
            raise ValueError(f"Invalid parameters for Rzz gate: {parameters}")

    def array(self) -> NDArray[np.complex128]:
        dim_ctrl, dim_target = self.dimensions
        dim_total = dim_ctrl * dim_target
        phi = self.phi

        result = np.zeros((dim_total, dim_total), dtype=np.complex128)

        # Erstelle Z-Vektoren für beide Qudits
        # Logik: 1 für Indizes zwischen lev_a und lev_b, sonst -1
        z_vals_ctrl = np.array([
            1 if self.lev_a <= i <= self.lev_b else -1 
            for i in range(dim_ctrl)
        ])
        
        z_vals_target = np.array([
            1 if self.lev_a <= j <= self.lev_b else -1 
            for j in range(dim_target)
        ])

        # Berechne die diagonale Matrix für exp(-i * phi/2 * Z ⊗ Z)
        for i in range(dim_ctrl):
            for j in range(dim_target):
                # Das Produkt der Z-Werte bestimmt die Phase
                z_product = z_vals_ctrl[i] * z_vals_target[j]
                phase = np.exp(-1j * phi * z_product / 2)
                
                index = i * dim_target + j
                result[index, index] = phase

        return result

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if not isinstance(parameter, list):
            return False

        # Erlaube entweder [phi] ODER [lev_a, lev_b, phi]
        if len(parameter) == 1:
            return isinstance(parameter[0], (int, float))
        
        if len(parameter) == 3:
            # Prüfe Typen: [int, int, float]
            if not (isinstance(parameter[0], int) and 
                    isinstance(parameter[1], int) and 
                    isinstance(parameter[2], (int, float))):
                return False
            
            # Logische Prüfung der Level
            if parameter[0] < 0 or parameter[1] < 0:
                return False
            if parameter[0] == parameter[1]:
                return False
            return True

        return False

    @property
    def dimensions(self) -> list[int]:
        return cast(list[int], self._dimensions)