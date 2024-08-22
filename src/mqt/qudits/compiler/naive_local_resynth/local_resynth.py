from __future__ import annotations

from typing import NoReturn

import numpy as np

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.core.lanes import Lanes
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import CustomOne


class NaiveLocResynthOptPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def transpile_gate(self, gate) -> NoReturn:
        raise NotImplementedError

    def transpile(self, circuit):
        self.circuit = circuit
        self.lanes = Lanes(self.circuit)

        for line in sorted(self.lanes.index_dict.keys()):
            grouped_line = self.lanes.find_consecutive_singles(self.lanes.index_dict[line])
            new_line = []
            for group in grouped_line[line]:
                if group[0][1].gate_type == GateTypes.SINGLE:
                    matrix = np.identity(self.circuit.dimensions[line])
                    for gate_tuple in group:
                        gate = gate_tuple[1]
                        gm = gate.to_matrix()
                        matrix = gm @ matrix
                    new_line.append((
                        group[0][0],
                        CustomOne(self.circuit, "CUm", line, matrix, self.circuit.dimensions[line]),
                    ))
                else:
                    new_line.append(group[0])

            self.lanes.index_dict[line] = new_line

        new_instructions = self.lanes.extract_instructions()

        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)
