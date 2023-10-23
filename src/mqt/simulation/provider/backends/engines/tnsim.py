from typing import Iterable, List, Union

import numpy as np
import tensornetwork as tn

from mqt.circuit.circuit import QuantumCircuit
from mqt.circuit.components.instructions.gate import Gate
from mqt.circuit.components.instructions.gate_extensions.gatetypes import GateTypes
from mqt.simulation.provider.backend_properties.quditproperties import QuditProperties
from mqt.simulation.provider.backends.backendv2 import Backend


class TNSim(Backend):
    @property
    def target(self):
        raise NotImplementedError

    @property
    def max_circuits(self):
        raise NotImplementedError

    def qudit_properties(self, qudit: Union[int, List[int]]) -> Union[QuditProperties, List[QuditProperties]]:
        raise NotImplementedError

    def drive_channel(self, qudit: int):
        raise NotImplementedError

    def measure_channel(self, qudit: int):
        raise NotImplementedError

    def acquire_channel(self, qudit: int):
        raise NotImplementedError

    def control_channel(self, qudits: Iterable[int]):
        raise NotImplementedError

    def run(self, circuit: QuantumCircuit, **options):
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions
        return self.__execute(self.system_sizes, self.circ_operations)

    @classmethod
    def _default_options(cls):
        pass

    def __init__(self, **fields):
        self.system_sizes = None
        self.circ_operations = None
        super().__init__(**fields)

    def __apply_gate(self, qudit_edges, gate, operating_qudits):
        op = tn.Node(gate)
        for i, bit in enumerate(operating_qudits):
            tn.connect(qudit_edges[bit], op[i])
            qudit_edges[bit] = op[i + len(operating_qudits)]

    def __execute(self, system_sizes, operations: List[Gate]):
        all_nodes = []

        with tn.NodeCollection(all_nodes):
            state_nodes = []
            for s in system_sizes:
                z = [0] * s
                z[0] = 1
                state_nodes.append(tn.Node(np.array(z, dtype="complex")))

            qudits_legs = [node[0] for node in state_nodes]
            for op in operations:
                op_matrix = op.to_matrix()
                lines = op.ref_lines

                if op.gate_type == GateTypes.TWO:
                    op_matrix = op_matrix.reshape(
                        system_sizes[lines[0]], system_sizes[lines[1]], system_sizes[lines[0]], system_sizes[lines[1]]
                    )

                # TODO MULTI-QUDIT CASE
                self.__apply_gate(qudits_legs, op_matrix, lines)

        return tn.contractors.optimal(all_nodes, output_edge_order=qudits_legs)
