from unittest import TestCase
import numpy as np

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestUnitaryVerifier(TestCase):

    def setUp(self) -> None:
        edges = [(0, 3, {"delta_m": 1, "sensitivity": 5}),
                 (0, 4, {"delta_m": 0, "sensitivity": 3}),
                 (1, 4, {"delta_m": 0, "sensitivity": 3}),
                 (1, 2, {"delta_m": 1, "sensitivity": 5})
                 ]
        nodes = [0, 1, 2, 3, 4]
        nodes_map = [0, 2, 1, 4, 3]
        self.circuit = QuantumCircuit(1, [5, 2, 3], 0)
        self.graph = LevelGraph(edges, nodes, nodes_map, [0], 0, self.circuit)

    def test_verify(self):
        dimension = 2

        sequence = [self.circuit.cu_one(1, np.identity(dimension, dtype='complex')),
                    self.circuit.h(1),
                    self.circuit.h(1)]
        target = self.circuit.cu_one(1, np.identity(dimension, dtype='complex'))

        nodes = [0, 1]
        initial_map = [0, 1]
        final_map = [0, 1]

        V1 = UnitaryVerifier(sequence, target, [dimension], nodes, initial_map, final_map)

        self.assertTrue(V1.verify())

        ##################################################################

        dimension = 3

        nodes_3 = [0, 1, 2]
        initial_map_3 = [0, 1, 2]
        final_map_3 = [0, 2, 1]

        sequence_3 = [self.circuit.cu_one(2, np.identity(dimension, dtype='complex')),
                      self.circuit.h(2),
                      self.circuit.x(2),
                      self.circuit.x(2),
                      self.circuit.x(2)]

        target_3 = self.circuit.h(2)

        V1 = UnitaryVerifier(sequence_3, target_3, [dimension], nodes_3, initial_map_3, final_map_3)

        self.assertTrue(V1.verify())
