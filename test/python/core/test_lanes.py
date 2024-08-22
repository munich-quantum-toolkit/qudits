from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.core.lanes import Lanes
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.gates import CEx, CustomMulti, R


class TestLanes(TestCase):
    def setUp(self):
        self.circuit = QuantumCircuit(3, [3, 3, 3], 0)

    def test_create_lanes(self):
        gates = [
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            CustomMulti(
                self.circuit, "CUm", [0, 1, 2], np.identity(18), [self.circuit.dimensions[i] for i in [0, 1, 2]]
            ),
        ]
        self.circuit.instructions = gates
        self.lanes = Lanes(self.circuit)
        expected_index_dict = {
            0: [(0, gates[0]), (0, gates[1]), (1, gates[2]), (2, gates[4])],
            1: [(1, gates[2]), (1, gates[3]), (2, gates[4])],
            2: [(2, gates[4])],
        }
        assert self.lanes.index_dict == expected_index_dict

    def test_extract_circuit(self):
        gates = [
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            CustomMulti(
                self.circuit, "CUm", [0, 1, 2], np.identity(18), [self.circuit.dimensions[i] for i in [0, 1, 2]]
            ),
        ]
        self.circuit.instructions = gates
        self.lanes = Lanes(self.circuit)
        result = self.lanes.extract_instructions()
        assert result == [gates[0], gates[1], gates[2], gates[3], gates[4]]

    def test_extract_lane(self):
        gates = [
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            CustomMulti(
                self.circuit, "CUm", [0, 1, 2], np.identity(18), [self.circuit.dimensions[i] for i in [0, 1, 2]]
            ),
        ]
        self.circuit.instructions = gates
        self.lanes = Lanes(self.circuit)
        result = self.lanes.extract_lane(1)
        assert result == [gates[2], gates[3], gates[4]]

    def test_find_consecutive_singles(self):
        gates = [
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            CustomMulti(
                self.circuit, "CUm", [0, 1, 2], np.identity(18), [self.circuit.dimensions[i] for i in [0, 1, 2]]
            ),
            R(self.circuit, "R", 2, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[2]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 2, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[2]),
        ]
        self.circuit.instructions = gates
        self.lanes = Lanes(self.circuit)
        result = self.lanes.find_consecutive_singles()
        expected_result = {
            0: [
                [(0, gates[0]), (0, gates[2])],
                [(1, gates[4])],
                [(1, gates[7]), (1, gates[8])],
                [(2, gates[9])],
                [(3, gates[11])],
            ],
            1: [
                [(0, gates[1]), (0, gates[3])],
                [(1, gates[4])],
                [(1, gates[5]), (1, gates[6])],
                [(2, gates[9])],
                [(3, gates[11])],
            ],
            2: [[(2, gates[9])], [(2, gates[10]), (3, gates[12])]],
        }
        assert result == expected_result
