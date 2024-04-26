from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools import new_mod, phi_cost, pi_mod, regulate_theta, rotation_cost_calc, \
    swap_elements, \
    theta_cost
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCompilationMiniTools(TestCase):

    def test_swap_elements(self):
        example = [0, 1, 2, 3]
        test_swapped = [3, 1, 2, 0]
        swapped_example = swap_elements(example, 0, 3)

        self.assertEqual(swapped_example, test_swapped)

    def test_pi_mod(self):
        res = pi_mod(3 * np.pi / 2)
        self.assertEqual(res, -np.pi / 2)

        res = pi_mod(-3 * np.pi / 2)
        self.assertEqual(res, np.pi / 2)

    def test_new_mod(self):
        res = new_mod(-5 * np.pi / 2)
        self.assertEqual(res, -np.pi / 2)

        res = new_mod(5 * np.pi / 2)
        self.assertEqual(res, np.pi / 2)

    def test_regulate_theta(self):
        newang = regulate_theta(-5 * np.pi)
        self.assertEqual(newang, -1 * np.pi)

        newang = regulate_theta(0.1 * np.pi)
        self.assertEqual(newang, 4.1 * np.pi)

    def test_theta_cost(self):
        cost = theta_cost(np.pi / 8)
        self.assertEqual(cost, 6.25e-05)

        cost = theta_cost(np.pi / 4)
        self.assertEqual(cost, 1.25e-04)

    def test_phi_cost(self):
        cost = phi_cost(np.pi / 8)
        self.assertEqual(cost, 1.25e-05)

        cost = phi_cost(np.pi / 4)
        self.assertEqual(cost, 2.5e-05)

    def test_rotation_cost_calc(self):
        test_sample_edges_1 = [(0, 1, {"delta_m": 1, "sensitivity": 1}),
                               (0, 3, {"delta_m": 0, "sensitivity": 1}),
                               (4, 3, {"delta_m": 0, "sensitivity": 1}),
                               (4, 5, {"delta_m": 0, "sensitivity": 1}),
                               (4, 2, {"delta_m": 0, "sensitivity": 1})
                               ]
        test_sample_nodes_1 = [0, 1, 2, 3, 4, 5]
        test_sample_nodes_map = [0, 1, 2, 3, 4, 5]

        circuit = QuantumCircuit(1, [6], 0)
        # NODES CAN BE INFERRED BY THE EDGES
        test_graph_1 = LevelGraph(test_sample_edges_1, test_sample_nodes_1, test_sample_nodes_map, [1], 0, circuit)

        R_1 = circuit.r(0, [2, 4, np.pi / 4, 0.])  # R(np.pi / 4, 0, 2, 4, 6)
        cost_1 = rotation_cost_calc(R_1, test_graph_1)
        self.assertEqual(cost_1, 4 * 1.25e-4)

        R_2 = circuit.r(0, [3, 4, np.pi / 4, 0.])  # R(np.pi / 4, 0, 3, 4, 6)
        cost_2 = rotation_cost_calc(R_2, test_graph_1)
        self.assertEqual(cost_2, 3 * 1.25e-4)
