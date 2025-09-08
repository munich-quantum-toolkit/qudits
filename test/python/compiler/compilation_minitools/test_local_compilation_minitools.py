from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools import (
    new_mod,
    phi_cost,
    pi_mod,
    regulate_theta,
    rotation_cost_calc,
    swap_elements,
    theta_cost,
)
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCompilationMiniTools(TestCase):
    @staticmethod
    def test_swap_elements():
        example = [0, 1, 2, 3]
        test_swapped = [3, 1, 2, 0]
        swapped_example = swap_elements(example, 0, 3)

        assert swapped_example == test_swapped

    @staticmethod
    def test_pi_mod():
        res = pi_mod(3 * np.pi / 2)
        assert res == -np.pi / 2

        res = pi_mod(-3 * np.pi / 2)
        assert res == np.pi / 2

    @staticmethod
    def test_new_mod():
        res = new_mod(-5 * np.pi / 2)
        assert res == -np.pi / 2

        res = new_mod(5 * np.pi / 2)
        assert res == np.pi / 2

    @staticmethod
    def test_regulate_theta():
        newang = regulate_theta(-5 * np.pi)
        assert newang == -1 * np.pi

        newang = regulate_theta(0.1 * np.pi)
        assert newang == 4.1 * np.pi

    @staticmethod
    def test_theta_cost():
        cost = theta_cost(np.pi / 8)
        assert cost == 6.25e-05

        cost = theta_cost(np.pi / 4)
        assert cost == 0.000125

    @staticmethod
    def test_phi_cost():
        cost = phi_cost(np.pi / 8)
        assert cost == 1.25e-05

        cost = phi_cost(np.pi / 4)
        assert cost == 2.5e-05

    @staticmethod
    def test_rotation_cost_calc():
        test_sample_edges_1 = [
            (0, 1, {"delta_m": 1, "sensitivity": 1}),
            (0, 3, {"delta_m": 0, "sensitivity": 1}),
            (4, 3, {"delta_m": 0, "sensitivity": 1}),
            (4, 5, {"delta_m": 0, "sensitivity": 1}),
            (4, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_nodes_1 = [0, 1, 2, 3, 4, 5]
        test_sample_nodes_map = [0, 1, 2, 3, 4, 5]

        circuit = QuantumCircuit(1, [6], 0)
        # NODES CAN BE INFERRED BY THE EDGES
        test_graph_1 = LevelGraph(test_sample_edges_1, test_sample_nodes_1, test_sample_nodes_map, [1], 0, circuit)

        r_1 = circuit.r(0, [2, 4, np.pi / 4, 0.0])  # R(np.pi / 4, 0, 2, 4, 6)
        cost_1 = rotation_cost_calc(r_1, test_graph_1)
        assert cost_1 == 4 * 0.000125

        r_2 = circuit.r(0, [3, 4, np.pi / 4, 0.0])  # R(np.pi / 4, 0, 3, 4, 6)
        cost_2 = rotation_cost_calc(r_2, test_graph_1)
        assert cost_2 == 3 * 0.000125
