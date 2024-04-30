from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.onedit.local_operation_swap.swap_routine import (
    cost_calculator,
    find_logic_from_phys,
    gate_chain_condition,
    graph_rule_ongate,
    graph_rule_update,
    route_states2rotate_basic,
)
from mqt.qudits.compiler.onedit.local_operation_swap.swap_routine import cost_calculator, find_logic_from_phys, \
    gate_chain_condition, \
    graph_rule_ongate, \
    graph_rule_update, route_states2rotate_basic
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.gates import R


class Test(TestCase):
    def setUp(self) -> None:
        test_sample_edges = [
            (0, 4, {"delta_m": 0, "sensitivity": 1}),
            (0, 3, {"delta_m": 1, "sensitivity": 3}),
            (0, 2, {"delta_m": 1, "sensitivity": 3}),
            (1, 4, {"delta_m": 0, "sensitivity": 1}),
            (1, 3, {"delta_m": 1, "sensitivity": 3}),
            (1, 2, {"delta_m": 1, "sensitivity": 3}),
        ]
        test_sample_edges = [(0, 4, {"delta_m": 0, "sensitivity": 1}),
                             (0, 3, {"delta_m": 1, "sensitivity": 3}),
                             (0, 2, {"delta_m": 1, "sensitivity": 3}),
                             (1, 4, {"delta_m": 0, "sensitivity": 1}),
                             (1, 3, {"delta_m": 1, "sensitivity": 3}),
                             (1, 2, {"delta_m": 1, "sensitivity": 3})
                             ]
        test_sample_nodes = [0, 1, 2, 3, 4]
        test_sample_nodes_map = [3, 2, 4, 1, 0]

        self.circuit = QuantumCircuit(1, [5], 0)
        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0], 0, self.circuit)
        self.graph_1.phase_storing_setup()

    def test_find_logic_from_phys(self):
        plev_a = 0
        plev_b = 1

        la, lb = find_logic_from_phys(plev_a, plev_b, self.graph_1)
        assert la == 4
        assert lb == 3
        self.assertEqual(la, 4)
        self.assertEqual(lb, 3)

    def test_graph_rule_update(self):
        self.circuit = QuantumCircuit(1, [5], 0)
        gate = self.circuit.r(0, [0, 1, np.pi, np.pi / 2])  # R(np.pi, np.pi / 2, 0, 1, 5)
        nodes_data = self.graph_1.nodes(data=True)

        _la, lb = find_logic_from_phys(0, 1, self.graph_1)
        assert nodes_data[lb]["phase_storage"] == 0.0
        la, lb = find_logic_from_phys(0, 1, self.graph_1)
        self.assertEqual(nodes_data[lb]['phase_storage'], 0.0)

        graph_rule_update(gate, self.graph_1)

        assert nodes_data[lb]["phase_storage"] == np.pi
        self.assertEqual(nodes_data[lb]['phase_storage'], np.pi)

    def test_graph_rule_ongate(self):
        self.circuit = QuantumCircuit(1, [5], 0)
        gate = self.circuit.r(0, [0, 1, np.pi, np.pi / 2])
        self.graph_1.nodes(data=True)
        nodes_data = self.graph_1.nodes(data=True)

        graph_rule_update(gate, self.graph_1)

        new_gate = graph_rule_ongate(gate, self.graph_1)

        assert new_gate.phi == 3 / 2 * np.pi
        self.assertEqual(new_gate.phi, 3 / 2 * np.pi)

    def test_gate_chain_condition(self):
        R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0])
        pi_pulses = [
            R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [1, 2, np.pi, np.pi / 2], self.circuit.dimensions[0]),
        ]
        pi_pulses = [R(self.circuit, "R", 0, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[0]),
                     R(self.circuit, "R", 0, [1, 2, np.pi, np.pi / 2], self.circuit.dimensions[0])]
        gate = R(self.circuit, "R", 0, [0, 2, np.pi / 3, np.pi / 2], self.circuit.dimensions[0])
        # R(np.pi / 3, np.pi / 2, 0, 2, 5)

        new_gate = gate_chain_condition(pi_pulses, gate)

        assert new_gate.theta == -np.pi / 3
        self.assertEqual(new_gate.theta, -np.pi / 3)

    def test_cost_calculator(self):
        test_sample_edges_1 = [
            (0, 1, {"delta_m": 1, "sensitivity": 1}),
            (0, 3, {"delta_m": 0, "sensitivity": 1}),
            (4, 3, {"delta_m": 0, "sensitivity": 1}),
            (4, 5, {"delta_m": 0, "sensitivity": 1}),
            (4, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_edges_1 = [(0, 1, {"delta_m": 1, "sensitivity": 1}),
                               (0, 3, {"delta_m": 0, "sensitivity": 1}),
                               (4, 3, {"delta_m": 0, "sensitivity": 1}),
                               (4, 5, {"delta_m": 0, "sensitivity": 1}),
                               (4, 2, {"delta_m": 0, "sensitivity": 1})
                               ]
        test_sample_nodes_1 = [0, 1, 2, 3, 4, 5]
        test_sample_nodes_map = [0, 1, 2, 3, 4, 5]
        non_zeros = 2
        self.circuit_6 = QuantumCircuit(1, [6], 0)
        # NODES CAN BE INFERRED BY THE EDGES
        test_graph_1 = LevelGraph(
            test_sample_edges_1, test_sample_nodes_1, test_sample_nodes_map, [1], 0, self.circuit_6
        )
        r_1 = R(self.circuit_6, "R", 0, [1, 4, np.pi / 4, 0.0], self.circuit_6.dimensions[0])
        test_graph_1 = LevelGraph(test_sample_edges_1, test_sample_nodes_1, test_sample_nodes_map,
                                  [1], 0, self.circuit_6)
        r_1 = R(self.circuit_6, "R", 0, [1, 4, np.pi / 4, 0.], self.circuit_6.dimensions[0])
        # R(np.pi / 4, 0, 1, 4, 6)

        total_costing, pi_pulses_routing, _new_placement, cost_of_pi_pulses, _gate_cost = cost_calculator(
            r_1, test_graph_1, non_zeros
        )
        total_costing, pi_pulses_routing, new_placement, cost_of_pi_pulses, gate_cost = cost_calculator(r_1,
                                                                                                        test_graph_1,
                                                                                                        non_zeros)

        assert total_costing == 0.00425
        assert len(pi_pulses_routing) == 2
        self.assertEqual(total_costing, 0.00425)
        self.assertEqual(len(pi_pulses_routing), 2)

        assert cost_of_pi_pulses == 0.002
        self.assertEqual(cost_of_pi_pulses, 2e-3)

    def test_route_states2rotate_basic(self):
        self.circuit_5 = QuantumCircuit(1, [5], 0)
        gate = R(self.circuit_5, "R", 0, [2, 4, np.pi / 3, np.pi / 2], self.circuit_5.dimensions[0])
        # R(np.pi / 3, np.pi / 2, 2, 4, 5)
        cost_of_pi_pulses, pi_pulses_routing, placement = route_states2rotate_basic(gate, self.graph_1)

        assert cost_of_pi_pulses == 0.0004
        assert len(pi_pulses_routing) == 1
        assert placement.log_phy_map == [0, 2, 4, 1, 3]
        self.assertEqual(cost_of_pi_pulses, 0.0004)
        self.assertEqual(len(pi_pulses_routing), 1)
        self.assertEqual(placement.log_phy_map, [0, 2, 4, 1, 3])
