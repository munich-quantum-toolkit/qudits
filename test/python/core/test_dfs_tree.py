from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.core import LevelGraph, NAryTree, Node
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.gates import R


class TestNode(TestCase):
    def setUp(self) -> None:
        test_sample_edges = [
            (0, 5, {"delta_m": 0, "sensitivity": 1}),
            (0, 4, {"delta_m": 0, "sensitivity": 1}),
            (0, 3, {"delta_m": 1, "sensitivity": 3}),
            (0, 2, {"delta_m": 1, "sensitivity": 3}),
            (1, 5, {"delta_m": 0, "sensitivity": 1}),
            (1, 4, {"delta_m": 0, "sensitivity": 1}),
            (1, 3, {"delta_m": 1, "sensitivity": 3}),
            (1, 2, {"delta_m": 1, "sensitivity": 3}),
        ]
        test_sample_nodes = [0, 1, 2, 3, 4, 5]
        test_sample_nodes_map = [0, 2, 5, 4, 1, 3]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])

        circuit = QuantumCircuit(1, [6], 0)
        self.U = circuit.h(0).to_matrix(identities=0)
        self.r = R(circuit, "R", 0, [0, 1, 0.1, 0.3], 6)

        self.root = Node(0, self.r, self.U, self.graph_1, 0, 10e-4, (1.0, 1.0), [], -1)

    def test_add(self):
        self.root.add(1, self.r, self.U, self.graph_1, 0, 10e-4, (1.0, 1.0), [])
        assert self.root.size == 1
        assert self.root.children[0].key == 1
        assert self.root.children[0].current_cost == 0

    def test_print(self):
        self.root.add(1, self.r, self.U, self.graph_1, 0, 10e-4, (1.0, 1.0), [])
        print(self.root.children[0])


class TestNAryTree(TestCase):
    def setUp(self) -> None:
        test_sample_edges = [
            (0, 5, {"delta_m": 0, "sensitivity": 1}),
            (0, 4, {"delta_m": 0, "sensitivity": 1}),
            (0, 3, {"delta_m": 1, "sensitivity": 3}),
            (0, 2, {"delta_m": 1, "sensitivity": 3}),
            (1, 5, {"delta_m": 0, "sensitivity": 1}),
            (1, 4, {"delta_m": 0, "sensitivity": 1}),
            (1, 3, {"delta_m": 1, "sensitivity": 3}),
            (1, 2, {"delta_m": 1, "sensitivity": 3}),
        ]
        test_sample_nodes = [0, 1, 2, 3, 4, 5]
        test_sample_nodes_map = [0, 2, 5, 4, 1, 3]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])

        circuit = QuantumCircuit(1, [6], 0)
        self.U = circuit.h(0).to_matrix(identities=0)
        self.r = R(circuit, "R", 0, [0, 1, 0.1, 0.3], 6)
        np.array([[1, 0], [0, 1]], dtype=np.float64)

        self.T = NAryTree()
        self.T.add(0, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [])

    def test_add(self):
        # new_key, rotation, U_of_level, graph_current, current_cost, current_decomp_cost, max_cost, pi_pulses,
        # parent_key
        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)

        assert self.T.root.children[0].key == 2
        assert self.T.root.children[1].key == 3

    def test_find_node(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)

        node = self.T.find_node(self.T.root, 2)
        if node is not None:
            assert node.parent_key == 0
            assert node.key == 2

    def test_depth(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)

        d0 = self.T.depth(0)
        d2 = self.T.depth(2)
        assert d0 == 2
        assert d2 == 1

    def test_max_depth(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 3)

        node = self.T.find_node(self.T.root, 2)
        if node is not None:
            d2 = self.T.max_depth(node)
        assert d2 == 2

    def test_size_refresh(self):
        size = self.T.size_refresh(self.T.root)

        assert size == 0

        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)
        self.T.add(5, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 3)

        size = self.T.size_refresh(self.T.root)

        assert size == 4

    def test_found_checker(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0.1, 0.1, (10.0, 10.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0.11, 0.1, (10.0, 10.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0.01, 0.01, (10.0, 10.0), [], 2)

        self.T.root.finished = False
        self.T.root.children[0].finished = False
        self.T.root.children[0].children[0].finished = False
        self.T.root.children[0].children[1].finished = True

        assert self.T.found_checker(self.T.root)
        self.T.root.finished = False
        self.T.root.children[0].finished = False
        self.T.root.children[0].children[0].finished = False
        self.T.root.children[0].children[1].finished = False

        assert not self.T.found_checker(self.T.root)

    def test_min_cost_decomp(self):
        R(QuantumCircuit(), "R", 0, [0, 1, 0.1, 0.3], 2)
        np.array([[1, 0], [0, 1]], dtype=np.float64)
        self.T.add(2, self.r, self.U, self.graph_1, 0.1, 0.1, (10.0, 10.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0.11, 0.1, (10.0, 10.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0.01, 0.01, (10.0, 10.0), [], 2)

        self.T.root.finished = True
        self.T.root.children[0].finished = True
        self.T.root.children[0].children[0].finished = True
        self.T.root.children[0].children[1].finished = True

        node_seq, best_cost, _final_graph = self.T.min_cost_decomp(self.T.root)
        assert best_cost[0] == 0.01
        assert best_cost[1] == 0.01
        assert node_seq[0].key == 0
        assert node_seq[1].key == 2
        assert node_seq[2].key == 4

    def test_retrieve_decomposition(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0.1, 0.1, (10.0, 10.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0.11, 0.1, (10.0, 10.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0.01, 0.01, (10.0, 10.0), [], 2)

        self.T.root.finished = False
        self.T.root.children[0].finished = False
        self.T.root.children[0].children[0].finished = False
        self.T.root.children[0].children[1].finished = True

        decomp_nodes, best_cost, graph = self.T.retrieve_decomposition(self.T.root)

        assert best_cost[0] == 0.01
        assert best_cost[1] == 0.01
        assert graph == self.graph_1
        assert decomp_nodes[0].key == 0
        assert decomp_nodes[1].key == 2
        assert decomp_nodes[2].key == 4

    def test_is_empty(self):
        self.T = NAryTree()
        assert self.T.is_empty()

        self.T.add(0, self.r, self.U, self.graph_1, 0.0, 0.0, (10.0, 10.0), [])
        assert not self.T.is_empty()

        self.T.add(2, self.r, self.U, self.graph_1, 0.1, 0.1, (10.0, 10.0), [], 0)
        assert not self.T.is_empty()

    def test_total_size(self):
        size = self.T.total_size
        R(QuantumCircuit(), "R", 0, [0, 1, 0.1, 0.3], 2)
        np.array([[1, 0], [0, 1]], dtype=np.float64)

        assert size == 1

        self.T.add(2, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 2)
        self.T.add(5, self.r, self.U, self.graph_1, 0, 0, (0.0, 0.0), [], 3)

        size = self.T.total_size

        assert size == 5

    def test_print_tree(self):
        self.T.add(2, self.r, self.U, self.graph_1, 0.1, 0.1, (10.0, 10.0), [], 0)
        self.T.add(3, self.r, self.U, self.graph_1, 0.11, 0.1, (10.0, 10.0), [], 2)
        self.T.add(4, self.r, self.U, self.graph_1, 0.01, 0.01, (10.0, 10.0), [], 2)

        tree_string = self.T.print_tree(self.T.root, "")

        assert tree_string == "N0(\n\tN2(\n\tN3(),N4()))"
