from __future__ import annotations

from unittest import TestCase

from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestLevelGraph(TestCase):
    def setUp(self) -> None:
        test_sample_edges = [
            (0, 4, {"delta_m": 0, "sensitivity": 1}),
            (0, 3, {"delta_m": 1, "sensitivity": 3}),
            (0, 2, {"delta_m": 1, "sensitivity": 3}),
            (1, 4, {"delta_m": 0, "sensitivity": 1}),
            (1, 3, {"delta_m": 1, "sensitivity": 3}),
            (1, 2, {"delta_m": 1, "sensitivity": 3}),
        ]
        test_sample_nodes = [0, 1, 2, 3, 4]
        test_sample_nodes_map = [3, 2, 4, 1, 0]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        self.graph_1.phase_storing_setup()

    def test_phase_storing_setup(self):
        self.graph_1.phase_storing_setup()
        for i in range(len(self.graph_1.nodes)):
            assert self.graph_1.nodes[i]["phase_storage"] == 0.0

    def test_distance_nodes(self):
        assert self.graph_1.distance_nodes(2, 3) == 2
        assert self.graph_1.distance_nodes(2, 4) == 2

    def test_distance_nodes_pi_pulses_fixed_ancilla(self):
        assert self.graph_1.distance_nodes_pi_pulses_fixed_ancilla(0, 1) == 1
        assert self.graph_1.distance_nodes_pi_pulses_fixed_ancilla(2, 4) == 1

    def test_logic_physical_map(self):
        assert self.graph_1.log_phy_map == [3, 2, 4, 1, 0]

    def test_define__states(self):
        test_sample_edges = [
            (0, 4, {"delta_m": 0, "sensitivity": 1}),
            (0, 3, {"delta_m": 1, "sensitivity": 3}),
            (0, 2, {"delta_m": 1, "sensitivity": 3}),
            (1, 4, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_nodes = [0, 1, 2, 3, 4]
        test_sample_nodes_map = [3, 2, 4, 1, 0]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        assert self.graph_1.fst_inode == 0
        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [1])
        assert self.graph_1.fst_inode == 1

    def test_update_list(self):
        graph_1_list = [(0, 5), (0, 4), (0, 3), (0, 2), (1, 5), (1, 4), (1, 3), (1, 2)]
        assert self.graph_1.update_list(graph_1_list, 0, 5) != [
            (6, 0),
            (5, 4),
            (5, 3),
            (5, 2),
            (1, 0),
            (1, 4),
            (1, 3),
            (1, 2),
        ]
        assert self.graph_1.update_list(graph_1_list, 0, 5) == [
            (5, 0),
            (5, 4),
            (5, 3),
            (5, 2),
            (1, 0),
            (1, 4),
            (1, 3),
            (1, 2),
        ]

    def test_deep_copy_func(self):
        graph_1_list = [(0, 5), (0, 4), (0, 3), (0, 2), (1, 5), (1, 4), (1, 3), (1, 2)]
        new_list = self.graph_1.deep_copy_func(graph_1_list)
        assert new_list == graph_1_list

    def test_index(self):
        graph_1_list = [(0, 5), (0, 4), (0, 3), (0, 2), (1, 5), (1, 4), (1, 3), (1, 2)]
        i = self.graph_1.index(graph_1_list, 0)
        assert i == 0

    def test_swap_node_attributes(self):
        test_sample_edges = [
            (0, 2, {"delta_m": 0, "sensitivity": 1}),
            (1, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_nodes = [0, 1, 2]
        test_sample_nodes_map = [1, 0, 2]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        self.graph_1.phase_storing_setup()
        self.graph_1.nodes[0]["phase_storage"] = 0.0
        self.graph_1.nodes[1]["phase_storage"] = 1.0
        self.graph_1.nodes[2]["phase_storage"] = 2.0

        liste = self.graph_1.swap_node_attributes(0, 1)
        assert liste[0][1]["phase_storage"] == 1.0
        assert liste[1][1]["phase_storage"] == 0.0

    def test_swap_node_attr_simple(self):
        test_sample_edges = [
            (0, 2, {"delta_m": 0, "sensitivity": 1}),
            (1, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_nodes = [0, 1, 2]
        test_sample_nodes_map = [1, 0, 2]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        self.graph_1.phase_storing_setup()
        self.graph_1.nodes[0]["phase_storage"] = 0.0
        self.graph_1.nodes[1]["phase_storage"] = 1.0
        self.graph_1.nodes[2]["phase_storage"] = 2.0
        self.graph_1.swap_node_attr_simple(0, 1)
        assert self.graph_1.nodes[0]["phase_storage"] == 1.0
        assert self.graph_1.nodes[1]["phase_storage"] == 0.0

    def test_swap_nodes(self):
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
        test_sample_nodes_map = [3, 2, 4, 1, 0, 5]

        self.graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        self.graph_1.phase_storing_setup()
        list_1 = [(5, 0), (5, 4), (5, 3), (5, 2), (1, 0), (1, 4), (1, 3), (1, 2)]
        list_2 = [(0, 5), (0, 4), (0, 3), (0, 2), (1, 5), (1, 4), (1, 3), (1, 2)]

        nodes = list(self.graph_1.nodes)
        list(self.graph_1.edges)

        temp1 = self.graph_1.swap_nodes(0, 5)
        temp2 = self.graph_1.swap_nodes(0, 1)

        assert list(temp1.nodes) == nodes
        assert list(temp2.nodes) == nodes

        edge1 = temp1.edges
        edge2 = temp2.edges
        bool1 = True
        bool2 = True

        for tup in list_1:
            tempbool = False
            for e in edge1:
                if tup[0] in e and tup[1] in e:
                    tempbool = True

            bool1 = bool1 and tempbool

        assert bool1

        for tup in list_2:
            tempbool = False
            for e in edge2:
                if tup[0] in e and tup[1] in e:
                    tempbool = True
            bool2 = bool2 and tempbool

        assert bool2

    def test_get_rz_gates(self):
        self.graph_1.set_circuit(QuantumCircuit(1, [6], 0))
        self.graph_1.set_qudits_index(0)
        self.graph_1.nodes[0]["phase_storage"] = 1.0
        self.graph_1.nodes[1]["phase_storage"] = 1.0
        rzs = self.graph_1.get_vrz_gates()
        assert len(rzs) == 2
        assert rzs[0].lev_a == 3
        assert rzs[1].lev_a == 2
        assert rzs[0].phi == 1
        assert rzs[1].phi == 1

    def test_get_node_sensitivity_cost(self):
        sensitivity = self.graph_1.get_node_sensitivity_cost(0)
        assert sensitivity == 7
        sensitivity = self.graph_1.get_node_sensitivity_cost(2)
        assert sensitivity == 6

    def test_get_edge_sensitivity(self):
        sensitivity = self.graph_1.get_edge_sensitivity(1, 2)
        assert sensitivity == 3
        sensitivity = self.graph_1.get_edge_sensitivity(0, 4)
        assert sensitivity == 1

    def test_fst_rnode(self):
        assert self.graph_1.fst_rnode == 1

    def test_fstinode(self):
        assert self.graph_1.fst_inode == 0

    def test_is_inode(self):
        assert self.graph_1.is_inode(0)
        assert not self.graph_1.is_inode(1)
        assert not self.graph_1.is_inode(2)
        assert not self.graph_1.is_inode(3)
        assert not self.graph_1.is_inode(4)

    def test_is_irnode(self):
        assert self.graph_1.is_irnode(1)
        assert self.graph_1.is_irnode(2)
        assert self.graph_1.is_irnode(3)
        assert self.graph_1.is_irnode(4)
        assert not self.graph_1.is_irnode(0)

    def test_lpmap(self):
        assert self.graph_1.log_phy_map == [3, 2, 4, 1, 0]
