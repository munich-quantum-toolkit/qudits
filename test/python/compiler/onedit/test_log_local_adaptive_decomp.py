from __future__ import annotations

from unittest import TestCase


class TestLogLocAdaPass(TestCase):
    def test_transpile(self):
        pass


class TestLogAdaptiveDecomposition(TestCase):
    """def test_execute(self):
        dim = 3
        test_sample_edges = [(0, 4, {"delta_m": 0, "sensitivity": 1}),
                             (0, 3, {"delta_m": 1, "sensitivity": 3}),
                             (0, 2, {"delta_m": 1, "sensitivity": 3}),
                             (1, 4, {"delta_m": 0, "sensitivity": 1}),
                             (1, 3, {"delta_m": 1, "sensitivity": 3}),
                             (1, 2, {"delta_m": 1, "sensitivity": 3})
                             ]
        test_sample_nodes = [0, 1, 2, 3, 4]
        test_sample_nodes_map = [3, 2, 4, 1, 0]

        circuit_5 = QuantumCircuit(1, [5], 0)
        graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0], 0, circuit_5)

        Htest = circuit_5.h(0)
        graph_1.phase_storing_setup()

        QR = QrDecomp(Htest, graph_1, Z_prop=False, not_stand_alone=False)
        # gate, graph_orig, Z_prop=False, not_stand_alone=True

        decomp, _algorithmic_cost, _total_cost = QR.execute()

        ADA = LogAdaptiveDecomposition(Htest, graph_1, cost_limit=(1.1 * _algorithmic_cost, 1.1 * _total_cost),
                                       dimension=5, Z_prop=False)
        # gate, graph_orig, cost_limit=(0, 0), dimension=-1, Z_prop=False
        matrices_decomposed, best_cost, final_graph = ADA.execute()
        # ##############################################

        V = UnitaryVerifier(
                matrices_decomposed, Htest, [dim], test_sample_nodes, test_sample_nodes_map, graph_1.log_phy_map
        )
        # self.assertEqual(len(matrices_decomposed), 17)
        self.assertTrue(V.verify())

    def test_dfs(self):
        pass"""
