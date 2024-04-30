from __future__ import annotations

from unittest import TestCase


class TestLogLocAdaPass(TestCase):
    def test_transpile(self):
        pass


class TestLogAdaptiveDecomposition(TestCase):

    def test_execute(self):
        pass
        # dim = 5
        # test_sample_edges = [(0, 4, {"delta_m": 0, "sensitivity": 1}),
        #                      (0, 3, {"delta_m": 1, "sensitivity": 3}),
        #                      (0, 2, {"delta_m": 1, "sensitivity": 3}),
        #                      (1, 4, {"delta_m": 0, "sensitivity": 1}),
        #                      (1, 3, {"delta_m": 1, "sensitivity": 3}),
        #                      (1, 2, {"delta_m": 1, "sensitivity": 3})
        #                      ]
        # test_sample_nodes = [0, 1, 2, 3, 4]
        # test_sample_nodes_map = [3, 2, 4, 1, 0]
        #
        # graph_1 = level_Graph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0])
        # graph_1.phase_storing_setup()
        #
        # Htest = H(dim)
        #
        # #-----------------
        # QR = QR_decomp( Htest, graph_1, Z_prop = False, not_stand_alone = True )
        #
        # decomp_qr, algorithmic_cost_qr, total_cost_qr = QR.execute()
        # #----------------
        #
        # ADA = Adaptive_decomposition(Htest, graph_1, cost_limit=(1.1 * algorithmic_cost_qr, 1.1 * total_cost_qr), dimension=dim, Z_prop=False)
        #
        # matrices_decomposed, best_cost, final_graph = ADA.execute()
        # ##############################################
        #
        #
        # V = Verifier(matrices_decomposed, Htest, test_sample_nodes, test_sample_nodes_map, graph_1.lpmap, dim)
        # self.assertEqual( len(matrices_decomposed), 17)
        # self.assertTrue(V.verify())

    def test_dfs(self):
        pass
