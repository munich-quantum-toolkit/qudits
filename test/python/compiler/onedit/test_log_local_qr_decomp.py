from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_qr_decomp import QrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestLogLocQRPass(TestCase):
    def test_transpile(self):
        pass


class TestQrDecomp(TestCase):
    def test_execute(self):
        # DIM 3
        dim = 3
        test_sample_edges = [
            (0, 2, {"delta_m": 0, "sensitivity": 1}),
            (1, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_edges = [(0, 2, {"delta_m": 0, "sensitivity": 1}),
                             (1, 2, {"delta_m": 0, "sensitivity": 1}),
                             ]
        test_sample_nodes = [0, 1, 2]
        test_sample_nodes_map = [0, 1, 2]

        circuit_3 = QuantumCircuit(1, [3], 0)
        graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0], 0, circuit_3)

        Htest = circuit_3.h(0)

        QR = QrDecomp(Htest, graph_1, Z_prop=False, not_stand_alone=False)
        # gate, graph_orig, Z_prop=False, not_stand_alone=True

        decomp, _algorithmic_cost, _total_cost = QR.execute()
        decomp, algorithmic_cost, total_cost = QR.execute()

        V = UnitaryVerifier(
            decomp, Htest.to_matrix(identities=0), [dim], test_sample_nodes, test_sample_nodes_map, graph_1.log_phy_map
        )
        V = UnitaryVerifier(decomp, Htest.to_matrix(identities=0), [dim],
                            test_sample_nodes, test_sample_nodes_map, graph_1.log_phy_map)
        # sequence, target, dimensions, nodes=None, initial_map=None, final_map=None
        assert len(decomp) == 5
        assert V.verify()
        self.assertEqual(len(decomp), 5)
        self.assertTrue(V.verify())

        assert (decomp[0].lev_a, decomp[0].lev_b) == (1, 2)
        assert (decomp[1].lev_a, decomp[1].lev_b) == (0, 1)
        assert (decomp[2].lev_a, decomp[2].lev_b) == (1, 2)
        assert decomp[3].lev_a == 1
        assert decomp[4].lev_a == 2
        self.assertEqual((decomp[0].lev_a, decomp[0].lev_b), (1, 2))
        self.assertEqual((decomp[1].lev_a, decomp[1].lev_b), (0, 1))
        self.assertEqual((decomp[2].lev_a, decomp[2].lev_b), (1, 2))
        self.assertEqual(decomp[3].lev_a, 1)
        self.assertEqual(decomp[4].lev_a, 2)
