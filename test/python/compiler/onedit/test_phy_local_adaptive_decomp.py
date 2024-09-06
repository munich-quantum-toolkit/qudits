from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyAdaptiveDecomposition, PhyQrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestPhyLocAdaPass(TestCase):
    def test_transpile(self):
        pass


class TestPhyAdaptiveDecomposition(TestCase):
    @staticmethod
    def test_execute():
        dim = 5
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

        circuit_5 = QuantumCircuit(1, [5], 0)
        graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0], 0, circuit_5)

        htest = circuit_5.h(0)
        graph_1.phase_storing_setup()

        qr = PhyQrDecomp(htest, graph_1, z_prop=False, not_stand_alone=False)
        # gate, graph_orig, Z_prop=False, not_stand_alone=True

        _decomp, _algorithmic_cost, _total_cost = qr.execute()

        v = UnitaryVerifier(_decomp, htest, [dim], test_sample_nodes, test_sample_nodes_map, test_sample_nodes_map)
        assert len(_decomp) == 30
        assert v.verify()

        ada = PhyAdaptiveDecomposition(
            htest, graph_1, cost_limit=(1.1 * _algorithmic_cost, 1.1 * _total_cost), dimension=5, z_prop=False
        )
        # gate, graph_orig, cost_limit=(0, 0), dimension=-1, Z_prop=False
        matrices_decomposed, _best_cost, final_graph = ada.execute()
        # ##############################################

        v = UnitaryVerifier(
            matrices_decomposed, htest, [dim], test_sample_nodes, test_sample_nodes_map, final_graph.log_phy_map
        )
        assert len(matrices_decomposed) == 17
        assert v.verify()

    def test_dfs(self):
        pass
