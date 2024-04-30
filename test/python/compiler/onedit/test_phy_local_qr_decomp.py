from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyQrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestPhyLocQRPass(TestCase):
    def test_transpile(self):
        pass


class TestPhyQRDecomposition(TestCase):
    def test_execute(self):
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

        Htest = circuit_5.h(0)
        graph_1.phase_storing_setup()

        QR = PhyQrDecomp(Htest, graph_1, Z_prop=False, not_stand_alone=False)
        # gate, graph_orig, Z_prop=False, not_stand_alone=True

        decomp, _algorithmic_cost, _total_cost = QR.execute()

        # ##############################################

        V = UnitaryVerifier(decomp, Htest, [dim], test_sample_nodes, test_sample_nodes_map, graph_1.log_phy_map)
        assert len(decomp) == 30
        assert V.verify()
