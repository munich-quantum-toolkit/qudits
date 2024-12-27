from __future__ import annotations

from typing import cast
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.onedit import PhyLocAdaPass, ZPropagationOptPass
from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyAdaptiveDecomposition, PhyQrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


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

        decomp, algorithmic_cost, total_cost = qr.execute()

        v = UnitaryVerifier(decomp, htest, [dim], test_sample_nodes, test_sample_nodes_map, test_sample_nodes_map)
        assert len(decomp) == 30
        assert v.verify()

        ada = PhyAdaptiveDecomposition(
            htest, graph_1, cost_limit=(1.1 * algorithmic_cost, 1.1 * total_cost), dimension=5, z_prop=False
        )
        # gate, graph_orig, cost_limit=(0, 0), dimension=-1, Z_prop=False
        matrices_decomposed, _best_cost, final_graph = ada.execute()
        # ##############################################

        v = UnitaryVerifier(
            matrices_decomposed, htest, [dim], test_sample_nodes, test_sample_nodes_map, final_graph.log_phy_map
        )
        assert v.verify()

    def test_dfs(self):
        pass

    @staticmethod
    def test_execute_consecutive():
        dim = 3
        c = QuantumCircuit(1, [dim], 0)
        circuit_d = QuantumCircuit(1, [dim], 0)

        for _i in range(200):
            # r3 = circuit_d.h(0)
            r3 = circuit_d.cu_one(0, c.randu([0]).to_matrix())

        test_circ = circuit_d.copy()

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2six")

        inimap = backend_ion.energy_level_graphs[0].log_phy_map[:dim]

        phloc = PhyLocAdaPass(backend_ion)
        new_circuit = phloc.transpile(circuit_d)

        fmap = phloc.backend.energy_level_graphs[0].log_phy_map[:dim]

        v = UnitaryVerifier(new_circuit.instructions, r3, [dim], list(range(dim)), inimap, fmap)

        uni_l = mini_unitary_sim(circuit_d)
        uni = mini_unitary_sim(new_circuit)
        tpuni = uni @ v.get_perm_matrix(list(range(dim)), fmap)  # Pf
        tpuni = v.get_perm_matrix(list(range(dim)), inimap).T @ tpuni  # Pi dag
        assert np.allclose(tpuni, uni_l, rtol=1e-5, atol=1e-5)

        z_propagation_pass = ZPropagationOptPass(backend=backend_ion, back=False)
        new_transpiled_circuit = z_propagation_pass.transpile(new_circuit)
        mini_unitary_sim(new_transpiled_circuit).round(4)
        tpuni2 = uni @ v.get_perm_matrix(list(range(dim)), fmap)  # Pf
        tpuni2 = v.get_perm_matrix(list(range(dim)), inimap).T @ tpuni2  # Pi dag
        assert np.allclose(tpuni2, uni_l, rtol=1e-5, atol=1e-5)

        adapt_circ = test_circ.compileO1("faketraps2six", "adapt")
        u2a = mini_unitary_sim(adapt_circ)
        tpuni2a = u2a @ v.get_perm_matrix(list(range(dim)), cast("list[list[int]]", adapt_circ.final_mappings)[0])  # Pf
        tpuni2a = v.get_perm_matrix(list(range(dim)), inimap).T @ tpuni2a  # Pi dag
        assert np.allclose(tpuni2a, uni_l, rtol=1e-5, atol=1e-5)
