# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyAdaptiveDecomposition, PhyQrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _qft_matrix(dimension: int) -> NDArray[np.complex128]:
    levels = np.arange(dimension)
    return np.asarray(
        np.exp(2j * np.pi * np.outer(levels, levels) / dimension) / np.sqrt(dimension),
        dtype=np.complex128,
    )


class TestPhyLocAdaPass(TestCase):
    @staticmethod
    def test_transpile():
        dimension = 4
        nodes = list(range(dimension))
        initial_mapping = [2, 0, 3, 1]
        circuit = QuantumCircuit(1, [dimension], 0)
        circuit.cu_one(0, _qft_matrix(dimension))
        graph = LevelGraph(
            [(level, level + 1, {}) for level in range(dimension - 1)],
            nodes,
            initial_mapping,
            [0],
            0,
            circuit,
        )
        backend = MQTQuditProvider().get_backend("faketraps2six")
        backend.energy_level_graphs[0] = graph

        tree_sizes = []
        original_execute = PhyAdaptiveDecomposition.execute

        def execute_and_record(decomposition: PhyAdaptiveDecomposition):
            result = original_execute(decomposition)
            tree_sizes.append(decomposition.TREE.total_size)
            return result

        with patch.object(PhyAdaptiveDecomposition, "execute", execute_and_record):
            compiled = QuditCompiler.compile_O2(backend, circuit)

        assert tree_sizes[0] < 100
        assert compiled.mappings is not None
        actual = np.eye(dimension, dtype=np.complex128)
        for gate in compiled.instructions:
            actual = gate.to_matrix(identities=0) @ actual
        initial_permutation = np.eye(dimension)[:, initial_mapping]
        final_permutation = np.eye(dimension)[:, compiled.mappings[0]]
        assert np.allclose(initial_permutation.T @ actual @ final_permutation, _qft_matrix(dimension))


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
        assert len(matrices_decomposed) == 17
        assert v.verify()
