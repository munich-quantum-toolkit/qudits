# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import unitary_group

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_adaptive_decomp import (
    LogAdaptiveDecomposition,
    LogLocAdaPass,
)
from mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_qr_decomp import QrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.core.dfs_tree import Node
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


@pytest.mark.parametrize("dimension", [4, 10])
@pytest.mark.parametrize("dense", [False, True], ids=["shift", "random-unitary"])
def test_adaptive_column_progress(dimension: int, dense: bool):
    unitary = (
        unitary_group(dimension).rvs(random_state=np.random.default_rng(42))
        if dense
        else np.roll(np.eye(dimension, dtype=np.complex128), 1, axis=1)
    )
    circuit = QuantumCircuit(1, [dimension], 0)
    target = circuit.cu_one(0, unitary)
    mapping = list(range(dimension))
    graph = LevelGraph([(0, level, {}) for level in range(1, dimension)], mapping, mapping, [0], 0, circuit)
    # The budget permits a complete elimination, without unused siblings.
    max_nodes = dimension * (dimension - 1) // 2 if dense else dimension - 1
    adaptive = LogAdaptiveDecomposition(target, graph, (np.inf, np.inf), dimension, max_nodes=max_nodes)

    decomposition, best_cost, _ = adaptive.execute()

    assert np.isfinite(best_cost[1])
    assert adaptive.TREE.total_size <= max_nodes + 1
    assert UnitaryVerifier(decomposition, target, [dimension]).verify()


@pytest.mark.parametrize("dimension", range(6, 11))
def test_adaptive_dense_path_graph(dimension: int):
    rng = np.random.default_rng(427)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    unitary, _ = np.linalg.qr(matrix)
    circuit = QuantumCircuit(1, [dimension], 0)
    target = circuit.cu_one(0, unitary)
    graph = LevelGraph(
        [(level, level + 1, {}) for level in range(dimension - 1)],
        list(range(dimension)),
        rng.permutation(dimension).tolist(),
        [0],
        0,
        circuit,
    )
    backend = MQTQuditProvider().get_backend("faketraps2six")
    backend.energy_level_graphs[0] = graph
    tree_sizes = []
    original_execute = LogAdaptiveDecomposition.execute
    original_add = Node.add

    def execute_and_record(decomposition: LogAdaptiveDecomposition):
        result = original_execute(decomposition)
        tree_sizes.append(decomposition.TREE.total_size)
        return result

    def add_with_limit(node: Node, new_key: int, *args):
        assert new_key <= 1000, "Adaptive search exceeded its node budget"
        return original_add(node, new_key, *args)

    with (
        patch.object(LogAdaptiveDecomposition, "execute", execute_and_record),
        patch.object(Node, "add", add_with_limit),
    ):
        decomposition = LogLocAdaPass(backend).transpile_gate(target)

    assert 1 < tree_sizes[0] <= 1001
    assert UnitaryVerifier(decomposition, target, [dimension]).verify()


class TestLogLocQRPass(TestCase):
    @staticmethod
    def test_adaptive_qr_fallback():
        dimension = 4
        levels = np.arange(dimension)
        qft = np.exp(2j * np.pi * np.outer(levels, levels) / dimension) / np.sqrt(dimension)
        circuit = QuantumCircuit(1, [dimension], 0)
        target = circuit.cu_one(0, qft)
        graph = LevelGraph(
            [(level, level + 1, {}) for level in range(dimension - 1)],
            list(range(dimension)),
            list(range(dimension)),
            [0],
            0,
            circuit,
        )
        backend = MQTQuditProvider().get_backend("faketraps2six")
        backend.energy_level_graphs[0] = graph

        searches = []

        def bounded_search(*args, **kwargs):
            search = LogAdaptiveDecomposition(*args, **kwargs, max_nodes=1)
            searches.append(search)
            return search

        with patch(
            "mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_adaptive_decomp.LogAdaptiveDecomposition",
            side_effect=bounded_search,
        ):
            decomposition = LogLocAdaPass(backend).transpile_gate(target)

        assert searches[0].TREE.total_size == 2
        assert not searches[0].TREE.root.finished
        assert decomposition
        assert UnitaryVerifier(decomposition, target, [dimension]).verify()


class TestQrDecomp(TestCase):
    @staticmethod
    def test_execute():
        dim = 3
        test_sample_edges = [
            (0, 2, {"delta_m": 0, "sensitivity": 1}),
            (1, 2, {"delta_m": 0, "sensitivity": 1}),
        ]
        test_sample_nodes = [0, 1, 2]
        test_sample_nodes_map = [0, 1, 2]

        circuit_3 = QuantumCircuit(1, [3], 0)
        graph_1 = LevelGraph(test_sample_edges, test_sample_nodes, test_sample_nodes_map, [0], 0, circuit_3)

        htest = circuit_3.h(0)

        qr = QrDecomp(htest, graph_1, z_prop=False, not_stand_alone=False)
        # gate, graph_orig, Z_prop=False, not_stand_alone=True

        decomp, _algorithmic_cost, _total_cost = qr.execute()

        v = UnitaryVerifier(decomp, htest, [dim], test_sample_nodes, test_sample_nodes_map, graph_1.log_phy_map)
        # sequence, target, dimensions, nodes=None, initial_map=None, final_map=None
        assert len(decomp) == 5
        assert v.verify()

        assert (decomp[0].lev_a, decomp[0].lev_b) == (1, 2)
        assert (decomp[1].lev_a, decomp[1].lev_b) == (0, 1)
        assert (decomp[2].lev_a, decomp[2].lev_b) == (1, 2)
        assert decomp[3].lev_a == 1
        assert decomp[4].lev_a == 2


class TestLogAdaptiveDecomposition(TestCase):
    @staticmethod
    def test_node_budget():
        dimension = 3
        circuit = QuantumCircuit(1, [dimension], 0)
        graph = LevelGraph([(0, 1, {}), (1, 2, {})], [0, 1, 2], [0, 1, 2], [0], 0, circuit)
        target = circuit.r(0, [0, 1, np.pi / 3, np.pi / 5])
        adaptive = LogAdaptiveDecomposition(target, graph, (np.inf, np.inf), dimension, max_nodes=0)
        decomposition, best_cost, _ = adaptive.execute()

        assert decomposition == []
        assert best_cost == (np.inf, np.inf)
        assert adaptive.TREE.total_size == 1

        adaptive = LogAdaptiveDecomposition(target, graph, (np.inf, np.inf), dimension, max_nodes=1)
        for _ in range(2):
            decomposition, best_cost, _ = adaptive.execute()
            assert np.isfinite(best_cost[1])
            assert adaptive.TREE.total_size == 2
            assert UnitaryVerifier(decomposition, target, [dimension]).verify()

        diagonal = circuit.cu_one(0, np.diag(np.exp(1j * np.array([0.2, -0.7, 0.4]))))
        adaptive = LogAdaptiveDecomposition(diagonal, graph, dimension=dimension, max_nodes=0)
        decomposition, best_cost, _ = adaptive.execute()

        assert best_cost == (0, 0)
        assert adaptive.TREE.total_size == 1
        assert UnitaryVerifier(decomposition, diagonal, [dimension]).verify()

        with pytest.raises(ValueError, match="max_nodes"):
            LogAdaptiveDecomposition(target, graph, dimension=dimension, max_nodes=-1)

    @staticmethod
    def test_execute_path_graph():
        dimension = 4
        nodes = list(range(dimension))
        levels = np.arange(dimension)
        qft = np.exp(2j * np.pi * np.outer(levels, levels) / dimension) / np.sqrt(dimension)
        circuit = QuantumCircuit(1, [dimension], 0)
        graph = LevelGraph(
            [(level, level + 1, {}) for level in range(dimension - 1)],
            nodes,
            nodes,
            [0],
            0,
            circuit,
        )
        target = circuit.cu_one(0, qft)

        _, algorithmic_cost, total_cost = QrDecomp(target, graph).execute()
        adaptive = LogAdaptiveDecomposition(target, graph, (algorithmic_cost, total_cost), dimension)
        decomposition, _, _ = adaptive.execute()

        assert adaptive.TREE.total_size < 100
        assert UnitaryVerifier(decomposition, target, [dimension]).verify()
