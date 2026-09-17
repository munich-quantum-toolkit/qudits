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
import pytest
from scipy.stats import unitary_group

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyAdaptiveDecomposition, PhyQrDecomp
from mqt.qudits.core import LevelGraph
from mqt.qudits.core.dfs_tree import Node
from mqt.qudits.quantum_circuit import QuantumCircuit, gates
from mqt.qudits.simulation import MQTQuditProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _qft_matrix(dimension: int) -> NDArray[np.complex128]:
    levels = np.arange(dimension)
    return np.asarray(
        np.exp(2j * np.pi * np.outer(levels, levels) / dimension) / np.sqrt(dimension),
        dtype=np.complex128,
    )


def _assert_compiled_unitary(
    compiled: QuantumCircuit, target: NDArray[np.complex128], initial_mapping: list[int]
) -> None:
    assert compiled.mappings is not None
    actual = np.eye(len(initial_mapping), dtype=np.complex128)
    for gate in compiled.instructions:
        actual = gate.to_matrix(identities=0) @ actual
    initial_permutation = np.eye(len(initial_mapping))[:, initial_mapping]
    final_permutation = np.eye(len(initial_mapping))[:, compiled.mappings[0]]
    assert np.allclose(final_permutation.T @ actual @ initial_permutation, target)


@pytest.mark.parametrize("dimension", range(4, 11))
@pytest.mark.parametrize("dense", [False, True], ids=["shift", "random-unitary"])
def test_compile_star_graph(dimension: int, dense: bool):
    unitary = (
        unitary_group(dimension).rvs(random_state=np.random.default_rng(42))
        if dense
        else np.roll(np.eye(dimension, dtype=np.complex128), -1, axis=1)
    )
    circuit = QuantumCircuit(1, [dimension], 0)
    circuit.cu_one(0, unitary)
    mapping = list(range(dimension))
    graph = LevelGraph([(0, level, {}) for level in range(1, dimension)], mapping, mapping, [0], 0, circuit)
    backend = MQTQuditProvider().get_backend("faketraps2six")
    backend.energy_level_graphs[0] = graph

    compiled = QuditCompiler().compile(backend, circuit, ["LocAdaPass"])

    max_rotations = dimension * (dimension - 2) if dense else dimension
    assert sum(isinstance(gate, gates.R) for gate in compiled.instructions) <= max_rotations
    _assert_compiled_unitary(compiled, unitary, mapping)


def test_compile_deep_search_uses_qr_fallback():
    dimension = 60
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    unitary, _ = np.linalg.qr(matrix)
    circuit = QuantumCircuit(1, [dimension], 0)
    circuit.cu_one(0, unitary)
    mapping = list(range(dimension))
    graph = LevelGraph([(0, level, {}) for level in range(1, dimension)], mapping, mapping, [0], 0, circuit)
    backend = MQTQuditProvider().get_backend("faketraps2six")
    backend.energy_level_graphs[0] = graph
    original_execute = PhyAdaptiveDecomposition.execute

    def execute_without_solution(search: PhyAdaptiveDecomposition):
        result = original_execute(search)
        assert result[0] == []
        assert result[1] == (np.inf, np.inf)
        assert search.TREE.total_size <= search.max_nodes + 1
        return result

    with patch.object(PhyAdaptiveDecomposition, "execute", execute_without_solution):
        compiled = QuditCompiler().compile(backend, circuit, ["LocAdaPass"])

    assert compiled.instructions
    _assert_compiled_unitary(compiled, unitary, mapping)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("last_max_nodes", [0, 1000])
def test_compile_propagated_phases(dimension: int, last_max_nodes: int):
    circuit = QuantumCircuit(1, [dimension], 0)
    if dimension == 2:
        circuit.cu_one(0, np.diag([1, np.exp(1j * np.pi / 3)]))
        circuit.h(0)
        initial_mapping = [0, 1]
        edges = [(0, 1, {})]
    else:
        rng = np.random.default_rng(42)
        for _ in range(3):
            matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
            unitary, _ = np.linalg.qr(matrix)
            circuit.cu_one(0, unitary)
        circuit.instructions[-1].dag()
        initial_mapping = rng.permutation(dimension).tolist()
        edges = [(0, 1, {}), (1, 2, {})]
    target = np.eye(dimension, dtype=np.complex128)
    for gate in circuit.instructions:
        target = gate.to_matrix(identities=0) @ target
    original_daggers = [gate.dagger for gate in circuit.instructions]
    graph = LevelGraph(
        edges,
        list(range(dimension)),
        initial_mapping,
        [0],
        0,
        circuit,
    )
    backend = MQTQuditProvider().get_backend("faketraps2six")
    backend.energy_level_graphs[0] = graph
    searches = []

    def bounded_search(*args, **kwargs):
        max_nodes = last_max_nodes if len(searches) == len(circuit.instructions) - 1 else 1000
        search = PhyAdaptiveDecomposition(*args, **kwargs, max_nodes=max_nodes)
        searches.append(search)
        return search

    with patch(
        "mqt.qudits.compiler.onedit.mapping_aware_transpilation.phy_local_adaptive_decomp.PhyAdaptiveDecomposition",
        side_effect=bounded_search,
    ):
        compiled = QuditCompiler.compile_O2(backend, circuit)

    assert all(search.TREE.root.finished and search.phase_propagation for search in searches[:-1])
    assert searches[-1].TREE.root.finished == bool(last_max_nodes)
    assert [gate.dagger for gate in circuit.instructions] == original_daggers
    final_graph = backend.energy_level_graphs[0]
    assert all(final_graph.nodes[node]["phase_storage"] == 0 for node in final_graph)
    _assert_compiled_unitary(compiled, target, initial_mapping)


@pytest.mark.parametrize("dimension", range(6, 11))
@pytest.mark.parametrize("matrix_kind", ["random", "qft", "shift"])
@pytest.mark.parametrize("permute_mapping", [False, True])
def test_compile_path_graph(dimension: int, matrix_kind: str, permute_mapping: bool):
    rng = np.random.default_rng(427)
    matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    unitary, _ = np.linalg.qr(matrix)
    if matrix_kind == "qft":
        unitary = _qft_matrix(dimension)
    elif matrix_kind == "shift":
        unitary = np.roll(np.eye(dimension, dtype=np.complex128), -1, axis=1)
    initial_mapping = rng.permutation(dimension).tolist() if permute_mapping else list(range(dimension))
    circuit = QuantumCircuit(1, [dimension], 0)
    circuit.cu_one(0, unitary)
    graph = LevelGraph(
        [(level, level + 1, {}) for level in range(dimension - 1)],
        list(range(dimension)),
        initial_mapping,
        [0],
        0,
        circuit,
    )
    backend = MQTQuditProvider().get_backend("faketraps2six")
    backend.energy_level_graphs[0] = graph
    tree_sizes = []
    original_execute = PhyAdaptiveDecomposition.execute
    original_add = Node.add

    def execute_and_record(decomposition: PhyAdaptiveDecomposition):
        result = original_execute(decomposition)
        tree_sizes.append(decomposition.TREE.total_size)
        return result

    def add_with_limit(node: Node, new_key: int, *args):
        assert new_key <= 1000, "Adaptive search exceeded its node budget"
        return original_add(node, new_key, *args)

    with (
        patch.object(PhyAdaptiveDecomposition, "execute", execute_and_record),
        patch.object(Node, "add", add_with_limit),
    ):
        compiled = QuditCompiler.compile_O2(backend, circuit)

    assert 1 < tree_sizes[0] <= 1001
    _assert_compiled_unitary(compiled, unitary, initial_mapping)


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
        _assert_compiled_unitary(compiled, _qft_matrix(dimension), initial_mapping)

    @staticmethod
    def test_qr_fallback():
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

        searches = []

        def bounded_search(*args, **kwargs):
            search = PhyAdaptiveDecomposition(*args, **kwargs, max_nodes=1)
            searches.append(search)
            return search

        with patch(
            "mqt.qudits.compiler.onedit.mapping_aware_transpilation.phy_local_adaptive_decomp.PhyAdaptiveDecomposition",
            side_effect=bounded_search,
        ):
            compiled = QuditCompiler.compile_O2(backend, circuit)

        assert searches[0].TREE.total_size == 2
        assert not searches[0].TREE.root.finished
        assert compiled.instructions
        _assert_compiled_unitary(compiled, _qft_matrix(dimension), initial_mapping)


class TestPhyAdaptiveDecomposition(TestCase):
    @staticmethod
    def test_node_budget():
        dimension = 3
        nodes = list(range(dimension))
        mapping = [1, 0, 2]
        circuit = QuantumCircuit(1, [dimension], 0)
        graph = LevelGraph([(0, 1, {}), (1, 2, {})], nodes, mapping, [0], 0, circuit)
        target = circuit.r(0, [0, 1, np.pi / 3, np.pi / 5])
        adaptive = PhyAdaptiveDecomposition(target, graph, (np.inf, np.inf), dimension, max_nodes=0)
        decomposition, best_cost, _ = adaptive.execute()

        assert decomposition == []
        assert best_cost == (np.inf, np.inf)
        assert adaptive.TREE.total_size == 1

        adaptive = PhyAdaptiveDecomposition(target, graph, (np.inf, np.inf), dimension, max_nodes=1)
        for _ in range(2):
            decomposition, best_cost, final_graph = adaptive.execute()
            assert np.isfinite(best_cost[1])
            assert adaptive.TREE.total_size == 2
            assert UnitaryVerifier(decomposition, target, [dimension], nodes, mapping, final_graph.log_phy_map).verify()

        diagonal = circuit.cu_one(0, np.diag(np.exp(1j * np.array([0.2, -0.7, 0.4]))))
        adaptive = PhyAdaptiveDecomposition(diagonal, graph, dimension=dimension, max_nodes=0)
        decomposition, best_cost, final_graph = adaptive.execute()

        assert best_cost == (0, 0)
        assert adaptive.TREE.total_size == 1
        assert UnitaryVerifier(decomposition, diagonal, [dimension], nodes, mapping, final_graph.log_phy_map).verify()

        with pytest.raises(ValueError, match="max_nodes"):
            PhyAdaptiveDecomposition(target, graph, dimension=dimension, max_nodes=-1)

    @staticmethod
    def test_execute_preserves_later_column_branch():
        dimension = 4
        rng = np.random.default_rng(162)
        matrix = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        unitary, triangular = np.linalg.qr(matrix)
        unitary *= (np.diag(triangular) / np.abs(np.diag(triangular))).conj()

        mapping = [0, 3, 2, 1]
        circuit = QuantumCircuit(1, [dimension], 0)
        target = circuit.cu_one(0, unitary)
        graph = LevelGraph(
            [(0, 2, {}), (2, 1, {}), (2, 3, {})],
            list(range(dimension)),
            mapping,
            [0],
            0,
            circuit,
        )

        _, algorithmic_cost, total_cost = PhyQrDecomp(target, graph).execute()
        adaptive = PhyAdaptiveDecomposition(target, graph, (algorithmic_cost, total_cost), dimension)
        decomposition, best_cost, final_graph = adaptive.execute()

        verifier = UnitaryVerifier(
            decomposition, target, [dimension], list(range(dimension)), mapping, final_graph.log_phy_map
        )
        assert best_cost[1] < total_cost
        assert verifier.verify()

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
