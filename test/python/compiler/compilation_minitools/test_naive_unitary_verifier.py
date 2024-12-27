from __future__ import annotations

import operator
from functools import reduce
from typing import cast
from unittest import TestCase

import numpy as np
import pytest

from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import (
    mini_phy_unitary_sim,
    mini_sim,
    mini_unitary_sim,
    naive_phy_sim,
)
from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import (
    random_sparse_unitary,
    random_unitary_matrix,
)
from mqt.qudits.core import LevelGraph
from mqt.qudits.quantum_circuit import QuantumCircuit

rng = np.random.default_rng()


def choice(x: list[bool]) -> bool:
    return cast("bool", rng.choice(x, size=1)[0])


class TestUnitaryVerifier(TestCase):
    def setUp(self) -> None:
        edges = [
            (0, 3, {"delta_m": 1, "sensitivity": 5}),
            (0, 4, {"delta_m": 0, "sensitivity": 3}),
            (1, 4, {"delta_m": 0, "sensitivity": 3}),
            (1, 2, {"delta_m": 1, "sensitivity": 5}),
        ]
        nodes = [0, 1, 2, 3, 4]
        nodes_map = [0, 2, 1, 4, 3]
        self.circuit = QuantumCircuit(1, [5, 2, 3], 0)
        self.graph = LevelGraph(edges, nodes, nodes_map, [0], 0, self.circuit)

    def test_verify(self):
        dimension = 2

        sequence = [
            self.circuit.cu_one(1, np.identity(dimension, dtype="complex")),
            self.circuit.h(1),
            self.circuit.h(1),
        ]
        target = self.circuit.cu_one(1, np.identity(dimension, dtype="complex"))

        nodes = [0, 1]
        initial_map = [0, 1]
        final_map = [0, 1]

        v1 = UnitaryVerifier(sequence, target, [dimension], nodes, initial_map, final_map)

        assert v1.verify()

        ##################################################################

        dimension = 3

        nodes_3 = [0, 1, 2]
        initial_map_3 = [0, 1, 2]
        final_map_3 = [0, 2, 1]

        sequence_3 = [
            self.circuit.cu_one(2, np.identity(dimension, dtype="complex")),
            self.circuit.h(2),
            self.circuit.x(2),
            self.circuit.x(2),
            self.circuit.x(2),
        ]

        target_3 = self.circuit.h(2)

        v1 = UnitaryVerifier(sequence_3, target_3, [dimension], nodes_3, initial_map_3, final_map_3)

        assert v1.verify()

    @staticmethod
    def test_mini_sim():
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        size = reduce(operator.mul, circuit.dimensions)
        state = np.array(size * [0.0 + 0.0j])
        state[0] = 1.0 + 0.0j
        for gate in circuit.instructions:
            state = gate.to_matrix(identities=2) @ state

        assert np.allclose(state, mini_sim(circuit))

    @staticmethod
    def test_mini_unitary_sim():
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        size = reduce(operator.mul, circuit.dimensions)
        id_mat = np.identity(size)
        for gate in circuit.instructions:
            id_mat = gate.to_matrix(identities=2) @ id_mat

        assert np.allclose(id_mat, mini_unitary_sim(circuit))

    @staticmethod
    def test_naive_phy_sim():
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        size = reduce(operator.mul, circuit.dimensions)
        state = np.array(size * [0.0 + 0.0j])
        state[0] = 1.0 + 0.0j
        for gate in circuit.instructions:
            state = gate.to_matrix(identities=2) @ state

        with pytest.raises(AssertionError):
            assert np.allclose(state, naive_phy_sim(circuit))

        circuit.set_initial_mappings([[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
        circuit.set_final_mappings([[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
        phstate = naive_phy_sim(circuit)
        assert np.allclose(state, phstate)

    @staticmethod
    def test_mini_phy_unitary_sim():
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        size = reduce(operator.mul, circuit.dimensions)
        id_mat = np.identity(size)
        for gate in circuit.instructions:
            id_mat = gate.to_matrix(identities=2) @ id_mat

        with pytest.raises(AssertionError):
            assert np.allclose(id_mat, mini_phy_unitary_sim(circuit))

        circuit.set_initial_mappings([[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
        circuit.set_final_mappings([[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]])
        phstate = mini_phy_unitary_sim(circuit)
        assert np.allclose(id_mat, phstate)
