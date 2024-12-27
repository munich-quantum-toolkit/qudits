#!/usr/bin/env python3
from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.quantum_circuit.gates import R, Rz, VirtRz


def permute_according_to_mapping(circuit: QuantumCircuit, mappings: list[list[int]]) -> NDArray:
    lines = list(range(circuit.num_qudits))
    dimensions = circuit.dimensions
    permutation = np.eye(dimensions[0])[:, mappings[0]]
    for line in lines[1:]:
        permutation = np.kron(permutation, np.eye(dimensions[line])[:, mappings[line]])
    return permutation


def mini_unitary_sim(circuit: QuantumCircuit) -> NDArray[np.complex128, np.complex128]:
    size = reduce(operator.mul, circuit.dimensions)
    id_mat = np.identity(size)
    for gate in circuit.instructions:
        id_mat = gate.to_matrix(identities=2) @ id_mat
    return id_mat


def mini_sim(circuit: QuantumCircuit) -> NDArray[np.complex128]:
    size = reduce(operator.mul, circuit.dimensions)
    state = np.array(size * [0.0 + 0.0j])
    state[0] = 1.0 + 0.0j
    for gate in circuit.instructions:
        state = gate.to_matrix(identities=2) @ state
    return state


def mini_phy_unitary_sim(circuit: QuantumCircuit) -> NDArray[np.complex128, np.complex128]:
    assert circuit.final_mappings is not None
    assert circuit.initial_mappings is not None

    dimensions = circuit.dimensions
    size = reduce(operator.mul, dimensions)
    id_mat = np.identity(size)

    final_permutation = permute_according_to_mapping(circuit, circuit.final_mappings)
    init_permutation = permute_according_to_mapping(circuit, circuit.initial_mappings)

    id_mat = init_permutation @ id_mat
    for gate in circuit.instructions:
        id_mat = gate.to_matrix(identities=2) @ id_mat

    return final_permutation.T @ id_mat


def naive_phy_sim(circuit: QuantumCircuit) -> NDArray[np.complex128]:
    assert circuit.final_mappings is not None
    assert circuit.initial_mappings is not None

    dimensions = circuit.dimensions
    state = np.array(np.prod(dimensions) * [0.0 + 0.0j])
    state[0] = 1.0 + 0.0j

    final_permutation = permute_according_to_mapping(circuit, circuit.final_mappings)
    init_permutation = permute_according_to_mapping(circuit, circuit.initial_mappings)

    state = init_permutation @ state

    for gate in circuit.instructions:
        state = gate.to_matrix(identities=2) @ state

    return final_permutation.T @ state


class UnitaryVerifier:
    """Verifies unitary matrices.

    sequence is a list of numpy arrays
    target is a numpy array
    dimensions is list of ints, equals to the dimensions of the qudits involved in the target operation
    initial_map is a list representing the mapping of the logic states
    to the physical ones at the beginning of the computation
    final_map is a list representing the mapping of the logic states to the physical ones at the end of the computation.
    """

    def __init__(
        self,
        sequence: Sequence[Gate | R | Rz | VirtRz],
        target: Gate,
        dimensions: list[int],
        nodes: list[int] | None = None,
        initial_map: list[int] | None = None,
        final_map: list[int] | None = None,
    ) -> None:
        self.decomposition = sequence
        self.target = target.to_matrix().copy()
        self.dimension = reduce(operator.mul, dimensions)

        if nodes is not None and initial_map is not None and final_map is not None:
            self.permutation_matrix_initial = self.get_perm_matrix(nodes, initial_map)
            self.permutation_matrix_final = self.get_perm_matrix(nodes, final_map)
            self.target = self.permutation_matrix_initial @ self.target
        else:
            self.permutation_matrix_initial = None
            self.permutation_matrix_final = None

    def get_perm_matrix(self, nodes: list[int], mapping: list[int]) -> NDArray:
        # sum ( |phy> <log| )
        perm = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):
            a = [0 for i in range(self.dimension)]
            b = [0 for i in range(self.dimension)]
            a[nodes[i]] = 1
            b[mapping[i]] = 1
            narr = np.array(a)
            marr = np.array(b)
            perm += np.outer(marr, narr)

        return perm

    def verify(self) -> bool:
        target = self.target.copy()

        for rotation in self.decomposition:
            target = rotation.to_matrix(identities=0) @ target
            target.round(3)

        if self.permutation_matrix_final is not None:
            target = np.linalg.inv(self.permutation_matrix_final) @ target
            target.round(3)

        target /= target[0][0]
        target.round(3)

        return bool((abs(target - np.identity(self.dimension, dtype="complex")) < 1e-4).all())
