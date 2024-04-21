from __future__ import annotations

import operator
from functools import reduce

import numpy as np


class UnitaryVerifier:
    """
    Verifies unitary matrices.
    sequence is a list of numpy arrays
    target is a numpy array
    dimensions is list of ints, equals to the dimensions of the qudits involved in the target operation
    initial_map is a list representing the mapping of the logic states to the physical ones at the beginning of the computation
    final_map is a list representing the mapping of the logic states to the physical ones at the end of the computation
    """

    def __init__(self, sequence, target, dimensions, nodes=None, initial_map=None, final_map=None) -> None:
        self.decomposition = sequence
        self.target = target.copy()
        self.dimension = reduce(operator.mul, dimensions)

        if nodes is not None and initial_map is not None and final_map is not None:
            self.permutation_matrix_initial = self.get_perm_matrix(nodes, initial_map)
            self.permutation_matrix_final = self.get_perm_matrix(nodes, final_map)
            self.target = self.permutation_matrix_initial @ self.target
        else:
            self.permutation_matrix_initial = None
            self.permutation_matrix_final = None

    def get_perm_matrix(self, nodes, mapping):
        # sum ( |phy> <log| )
        perm = np.zeros((self.dimension, self.dimension))

        for i in range(self.dimension):
            a = [0 for i in range(self.dimension)]
            b = [0 for i in range(self.dimension)]
            a[nodes[i]] = 1
            b[mapping[i]] = 1
            narr = np.array(a)
            marr = np.array(b)
            perm = perm + np.outer(marr, narr)

        return perm

    def verify(self):
        target = self.target.copy()

        for rotation in self.decomposition:
            target = rotation @ target

        if self.permutation_matrix_final is not None:
            target = np.linalg.inv(self.permutation_matrix_final) @ target

        target = target / target[0][0]

        return (abs(target - np.identity(self.dimension, dtype="complex")) < 1e-4).all()
