from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestRandU(TestCase):
    def is_unitary(self, matrix):
        conjugate_transpose = np.conjugate(matrix.T)
        product = np.dot(matrix, conjugate_transpose)
        identity_matrix = np.eye(matrix.shape[0])
        return np.allclose(product, identity_matrix)

    def test___array__(self):
        circuit = QuantumCircuit(2, [3, 7], 0)
        ru1 = circuit.randu([0, 1])
        ru2 = circuit.randu([0, 1])
        assert self.is_unitary(ru1.to_matrix())
        assert self.is_unitary(ru2.to_matrix())
        assert not np.allclose(ru1.to_matrix(), ru2.to_matrix())

    def test_validate_parameter(self):
        circuit = QuantumCircuit(1, [3], 0)
        ru = circuit.randu([0])
        assert ru.validate_parameter()
