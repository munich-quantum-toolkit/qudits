from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestX(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(3, [2, 3, 5], 0)

    def test___array__(self):
        x_0 = self.circuit_23.x(0)
        matrix_0 = x_0.to_matrix(identities=0)
        assert np.allclose(np.array([[0, 1], [1, 0]]), matrix_0)

        matrix_0_dag = x_0.dag().to_matrix(identities=0)
        assert np.allclose(matrix_0_dag, matrix_0.conj().T)

        x_1 = self.circuit_23.x(1)
        matrix_1 = x_1.to_matrix(identities=0)
        assert np.allclose(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), matrix_1)

        matrix_1_dag = x_1.dag().to_matrix(identities=0)
        assert np.allclose(matrix_1_dag, matrix_1.conj().T)
