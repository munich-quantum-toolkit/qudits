from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


def omega_s_d(d: int) -> complex:
    return np.e ** (2 * np.pi * 1j / d)


class TestS(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)
        self.circuit_4 = QuantumCircuit(1, [4], 0)

    def test___array__(self):
        s_0 = self.circuit_23.s(0)
        matrix_0 = s_0.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0], [0, 1j]]), matrix_0)

        matrix_1_dag = s_0.dag().to_matrix(identities=0)
        assert np.allclose(matrix_1_dag, matrix_0.conj().T)

        s_1 = self.circuit_23.s(1)
        matrix_1 = s_1.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0, 0], [0, omega_s_d(3), 0], [0, 0, 1]]), matrix_1)

        matrix_1_dag = s_1.dag().to_matrix(identities=0)
        assert np.allclose(matrix_1_dag, matrix_1.conj().T)
