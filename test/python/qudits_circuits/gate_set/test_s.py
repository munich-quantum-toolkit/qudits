from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


def omega_s_d(d):
    return np.e ** (2 * np.pi * 1j / d)


class TestS(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)
        self.circuit_4 = QuantumCircuit(1, [4], 0)

    def test___array__(self):
        s_0 = self.circuit_23.s(0)
        matrix_0 = s_0.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0], [0, omega_s_d(2)]]), matrix_0)

        s_1 = self.circuit_23.s(1)
        matrix_1 = s_1.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0, 0], [0, omega_s_d(3), 0], [0, 0, 1]]), matrix_1)

    def test_validate_parameter(self):
        s = self.circuit_23.s(0)
        assert s.validate_parameter()

        s = self.circuit_23.s(1)
        assert s.validate_parameter()

        try:
            self.circuit_4.s(4)
        except Exception:
            assert True
