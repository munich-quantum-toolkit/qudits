from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


def omega_d(d):
    return np.e ** (2 * np.pi * 1j / d)


class TestZ(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)

    def test___array__(self):
        z_0 = self.circuit_23.z(0)
        matrix_0 = z_0.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0], [0, -1]]), matrix_0)

        z_1 = self.circuit_23.z(1)
        matrix_1 = z_1.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0, 0], [0, omega_d(3), 0], [0, 0, (omega_d(3) ** 2)]]), matrix_1)

    def test_validate_parameter(self):
        z = self.circuit_23.z(0)
        assert z.validate_parameter()
