from unittest import TestCase

import numpy as np
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


def omega_d(d):
    return np.e ** (2 * np.pi * 1j / d)


class TestZ(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(1, [2], 0)

    def test___array__(self):
        s_0 = self.circuit_23.s(0)
        matrix_0 = s_0.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0], [0, 1j]]), matrix_0)

    def test_validate_parameter(self):
        z = self.circuit_23.z(0)
        assert z.validate_parameter()
