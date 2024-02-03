from unittest import TestCase

import numpy as np

from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


class TestX(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)

    def test___array__(self):
        x_0 = self.circuit_23.x(0)
        matrix_0 = x_0.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[0, 1,],
                                              [1, 0]]), matrix_0))

        x_1 = self.circuit_23.x(1)
        matrix_1 = x_1.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[0, 1, 0],
                                              [0, 0, 1],
                                              [1, 0, 0]]), matrix_1))

    def test_validate_parameter(self):
        x = self.circuit_23.x(0)
        self.assertTrue(x.validate_parameter())
