from unittest import TestCase

import numpy as np

from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


class TestCEx(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)
        self.circuit_32 = QuantumCircuit(2, [3, 2], 0)

    def test___array__(self):
        cx_23_01 = self.circuit_23.cx([0, 1])
        cx_23_10 = self.circuit_23.cx([1, 0])
        matrix_23_01 = cx_23_01.to_matrix(identities=0)
        matrix_23_10 = cx_23_10.to_matrix(identities=0)

        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, -1j, 0],
                                              [0, 0, 0, -1j, 0, 0],
                                              [0, 0, 0, 0, 0, 1]]), matrix_23_01))
        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 1]]), matrix_23_10))

        cx_23_01_p = self.circuit_23.cx([0, 1])
        cx_23_10_p = self.circuit_23.cx([0, 1])
        matrix_23_01_p = cx_23_01_p.to_matrix(identities=0)
        matrix_23_10_p = cx_23_10_p.to_matrix(identities=0)

        self.assertTrue(np.allclose(np.array([[0, 1, ],
                                              [1, 0]]), matrix_23_01_p))
        self.assertTrue(np.allclose(np.array([[0, 1, ],
                                              [1, 0]]), matrix_23_10_p))

    def test_validate_parameter(self):
        pass
