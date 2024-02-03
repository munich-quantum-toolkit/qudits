from unittest import TestCase

import numpy as np

from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


class TestCSum(TestCase):

    def setUp(self):
        self.circuit_33 = QuantumCircuit(2, [3, 3], 0)
        self.circuit_23 = QuantumCircuit(2, [2, 3], 0)
        self.circuit_32 = QuantumCircuit(2, [3, 2], 0)

    def test___array__(self):
        csum = self.circuit_33.csum([0, 1])
        matrix = csum.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 1, 0]]), matrix))

        csum_10 = self.circuit_33.csum([1, 0])
        matrix_10 = csum_10.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1],
                                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 1, 0]]), matrix_10))

        csum_23_01 = self.circuit_23.csum([0, 1])
        matrix_23_01 = csum_23_01.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 1],
                                              [0, 0, 0, 1, 0, 0]]), matrix_23_01))

        csum_32_01 = self.circuit_32.csum([0, 1])
        matrix_32_01 = csum_32_01.to_matrix(identities=0)
        self.assertTrue(np.allclose(np.array([[1, 0, 0, 0, 0, 0],
                                              [0, 1, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 1, 0],
                                              [0, 0, 0, 0, 0, 1]]), matrix_32_01))

    def test_validate_parameter(self):
        csum = self.circuit_33.csum([0, 1])
        self.assertTrue(csum.validate_parameter())
