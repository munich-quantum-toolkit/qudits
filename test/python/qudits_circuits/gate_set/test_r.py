from unittest import TestCase
import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestR(TestCase):
    def test___array__(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [1, 2, np.pi / 3, np.pi / 7])
        # R(np.pi / 3, np.pi / 7, 1, 2, 3)
        r1_test = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0.86602 + 0.j, -0.21694 - 0.45048j],
                            [0. + 0.j, 0.21694 - 0.45048j, 0.86602 + 0.j]])

        self.assertTrue(np.allclose(r.to_matrix(identities=0), r1_test))

        r1_test_dag = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0.86602 + 0.j, 0.21694 + 0.45048j],
                                [0. + 0.j, -0.21694 + 0.45048j, 0.86602 + 0.j]])

        self.assertTrue(np.allclose(r.dag().to_matrix(identities=0), r1_test_dag))

        circuit_4 = QuantumCircuit(1, [4], 0)
        r_2 = circuit_4.r(0, [0, 2, np.pi / 3, np.pi / 7])

        # R(np.pi / 3, np.pi / 7, 0, 2, 4)
        r_2_test = np.array([[0.8660254 + 0.j, 0. + 0.j, -0.21694187 - 0.45048443j, 0. + 0.j],
                             [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                             [0.21694187 - 0.45048443j, 0. + 0.j, 0.8660254 + 0.j, 0. + 0.j],
                             [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]])

        self.assertTrue(np.allclose(r_2.to_matrix(identities=0), r_2_test))

        r_2_test_dag = np.array([[0.8660254 + 0.j, 0. + 0.j, 0.21694187 + 0.45048443j, 0. + 0.j],
                                 [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                 [-0.21694187 + 0.45048443j, 0. + 0.j, 0.8660254 + 0.j, 0. + 0.j],
                                 [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]])

        self.assertTrue(np.allclose(r_2.dag().to_matrix(identities=0), r_2_test_dag))

    def test_regulate_theta(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [1, 2, 0.01 * np.pi, np.pi / 7])
        # R(0.01 * np.pi, np.pi / 7, 0, 2, 3)
        self.assertAlmostEqual(round(r.theta, 4), 12.5978)

    def test_levels_setter(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, 0.01 * np.pi, np.pi / 7])
        # R(0.01 * np.pi, np.pi / 7, 2, 0, 3)

        self.assertEqual(r.lev_a, 0)
        self.assertEqual(r.lev_b, 2)
        self.assertEqual(r.original_lev_a, 2)
        self.assertEqual(r.original_lev_b, 0)

    def test_cost(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, 0.01 * np.pi, np.pi / 7])
        self.assertEqual(round(r.cost, 4), 0.00160)

    def test_validate_parameter(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, np.pi, np.pi / 7])

        self.assertTrue(r.validate_parameter([2, 0, np.pi, np.pi / 7]))
        try:
            r.validate_parameter([3, 0, np.pi, np.pi / 7])
        except AssertionError:
            assert True
        try:
            r.validate_parameter([1, 3, np.pi, np.pi / 7])
        except AssertionError:
            assert True
