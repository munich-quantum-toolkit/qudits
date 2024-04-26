from unittest import TestCase
import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestRh(TestCase):
    def test___array__(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [1, 2])

        rh_test = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0 + 0.70711j, 0 + 0.70711j],
                            [0. + 0.j, 0 + 0.70711j, 0 - 0.70711j]])

        self.assertTrue(np.allclose(rh.to_matrix(identities=0), rh_test))
        rh_test_dag = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0 - 0.70711j, 0 - 0.70711j],
                                [0. + 0.j, 0 - 0.70711j, 0 + 0.70711j]])

        self.assertTrue(np.allclose(rh.dag().to_matrix(identities=0), rh_test_dag))

        circuit_4 = QuantumCircuit(1, [4], 0)
        rh = circuit_4.rh(0, [1, 2])

        rh_test = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0 + 0.70711j, 0 + 0.70711j, 0. + 0.j],
                            [0. + 0j, 0 + 0.70711j, 0 - 0.70711j, 0. + 0.j],
                            [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]
                            ]
                           )

        self.assertTrue(np.allclose(rh.to_matrix(identities=0), rh_test))

        rh_test_dag = np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0 - 0.70711j, 0 - 0.70711j, 0. + 0.j],
                                [0. + 0j, 0 - 0.70711j, 0 + 0.70711j, 0. + 0.j],
                                [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]
                                ])

        self.assertTrue(np.allclose(rh.dag().to_matrix(identities=0), rh_test_dag))

    def test_levels_setter(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [2, 0])
        self.assertEqual(rh.lev_a, 0)
        self.assertEqual(rh.lev_b, 2)
        self.assertEqual(rh.original_lev_a, 2)
        self.assertEqual(rh.original_lev_b, 0)

    def test_validate_parameter(self):
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [2, 0])

        self.assertTrue(rh.validate_parameter([2, 0]))
        try:
            rh.validate_parameter([3, 0])
        except AssertionError:
            assert True
        try:
            rh.validate_parameter([1, 3])
        except AssertionError:
            assert True
