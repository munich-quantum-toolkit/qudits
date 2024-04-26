from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestLS(TestCase):
    def test___array__(self):
        circuit = QuantumCircuit(1, [3, 3], 0)
        ls = circuit.ls([0, 1], [np.pi / 2]).to_matrix()

        matrix = np.array([
            [-0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, -0. - 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
            [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, -0. - 1.j]
        ])

        self.assertTrue(np.allclose(ls, matrix))

        ls_dag = circuit.ls([0, 1], [np.pi / 2]).dag().to_matrix()

        self.assertTrue(np.allclose(ls_dag, matrix.conj().T))

    def test_validate_parameter(self):
        circuit_4 = QuantumCircuit(1, [4, 4], 0)
        ls = circuit_4.ls([0, 1], [np.pi / 2])
        self.assertTrue(ls.validate_parameter([np.pi / 2]))
        try:
            ls.validate_parameter([4 * np.pi])
        except AssertionError:
            assert True
