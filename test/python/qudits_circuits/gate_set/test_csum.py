from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCSum(TestCase):
    @staticmethod
    def test___array__():
        # all 22 csum

        circuit_22 = QuantumCircuit(2, [2, 2], 0)
        csum = circuit_22.csum([0, 1])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_22 = QuantumCircuit(2, [2, 2], 0)
        csum = circuit_22.csum([1, 0])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # All 33 csum
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        csum = circuit_33.csum([0, 1])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        csum = circuit_33.csum([1, 0])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # all 23 csum

        circuit_23 = QuantumCircuit(2, [2, 3], 0)
        csum = circuit_23.csum([0, 1])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_23 = QuantumCircuit(2, [2, 3], 0)
        csum = circuit_23.csum([1, 0])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # all 32 csum

        circuit_32 = QuantumCircuit(2, [3, 2], 0)
        csum = circuit_32.csum([0, 1])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_32 = QuantumCircuit(2, [3, 2], 0)
        csum = circuit_32.csum([1, 0])
        matrix = csum.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0],
            ]),
            matrix,
        )

        matrix_dag = csum.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )
