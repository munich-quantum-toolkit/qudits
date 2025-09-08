from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCEx(TestCase):
    @staticmethod
    def test___array__():
        # Mve around the control and angle

        # control on 2
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx([0, 1], [0, 1, 2, 0.0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, -1j, 0],
                [0, 0, 0, 0, 0, 0, -1j, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # control on 2 but swap 1 and 2
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx([0, 1], [1, 2, 2, 0.0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, -1j],
                [0, 0, 0, 0, 0, 0, 0, -1j, 0],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # control on 2 but swap 1 and 2, change agle
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx([0, 1], [1, 2, 2, np.pi / 6])
        matrix = cx.to_matrix(identities=0)
        ang = np.pi / 6
        val1 = -1j * np.cos(ang) - np.sin(ang)
        val2 = -1j * np.cos(ang) + np.sin(ang)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, val1],
                [0, 0, 0, 0, 0, 0, 0, val2, 0],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )
        # all 22 cx

        circuit_22 = QuantumCircuit(2, [2, 2], 0)
        cx = circuit_22.cx([0, 1])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, -1j, 0]]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_22 = QuantumCircuit(2, [2, 2], 0)
        cx = circuit_22.cx([1, 0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([[1, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1, 0], [0, -1j, 0, 0]]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # All 33 cx
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx([0, 1])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx([1, 0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1j, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # all 23 cx

        circuit_23 = QuantumCircuit(2, [2, 3], 0)
        cx = circuit_23.cx([0, 1])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, -1j, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_23 = QuantumCircuit(2, [2, 3], 0)
        cx = circuit_23.cx([1, 0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1j, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        # all 32 cx

        circuit_32 = QuantumCircuit(2, [3, 2], 0)
        cx = circuit_32.cx([0, 1])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, -1j, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

        circuit_32 = QuantumCircuit(2, [3, 2], 0)
        cx = circuit_32.cx([1, 0])
        matrix = cx.to_matrix(identities=0)
        assert np.allclose(
            np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, -1j, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, -1j, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]),
            matrix,
        )

        matrix_dag = cx.dag().to_matrix(identities=0)
        assert np.allclose(
            matrix.conj().T,
            matrix_dag,
        )

    @staticmethod
    def test_validate_parameter():
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cx = circuit_33.cx(
            [1, 0],
        )
        assert cx.validate_parameter([0, 1, 2, np.pi])

        try:
            cx.validate_parameter([0, 3, 2, np.pi])
        except AssertionError:
            assert True

        try:
            cx.validate_parameter([0, 1, 2, 4 * np.pi])
        except AssertionError:
            assert True
