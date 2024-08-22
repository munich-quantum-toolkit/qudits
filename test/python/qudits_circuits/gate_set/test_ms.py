from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestMS(TestCase):
    def test___array__(self):
        circuit = QuantumCircuit(1, [3, 3], 0)
        ms = circuit.ms([0, 1], [np.pi / 2]).to_matrix()

        matrix = np.array([
            [
                0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                -0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.5 - 0.5j,
                0.0 + 0.0j,
                -0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.92387953 - 0.38268343j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                -0.5 - 0.5j,
                0.0 + 0.0j,
                0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                -0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 - 0.5j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.92387953 - 0.38268343j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.92387953 - 0.38268343j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.92387953 - 0.38268343j,
                0.0 + 0.0j,
            ],
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                1.0 + 0.0j,
            ],
        ])

        assert np.allclose(ms, matrix)

        ms_dag = circuit.ms([0, 1], [np.pi / 2]).dag().to_matrix()

        assert np.allclose(ms_dag, matrix.conj().T)

    def test_validate_parameter(self):
        circuit_4 = QuantumCircuit(1, [4, 4], 0)
        ms = circuit_4.ms([0, 1], [np.pi / 2])
        assert ms.validate_parameter([np.pi / 2])
        try:
            ms.validate_parameter([4 * np.pi])
        except AssertionError:
            assert True
