from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestNoiseY(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        ny = circuit_3.noisey(0, [1, 2])
        ny_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(ny.to_matrix(identities=0), ny_test)
        assert np.allclose(ny.dag().to_matrix(identities=0), ny_test.conj().T)

        circuit_4 = QuantumCircuit(1, [4], 0)
        ny_2 = circuit_4.noisey(0, [0, 2])

        ny_test_2 = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(ny_2.to_matrix(identities=0), ny_test_2)
        assert np.allclose(ny_2.dag().to_matrix(identities=0), ny_test_2.conj().T)
