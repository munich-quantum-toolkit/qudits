from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestNoiseX(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        nx = circuit_3.noisex(0, [1, 2])
        nx_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(nx.to_matrix(identities=0), nx_test)
        assert np.allclose(nx.dag().to_matrix(identities=0), nx_test.conj().T)

        circuit_4 = QuantumCircuit(1, [4], 0)
        nx_2 = circuit_4.noisex(0, [0, 2])

        nx_test_2 = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(nx_2.to_matrix(identities=0), nx_test_2)
        assert np.allclose(nx_2.dag().to_matrix(identities=0), nx_test_2.conj().T)
