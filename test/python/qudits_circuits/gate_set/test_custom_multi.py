from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCustomMulti(TestCase):
    @staticmethod
    def test___array__():
        # All 33 csum
        circuit_33 = QuantumCircuit(3, [3, 3, 3], 0)
        cu = circuit_33.cu_multi([0, 1, 2], 1j * np.identity(27))

        matrix = cu.to_matrix(identities=0)
        assert np.allclose(1j * np.identity(27), matrix)

        matrix_dag = cu.dag().to_matrix()
        assert np.allclose(-1j * np.identity(27), matrix_dag)

    @staticmethod
    def test_validate_parameter():
        circuit_33 = QuantumCircuit(3, [3, 3, 3], 0)
        cu = circuit_33.cu_multi([0, 1, 2], np.identity(27))
        assert cu.validate_parameter(np.identity(27))
