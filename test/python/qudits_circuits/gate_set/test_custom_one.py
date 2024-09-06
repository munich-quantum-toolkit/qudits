from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCustomOne(TestCase):
    @staticmethod
    def test___array__():
        # All 33 csum
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cu = circuit_33.cu_one(0, 1j * np.identity(3))

        matrix = cu.to_matrix(identities=0)
        assert np.allclose(1j * np.identity(3), matrix)

        matrix_dag = cu.dag().to_matrix()
        assert np.allclose(-1j * np.identity(3), matrix_dag)

    @staticmethod
    def test_validate_parameter():
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cu = circuit_33.cu_one(0, np.identity(3))
        assert cu.validate_parameter(np.identity(3))
