from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestRz(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        rz = circuit_3.rz(0, [1, 2, np.pi / 3])

        # Rz(np.pi / 3, 1, 3)
        rz1_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.8660254 - 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.8660254 + 0.5j],
        ])

        assert np.allclose(rz.to_matrix(identities=0), rz1_test)

        rz1_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.8660254 + 0.5j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.8660254 - 0.5j],
        ])

        assert np.allclose(rz.dag().to_matrix(identities=0), rz1_test_dag)

        circuit_4 = QuantumCircuit(1, [4], 0)
        rz = circuit_4.rz(0, [1, 3, np.pi / 3])
        # Rz(np.pi / 3, 1, 3)
        rz1_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.8660254 - 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.8660254 + 0.5j],
        ])

        assert np.allclose(rz.to_matrix(identities=0), rz1_test)

        rz1_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.8660254 + 0.5j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.8660254 - 0.5j],
        ])

        assert np.allclose(rz.dag().to_matrix(identities=0), rz1_test_dag)

    @staticmethod
    def test_regulate_theta():
        circuit_4 = QuantumCircuit(1, [4], 0)
        rz = circuit_4.rz(0, [1, 2, 0.01 * np.pi])
        # Rz(0.01 * np.pi, 1, 4)
        assert round(rz.phi, 4) == 12.5978

    @staticmethod
    def test_cost():
        circuit_4 = QuantumCircuit(1, [4], 0)
        rz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        assert round(rz.cost, 4) == 0.0004

    @staticmethod
    def test_validate_parameter():
        circuit_3 = QuantumCircuit(1, [3], 0)
        rz = circuit_3.rz(0, [2, 0, np.pi / 7])

        assert rz.validate_parameter([2, 0, np.pi / 7])
        try:
            rz.validate_parameter([3, 0, np.pi / 7])
        except AssertionError:
            assert True
        try:
            rz.validate_parameter([1, 3, np.pi / 7])
        except AssertionError:
            assert True
