from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestVirtRz(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        vrz = circuit_3.virtrz(0, [1, np.pi / 3])
        # Rz(np.pi / 3, 1, 3)
        vrz1_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 - 0.8660254j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(vrz.to_matrix(identities=0), vrz1_test)

        vrz1_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.8660254j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(vrz.dag().to_matrix(identities=0), vrz1_test_dag)

        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, np.pi / 3])
        # Rz(np.pi / 3, 1, 3)
        vrz1_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 - 0.8660254j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(vrz.to_matrix(identities=0), vrz1_test)

        vrz1_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.5 + 0.8660254j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(vrz.dag().to_matrix(identities=0), vrz1_test_dag)

    @staticmethod
    def test_regulate_theta():
        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        # Rz(0.01 * np.pi, 1, 4)
        assert round(vrz.phi, 4) == 12.5978

    @staticmethod
    def test_cost():
        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        assert round(vrz.cost, 4) == 0.0004

    @staticmethod
    def test_validate_parameter():
        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        assert vrz.validate_parameter([1, np.pi])
        try:
            vrz.validate_parameter([4, np.pi])
        except AssertionError:
            assert True
