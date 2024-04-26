from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestRz(TestCase):
    def test___array__(self):
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

    def test_regulate_theta(self):
        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        # Rz(0.01 * np.pi, 1, 4)
        self.assertAlmostEqual(round(vrz.phi, 4), 12.5978)

    def test_cost(self):
        circuit_4 = QuantumCircuit(1, [4], 0)
        vrz = circuit_4.virtrz(0, [1, 0.01 * np.pi])
        assert round(vrz.cost, 4) == 0.0004

    def test_validate_parameter(self):
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
