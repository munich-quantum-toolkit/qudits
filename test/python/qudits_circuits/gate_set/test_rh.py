from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestRh(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [1, 2])

        rh_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0 + 0.70711j, 0 + 0.70711j],
            [0.0 + 0.0j, 0 + 0.70711j, 0 - 0.70711j],
        ])

        assert np.allclose(rh.to_matrix(identities=0), rh_test)
        rh_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0 - 0.70711j, 0 - 0.70711j],
            [0.0 + 0.0j, 0 - 0.70711j, 0 + 0.70711j],
        ])

        assert np.allclose(rh.dag().to_matrix(identities=0), rh_test_dag)

        circuit_4 = QuantumCircuit(1, [4], 0)
        rh = circuit_4.rh(0, [1, 2])

        rh_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0 + 0.70711j, 0 + 0.70711j, 0.0 + 0.0j],
            [0.0 + 0j, 0 + 0.70711j, 0 - 0.70711j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(rh.to_matrix(identities=0), rh_test)

        rh_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0 - 0.70711j, 0 - 0.70711j, 0.0 + 0.0j],
            [0.0 + 0j, 0 - 0.70711j, 0 + 0.70711j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(rh.dag().to_matrix(identities=0), rh_test_dag)

    @staticmethod
    def test_levels_setter():
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [2, 0])
        assert rh.lev_a == 0
        assert rh.lev_b == 2
        assert rh.original_lev_a == 2
        assert rh.original_lev_b == 0

    @staticmethod
    def test_validate_parameter():
        circuit_3 = QuantumCircuit(1, [3], 0)
        rh = circuit_3.rh(0, [2, 0])

        assert rh.validate_parameter([2, 0])
        try:
            rh.validate_parameter([3, 0])
        except AssertionError:
            assert True
        try:
            rh.validate_parameter([1, 3])
        except AssertionError:
            assert True
