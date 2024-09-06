from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestGellMann(TestCase):
    @staticmethod
    def test___array__():
        circuit = QuantumCircuit(1, [4], 0)

        ga = circuit.gellmann(0, [0, 1, "a"])
        gs = circuit.gellmann(0, [0, 1, "s"])
        # ga = GellMann(0, 1, 'a', dimension)
        # gs = GellMann(0, 1, 's', dimension)

        test_ga = np.array([
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_gs = np.array([
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(ga.to_matrix(identities=0), test_ga)
        assert np.allclose(gs.to_matrix(identities=0), test_gs)

        circuit = QuantumCircuit(1, [3], 0)
        ga = circuit.gellmann(0, [0, 1, "a"])
        gs = circuit.gellmann(0, [0, 1, "s"])
        # ga = GellMann(0, 1, 'a', dimension)
        # gs = GellMann(0, 1, 's', dimension)

        test_ga = np.array([
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_gs = np.array([
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(ga, test_ga)
        assert np.allclose(gs, test_gs)

        circuit = QuantumCircuit(1, [3], 0)
        ga = circuit.gellmann(0, [0, 2, "a"])
        gs = circuit.gellmann(0, [0, 2, "s"])
        # ga = GellMann(0, 2, 'a', dimension)
        # gs = GellMann(0, 2, 's', dimension)

        test_ga = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_gs = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(ga, test_ga)
        assert np.allclose(gs, test_gs)

    @staticmethod
    def test_validate_parameter():
        circuit = QuantumCircuit(1, [3], 0)
        ga = circuit.gellmann(0, [0, 2, "a"])
        gs = circuit.gellmann(0, [0, 2, "s"])
        assert ga.validate_parameter([0, 2, "a"])
        assert gs.validate_parameter([0, 2, "s"])
