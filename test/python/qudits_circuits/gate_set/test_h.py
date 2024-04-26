from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestH(TestCase):
    def test___array__(self):
        circuit = QuantumCircuit(1, [3], 0)
        h = circuit.h(0)
        compare = np.array([
            [0.57735027 + 0.0j, 0.57735027 + 0.0j, 0.57735027 + 0.0j],
            [0.57735027 + 0.0j, -0.28867513 + 0.5j, -0.28867513 - 0.5j],
            [0.57735027 + 0.0j, -0.28867513 - 0.5j, -0.28867513 + 0.5j],
        ])
        compare = compare.round(8)
        h_m = h.to_matrix(identities=0)
        assert np.allclose(h_m, compare)

        circuit = QuantumCircuit(1, [4], 0)
        h = circuit.h(0)
        compare = np.array([
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
            [0.5 + 0.0j, 0.0 + 0.5j, -0.5 + 0.0j, -0.0 - 0.5j],
            [0.5 + 0.0j, -0.5 + 0.0j, 0.5 + 0.0j, -0.5 + 0.0j],
            [0.5 + 0.0j, -0.0 - 0.5j, -0.5 + 0.0j, 0.0 + 0.5j],
        ])
        compare = compare.round(8)
        h_m = h.to_matrix(identities=0)
        assert np.allclose(h_m, compare)

    def test_validate_parameter(self):
        circuit = QuantumCircuit(1, [3], 0)
        h = circuit.h(0)
        assert h.validate_parameter()
