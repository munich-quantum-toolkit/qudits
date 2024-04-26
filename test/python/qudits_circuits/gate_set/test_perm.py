from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestPerm(TestCase):
    def test___array__(self):
        circuit = QuantumCircuit(2, [3, 2], 0)
        ru1 = circuit.pm([0, 1], [0, 2, 1, 5, 3, 4]).to_matrix()
        matrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
        ])
        assert np.allclose(ru1, matrix)

    def test_validate_parameter(self):
        circuit = QuantumCircuit(1, [3, 3], 0)
        p = circuit.pm([0], [0, 1, 2])
        assert p.validate_parameter([0, 1, 2])
        try:
            p.validate_parameter([0, 1, 5])
        except AssertionError:
            assert True
