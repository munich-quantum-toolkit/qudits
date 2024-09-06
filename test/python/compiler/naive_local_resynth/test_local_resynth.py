from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_sim
from mqt.qudits.compiler.naive_local_resynth.local_resynth import NaiveLocResynthOptPass
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.gates import CEx, CustomMulti, R
from mqt.qudits.simulation import MQTQuditProvider


class TestNaiveLocResynthOptPass(TestCase):
    def setUp(self):
        self.circuit = QuantumCircuit(3, [3, 3, 3], 0)

    def test_transpile(self):
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps3six")

        gates = [
            R(self.circuit, "R", 0, [0, 1, np.pi / 3, np.pi / 6], self.circuit.dimensions[0]),
            R(self.circuit, "R", 1, [0, 1, np.pi / 7, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 0, [0, 1, -np.pi / 4, np.pi / 7], self.circuit.dimensions[0]),
            R(self.circuit, "R", 1, [0, 1, -np.pi / 4, np.pi / 7], self.circuit.dimensions[1]),
            CEx(self.circuit, "CEx12", [1, 2], None, [self.circuit.dimensions[i] for i in [1, 2]]),
            R(self.circuit, "R", 0, [0, 1, -np.pi / 5, np.pi / 7], self.circuit.dimensions[0]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 1, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 1, [1, 2, -np.pi, np.pi / 2], self.circuit.dimensions[1]),
            R(self.circuit, "R", 0, [0, 1, -np.pi / 5, np.pi / 7], self.circuit.dimensions[0]),
            R(self.circuit, "R", 0, [1, 2, -np.pi, np.pi / 2], self.circuit.dimensions[0]),
            CustomMulti(
                self.circuit, "CUm", [0, 1, 2], np.identity(27), [self.circuit.dimensions[i] for i in [0, 1, 2]]
            ),
            R(self.circuit, "R", 2, [0, 1, np.pi, np.pi / 2], self.circuit.dimensions[2]),
            CEx(self.circuit, "CEx", [0, 1], None, [self.circuit.dimensions[i] for i in [0, 1]]),
            R(self.circuit, "R", 2, [0, 1, -np.pi, np.pi / 2], self.circuit.dimensions[2]),
        ]
        self.circuit.set_instructions(gates)
        resynth = NaiveLocResynthOptPass(backend_ion)
        qc = resynth.transpile(self.circuit)

        s = mini_sim(self.circuit)
        s2 = mini_sim(qc)

        assert np.allclose(s, s2)
