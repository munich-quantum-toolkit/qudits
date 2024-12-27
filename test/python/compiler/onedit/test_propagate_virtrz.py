from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.onedit import ZPropagationOptPass
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestZPropagationOptPass(TestCase):
    def setUp(self):
        provider = MQTQuditProvider()
        self.compiler = QuditCompiler()
        self.passes = ["ZPropagationOptPass"]
        self.backend_ion = provider.get_backend("faketraps2trits")

    def test_transpile(self):
        qreg = QuantumRegister("test_reg", 1, [3])
        circ = QuantumCircuit(qreg)

        circ.r(qreg[0], [0, 1, np.pi, np.pi / 3]).dag()
        circ.virtrz(qreg[0], [0, np.pi / 3])
        circ.r(qreg[0], [0, 1, np.pi, np.pi / 3])
        circ.r(qreg[0], [0, 1, np.pi, np.pi / 3])
        circ.virtrz(qreg[0], [0, np.pi / 3])

        # R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3),
        # R(np.pi, np.pi / 3, 0, 1, 3), R(np.pi, np.pi / 3, 0, 1, 3), Rz(np.pi / 3, 0, 3)]
        pass_z = ZPropagationOptPass(backend=self.backend_ion, back=True)
        new_circuit = pass_z.transpile(circ)

        u1 = mini_unitary_sim(circ)
        u2 = mini_unitary_sim(new_circuit)

        assert np.allclose(u1, u2)

        # VirtZs
        assert new_circuit.instructions[0].phi == 2 * np.pi / 3
        assert new_circuit.instructions[1].phi == 4 * np.pi
        assert new_circuit.instructions[2].phi == 4 * np.pi
        # Rs
        assert new_circuit.instructions[3].phi == np.pi / 3 + 2 * np.pi / 3
        assert new_circuit.instructions[4].phi == 2 * np.pi / 3
        assert new_circuit.instructions[5].phi == 2 * np.pi / 3

        pass_z = ZPropagationOptPass(backend=self.backend_ion, back=False)
        new_circuit = pass_z.transpile(circ)

        # Rs
        assert new_circuit.instructions[0].phi == np.pi / 3
        assert new_circuit.instructions[1].phi == 0.0
        assert new_circuit.instructions[2].phi == 0.0
        # VirtZs
        assert new_circuit.instructions[3].phi == 2 * np.pi / 3
        assert new_circuit.instructions[4].phi == 4 * np.pi
        assert new_circuit.instructions[5].phi == 4 * np.pi

        u1nb = mini_unitary_sim(circ)
        u2nb = mini_unitary_sim(new_circuit)

        assert np.allclose(u1nb, u2nb)
