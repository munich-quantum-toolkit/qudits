from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestZRemovalPass(TestCase):
    def test_remove_z(self):
        provider = MQTQuditProvider()
        compiler = QuditCompiler()
        passes = ["ZRemovalPass"]
        backend_ion = provider.get_backend("faketraps2trits", shots=1000)

        qreg = QuantumRegister("test_reg", 3, [3, 3, 3])
        circ = QuantumCircuit(qreg)

        circ.virtrz(qreg[0], [0, np.pi / 3])
        circ.virtrz(qreg[1], [0, np.pi / 3])
        circ.r(qreg[1], [0, 1, np.pi, np.pi])
        circ.rz(qreg[0], [0, 1, np.pi / 3])

        circ.r(qreg[0], [0, 1, np.pi, np.pi / 2])
        circ.r(qreg[1], [0, 1, np.pi, np.pi / 3])
        circ.r(qreg[2], [0, 1, np.pi, np.pi / 4])
        circ.r(qreg[1], [0, 1, np.pi, np.pi / 5])

        circ.virtrz(qreg[0], [0, np.pi / 3])
        circ.r(qreg[1], [0, 1, np.pi, np.pi / 6])
        circ.rz(qreg[1], [0, 1, np.pi / 3])
        circ.rz(qreg[1], [0, 1, np.pi / 3])

        new_circuit = compiler.compile(backend_ion, circ, passes)

        # Rs
        assert len(new_circuit.instructions) == 6
        assert new_circuit.instructions[0].qasm_tag == "rxy"
        assert new_circuit.instructions[1].qasm_tag == "rxy"
        assert new_circuit.instructions[2].qasm_tag == "rxy"
        assert new_circuit.instructions[3].qasm_tag == "rxy"
        assert new_circuit.instructions[4].qasm_tag == "rxy"
        assert new_circuit.instructions[5].qasm_tag == "rxy"

        assert new_circuit.instructions[0].phi == np.pi
        assert new_circuit.instructions[1].phi == np.pi / 2
        assert new_circuit.instructions[2].phi == np.pi / 3
        assert new_circuit.instructions[3].phi == np.pi / 4
        assert new_circuit.instructions[4].phi == np.pi / 5
        assert new_circuit.instructions[5].phi == np.pi / 6
