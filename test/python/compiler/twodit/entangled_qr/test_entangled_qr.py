from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.dit_manager import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        MQTQuditProvider()

        self.circuit_33 = QuantumCircuit(2, [3, 3], 0)
        self.cx = self.circuit_33.cx([0, 1])

    def test_entangling_qr(self):
        self.cx.to_matrix()
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits", shots=1000)
        qudit_compiler = QuditCompiler()

        passes = ["LogEntQRCEXPass"]
        qudit_compiler.compile(backend_ion, self.circuit_33, passes)
