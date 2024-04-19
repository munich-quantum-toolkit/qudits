from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.dit_manager import QuditCompiler
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.simulation.provider.qudit_provider import MQTQuditProvider


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        provider = MQTQuditProvider()

        self.circuit_33 = QuantumCircuit(2, [3, 3], 0)
        self.cx = self.circuit_33.cx([0, 1])

    def test_entangling_qr(self):
        target = self.cx.to_matrix()
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits", shots=1000)
        qudit_compiler = QuditCompiler()

        passes = ["LogEntQRCEXPass"]
        compiled_circuit_qr = qudit_compiler.compile(backend_ion, self.circuit_33, passes)

