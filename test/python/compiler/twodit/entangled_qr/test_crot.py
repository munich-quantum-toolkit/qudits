from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.twodit.entanglement_qr import CRotGen
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


class TestCRot(TestCase):
    def setUp(self) -> None:
        self.circuit_33 = QuantumCircuit(2, [4, 4], 0)

    def test_crot_101_as_list(self):
        self.circuit_33.r(0, [0, 1, 1.0471975511965972, -2.513274122871836]).to_matrix().round(3)
        crot_gen = CRotGen(self.circuit_33, [0, 1])
        operations = crot_gen.crot_101_as_list(1.0471975511965972, -2.513274122871836)
        p_op = crot_gen.permute_crot_101_as_list(2, 1.0471975511965972, -2.513274122871836)
        mini_unitary_sim(self.circuit_33, operations).round(3)
        mini_unitary_sim(self.circuit_33, p_op).round(3)
        self.circuit_33.r(0, [0, 1, 1.0471975511965972, -2.513274122871836]).to_matrix()
