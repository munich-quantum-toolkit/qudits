from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.twodit.blocks.czrot import CZRotGen
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


class TestCRot(TestCase):
    def setUp(self) -> None:
        self.circuit_33 = QuantumCircuit(2, [4, 4], 0)

    def test_crot_101_as_list(self):
        self.circuit_33.rz(0, [0, 1, np.pi / 4]).to_matrix().round(3)
        czrot_gen = CZRotGen(self.circuit_33, [0, 1])
        p_op = czrot_gen.z_from_crot_101_list(0, np.pi / 4)
        mini_unitary_sim(self.circuit_33, p_op).round(3)
