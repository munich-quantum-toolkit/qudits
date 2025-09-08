from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.twodit.entanglement_qr import PSwapGen
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestPSwapGen(TestCase):
    def setUp(self) -> None:
        self.circuit_33 = QuantumCircuit(2, [3, 3], 0)

    def test_pswap_101_as_list(self):
        pswap_gen = PSwapGen(self.circuit_33, [0, 1])
        operations_p = pswap_gen.pswap_101_as_list_phases(np.pi / 4, -np.pi / 3)
        operations_np = pswap_gen.pswap_101_as_list_no_phases(np.pi / 4, -np.pi / 3)
        p_op = pswap_gen.permute_pswap_101_as_list(5, np.pi / 4, -np.pi / 3)

        mini_unitary_sim(self.circuit_33, operations_p).round(3)
        mini_unitary_sim(self.circuit_33, operations_np).round(3)
        mini_unitary_sim(self.circuit_33, p_op).round(3)

        self.circuit_33.r(0, [0, 1, np.pi / 4, -np.pi / 3]).to_matrix()
