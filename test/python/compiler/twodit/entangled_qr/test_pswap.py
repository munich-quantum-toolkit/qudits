from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.twodit.entanglement_qr import PSwapGen
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestPSwapGen(TestCase):
    @staticmethod
    def test_pswap_101_as_list_no_phases():
        # Create reference circuit and matrix
        circuit_1 = QuantumCircuit(1, [9], 0)
        rm = circuit_1.r(0, [2, 3, np.pi / 4, -np.pi / 3]).to_matrix(2)

        # Create test circuit and compare
        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        pswap_gen = PSwapGen(circuit_2, [0, 1])
        p_op = pswap_gen.pswap_101_as_list_no_phases(np.pi / 4, -np.pi / 3)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rm, mini_unitary_sim(circuit_2))

    @staticmethod
    def test_permute_pswap_101_as_list():
        # Create reference circuit and matrix
        circuit_1 = QuantumCircuit(1, [9], 0)
        rm = circuit_1.r(0, [5, 6, np.pi / 4, -np.pi / 3]).to_matrix(2)

        # Create test circuit and compare
        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        pswap_gen = PSwapGen(circuit_2, [0, 1])
        p_op = pswap_gen.permute_pswap_101_as_list(5, np.pi / 4, -np.pi / 3)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rm, mini_unitary_sim(circuit_2))
