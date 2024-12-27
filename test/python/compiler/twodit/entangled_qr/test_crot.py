from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.twodit.entanglement_qr import CRotGen
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCRot(TestCase):
    @staticmethod
    def test_crot_101_as_list_ctrlop():
        circuit_1 = QuantumCircuit(2, [3, 3], 0)
        rm = circuit_1.r(1, [0, 1, 1.0471975511965972, -2.513274122871836]).control([0], [1]).to_matrix(2)

        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        crot_gen = CRotGen(circuit_2, [0, 1])
        p_op = crot_gen.permute_crot_101_as_list(3, 1.0471975511965972, -2.513274122871836)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rm, mini_unitary_sim(circuit_2))

    @staticmethod
    def test_crot_101_as_list_embedded():
        circuit_1 = QuantumCircuit(1, [9], 0)
        rm = circuit_1.r(0, [0, 1, 1.0471975511965972, -2.513274122871836]).to_matrix(2)

        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        crot_gen = CRotGen(circuit_2, [0, 1])
        p_op = crot_gen.permute_crot_101_as_list(0, 1.0471975511965972, -2.513274122871836)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rm, mini_unitary_sim(circuit_2))
