from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.twodit.blocks.czrot import CZRotGen
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestCRot(TestCase):
    @staticmethod
    def test_czrot_101_as_list_ctrlop():
        circuit_1 = QuantumCircuit(2, [3, 3], 0)
        rzm = circuit_1.rz(1, [0, 1, np.pi / 4]).control([0], [1]).to_matrix(2)
        czrot_gen = CZRotGen(circuit_1, [0, 1])
        p_op = czrot_gen.z_from_crot_101_list(3, np.pi / 4)
        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rzm, mini_unitary_sim(circuit_2))

    @staticmethod
    def test_czrot_101_as_list_embedded():
        circuit_1 = QuantumCircuit(1, [9], 0)
        rzm = circuit_1.rz(0, [0, 1, np.pi / 4]).to_matrix(2)

        circuit_2 = QuantumCircuit(2, [3, 3], 0)
        czrot_gen = CZRotGen(circuit_2, [0, 1])
        p_op = czrot_gen.z_from_crot_101_list(0, np.pi / 4)
        circuit_2.set_instructions(p_op)
        assert np.allclose(rzm, mini_unitary_sim(circuit_2))
