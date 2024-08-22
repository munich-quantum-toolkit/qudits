from __future__ import annotations

import operator
from functools import reduce
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


def mini_unitary_sim(circuit, list_of_op):
    size = reduce(operator.mul, circuit.dimensions)
    id_mat = np.identity(size)
    for gate in list_of_op:
        id_mat = gate.to_matrix(identities=2) @ id_mat
        id_mat.round(2)
    return id_mat


def random_unitary_matrix(n):
    return unitary_group.rvs(n)


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        MQTQuditProvider()

        self.circuit_33 = QuantumCircuit(2, [5, 3], 0)
        self.circuit_s = QuantumCircuit(2, [5, 3], 0)

    def test_entangling_qr(self):
        target = random_unitary_matrix(15)

        self.circuit_33.cu_two([0, 1], target)
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits", shots=1000)
        qudit_compiler = QuditCompiler()

        passes = ["LogEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, self.circuit_33, passes)

        for rotation in new_circuit.instructions:
            target = rotation.to_matrix(identities=2) @ target
        target /= target[0][0]
        res = (abs(target - np.identity(15, dtype="complex")) < 10e-5).all()
        assert res
