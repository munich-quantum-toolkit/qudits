from __future__ import annotations

import typing
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group  # type: ignore[import-not-found]

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def random_unitary_matrix(n: int) -> NDArray[np.complex128, np.complex128]:
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
        backend_ion = provider.get_backend("faketraps2trits")
        qudit_compiler = QuditCompiler()

        passes = ["LogEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, self.circuit_33, passes)

        for rotation in new_circuit.instructions:
            target = rotation.to_matrix(identities=2) @ target
        target /= target[0][0]
        res = (abs(target - np.identity(15, dtype="complex")) < 10e-5).all()
        assert res
