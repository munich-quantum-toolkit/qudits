from __future__ import annotations

import typing
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group  # type: ignore[import-not-found]

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def random_unitary_matrix(n: int) -> NDArray[np.complex128, np.complex128]:
    return unitary_group.rvs(n)


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        MQTQuditProvider()

        self.circuit_53 = QuantumCircuit(2, [5, 3], 0)
        self.circuit_s = QuantumCircuit(2, [5, 3], 0)

    def test_entangling_qr(self):
        target = random_unitary_matrix(15)

        t = self.circuit_53.cu_two([0, 1], target)
        MQTQuditProvider()
        eqr = EntangledQRCEX(t)
        decomp, _countcr, _countpsw = eqr.execute()
        for rotation in decomp:
            target = rotation.to_matrix(identities=2) @ target
        target /= target[0][0]
        res = (abs(target - np.identity(15, dtype="complex")) < 10e-5).all()
        assert res

    @staticmethod
    def test_entangling_qr_2():
        # Create the original circuit
        circuit = QuantumCircuit(2, [3, 3], 0)
        circuit.x(0)
        circuit.csum([0, 1])

        # Simulate the original circuit
        original_state = circuit.simulate()
        print("Original circuit simulation result:")
        print(original_state.round(3))

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["LogEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        # Simulate the compiled circuit
        compiled_state = new_circuit.simulate()
        print("\nCompiled circuit simulation result:")
        print(compiled_state.round(3))

        # Compare the results
        is_close = np.allclose(original_state, compiled_state)
        print(f"\nAre the simulation results close? {is_close}")
