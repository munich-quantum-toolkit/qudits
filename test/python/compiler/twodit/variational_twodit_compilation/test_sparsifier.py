from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import compute_F, sparsify
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


def generate_unitary_matrix(n):
    random_matrix = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # Perform QR decomposition
    q, r = np.linalg.qr(random_matrix)

    # Ensure q is unitary
    return q @ np.diag(np.sign(np.diag(r)))


class TestAnsatzSearch(TestCase):
    def test_sparsify(self) -> None:
        self.circuit = QuantumCircuit(2, [3, 3], 0)
        x = self.circuit.x(0).to_matrix()
        check = np.exp(1j * np.pi / 15 * (np.kron(np.eye(3), x) + np.kron(x, np.eye(3))))
        sparsity_initial = compute_F(check)

        u = self.circuit.cu_two([0, 1], check)
        circuit = sparsify(u)
        op = mini_unitary_sim(self.circuit, circuit.instructions)
        sparsity_final = compute_F(op)
        assert sparsity_final < sparsity_initial
