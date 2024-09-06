from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import compute_f, sparsify
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestAnsatzSearch(TestCase):
    def test_sparsify(self) -> None:
        self.circuit = QuantumCircuit(2, [3, 3], 0)
        x = self.circuit.x(0).to_matrix()
        check = np.exp(1j * np.pi / 15 * (np.kron(np.eye(3), x) + np.kron(x, np.eye(3))))
        sparsity_initial = compute_f(check)

        u = self.circuit.cu_two([0, 1], check)
        circuit = sparsify(u)
        op = mini_unitary_sim(self.circuit, circuit.instructions)
        sparsity_final = compute_f(op)
        assert sparsity_final < sparsity_initial
