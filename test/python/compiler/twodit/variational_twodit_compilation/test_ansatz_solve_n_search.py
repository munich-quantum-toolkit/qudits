# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from unittest import TestCase

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestAnsatzSearch(TestCase):
    def test_binary_search_compile(self) -> None:
        self.circuit_og = QuantumCircuit(2, [2, 2], 0)
        self.circuit_og.cx([0, 1])
        # circuit = variational_compile(cx, 1e-1, "MS", 2)
        # op = mini_unitary_sim(circuit, circuit.instructions)
        # f = fidelity_on_unitares(op, cx.to_matrix())
        # assert 0 < f < 1
