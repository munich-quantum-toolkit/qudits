# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class CompilerPass(ABC):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @abstractmethod
    def transpile_gate(self, gate: Gate) -> list[Gate]: ...

    @abstractmethod
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit: ...
