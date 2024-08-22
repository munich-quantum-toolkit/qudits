from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate


class CompilerPass(ABC):
    def __init__(self, backend, **kwargs) -> None:
        self.backend = backend

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        pass

    @abstractmethod
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass
