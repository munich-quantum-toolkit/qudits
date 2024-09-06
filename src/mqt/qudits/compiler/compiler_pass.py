from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class CompilerPass(ABC):
    def __init__(self, backend: Backend) -> None:
        self.backend = backend

    @staticmethod  # type: ignore[misc]
    def transpile_gate(gate: Gate) -> list[Gate]:
        raise NotImplementedError

    @overload
    def transpile_gate(self, gate: Gate) -> list[Gate]:  # noqa: F811
        raise NotImplementedError

    @abstractmethod
    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        pass
