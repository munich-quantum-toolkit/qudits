from __future__ import annotations

from ..quantum_circuit import QuantumCircuit, gates
from ..quantum_circuit.components.extensions.gate_types import GateTypes


def draw_qudit_local(circuit: QuantumCircuit) -> None:
    for line in range(circuit.num_qudits):
        print("|0>---", end="")
        for gate in circuit.instructions:
            if gate.gate_type == GateTypes.SINGLE and line == gate.target_qudits:
                if isinstance(gate, gates.VirtRz):
                    print(f"--[VRz{gate.lev_a}({gate.phi:.2f})]--", end="")
                elif isinstance(gate, gates.R):
                    print(f"--[R{gate.lev_a}{gate.lev_b}({gate.theta:.2f},{gate.phi:.2f})]--", end="")
                elif isinstance(gate, gates.CustomOne):
                    print("--[CuOne]--", end="")
                else:
                    print("--G--", end="")
            else:
                print("--MG--", end="")
        print("---=||")
