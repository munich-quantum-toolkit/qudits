from __future__ import annotations

from ..quantum_circuit import QuantumCircuit, gates
from ..quantum_circuit.gate import GateTypes


def draw_qudit_local(circuit: QuantumCircuit) -> None:
    for line in range(circuit.num_qudits):
        print("|0>---", end="")
        for gate in circuit.instructions:
            if gate.gate_type == GateTypes.SINGLE and line == gate._target_qudits:
                if isinstance(gate, gates.VirtRz):
                    print("--[VRz" + str(gate.lev_a) + "(" + str(round(gate.phi, 2)) + ")]--", end="")

                elif isinstance(gate, gates.R):
                    print(
                        "--[R"
                        + str(gate.lev_a)
                        + str(gate.lev_b)
                        + "("
                        + str(round(gate.theta, 2))
                        + ","
                        + str(round(gate.phi, 2))
                        + ")]--",
                        end="",
                    )

                elif isinstance(gate, gates.CustomOne):
                    print("--[CuOne]--", end="")

                else:
                    print("--G--", end="")
            else:
                print("--MG--", end="")

        print("---=||")
