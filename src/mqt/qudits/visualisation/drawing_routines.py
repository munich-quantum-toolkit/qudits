from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.custom_one import CustomOne
from mqt.qudits.qudit_circuits.components.instructions.gate_set.r import R
from mqt.qudits.qudit_circuits.components.instructions.gate_set.virt_rz import VirtRz


def draw_qudit_local(circuit):
    for line in range(circuit.num_qudits):
        print("|0>---", end="")
        for gate in circuit.instructions:
            if gate.gate_type == GateTypes.SINGLE and line == gate._target_qudits:
                if isinstance(gate, VirtRz):
                    print("--[VRz" + str(gate.lev_a) + "(" + str(round(gate.phi, 2)) + ")]--", end="")

                elif isinstance(gate, R):
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

                elif isinstance(gate, CustomOne):
                    print("--[CuOne]--", end="")

                else:
                    print("--G--", end="")
            else:
                print("--MG--", end="")

        print("---=||")
