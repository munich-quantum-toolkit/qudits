import copy
from mqt.qudits.compiler.compiler_pass import CompilerPass
from mqt.qudits.qudit_circuits.components.instructions.gate_set.rz import Rz
from mqt.qudits.qudit_circuits.components.instructions.gate_set.virt_rz import VirtRz
from mqt.qudits.qudit_circuits.components.instructions.gate_set.z import Z


class ZRemovalPass(CompilerPass):
    def __init__(self, backend):
        super().__init__(backend)

    def transpile(self, circuit):
        circuit = self.remove_initial_rz(circuit)
        circuit = self.remove_trailing_rz_sequence(circuit)
        return circuit

    def remove_rz_gates(self, original_circuit, reverse=False):
        indices_to_remove = []
        circuit = original_circuit.copy()
        new_instructions = copy.deepcopy(circuit.instructions)

        seen_target_qudits = set()
        indices = range(len(circuit.instructions)) if not reverse else range(len(circuit.instructions) - 1, -1, -1)

        for i in indices:
            instruction = circuit.instructions[i]
            if len(seen_target_qudits) == circuit.num_qudits:
                # If all qubits are seen, break the loop
                break

            target_qudits = instruction._target_qudits
            if isinstance(target_qudits, list):
                # If target_qudits is a list, add each element to the set and
                # continue to the next iteration because we work only with single gates
                seen_target_qudits.update(target_qudits)
                continue
            else:
                if target_qudits not in seen_target_qudits and isinstance(instruction, (Rz, VirtRz, Z)):
                    indices_to_remove.append(i)
                else:
                    seen_target_qudits.add(target_qudits)

        new_instructions = [op for index, op in enumerate(new_instructions) if index not in indices_to_remove]
        return circuit.set_instructions(new_instructions)

    def remove_initial_rz(self, original_circuit):
        return self.remove_rz_gates(original_circuit)

    def remove_trailing_rz_sequence(self, original_circuit):
        return self.remove_rz_gates(original_circuit, reverse=True)

