from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ....quantum_circuit import gates
from ... import CompilerPass
from ...compilation_minitools import pi_mod

if TYPE_CHECKING:
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....quantum_circuit.gates import R, VirtRz
    from ....simulation.backends.backendv2 import Backend


class ZPropagationOptPass(CompilerPass):
    def __init__(self, backend: Backend, back: bool = True) -> None:
        super().__init__(backend)
        self.back = back

    @staticmethod
    def transpile_gate(gate: Gate) -> list[Gate]:
        return [gate]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        from ....core.lanes import Lanes
        from ....quantum_circuit.components.extensions.gate_types import GateTypes

        self.circuit = circuit
        self.lanes = Lanes(self.circuit)

        final_mappings = []
        for i, graph in enumerate(self.backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])

        for line in sorted(self.lanes.index_dict.keys()):
            extracted_line: list[tuple[int, Gate]] = self.lanes.index_dict[line]
            grouped_line: dict[int, list[list[tuple[int, Gate]]]] = self.lanes.find_consecutive_singles(extracted_line)
            new_line: list[tuple[int, Gate]] = []
            for group in grouped_line[line]:
                if group[0][1].gate_type == GateTypes.SINGLE:
                    new_line.extend(self.propagate_z(circuit, group, self.back))
                else:
                    new_line.append(group[0])

            self.lanes.index_dict[line] = new_line

        new_instructions = self.lanes.extract_instructions()
        transpiled_circuit = circuit.copy()
        initial_mappings = []
        for i, graph in enumerate(self.backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_initial_mappings(initial_mappings)
        transpiled_circuit.set_final_mappings(final_mappings)

        return transpiled_circuit.set_instructions(new_instructions)

    @staticmethod
    def propagate_z(circuit: QuantumCircuit, group: list[tuple[int, Gate]], back: bool) -> list[tuple[int, R | VirtRz]]:
        tag = group[0][0]
        line = [couple[1] for couple in group]
        z_angles: dict[int, float] = {}
        list_of_x_yrots: list[R] = []
        qudit_index = cast("int", line[0].target_qudits)
        dimension = cast("int", line[0].dimensions)

        for i in range(dimension):
            z_angles[i] = 0.0

        if back:
            line.reverse()

        from ....quantum_circuit.gates import R, VirtRz

        for gate_index in range(len(line)):
            if isinstance(line[gate_index], R):
                if back:
                    new_phi = pi_mod(
                        line[gate_index].phi + z_angles[line[gate_index].lev_a] - z_angles[line[gate_index].lev_b]
                    )
                else:
                    new_phi = pi_mod(
                        line[gate_index].phi - z_angles[line[gate_index].lev_a] + z_angles[line[gate_index].lev_b]
                    )
                list_of_x_yrots.append(
                    gates.R(
                        circuit,
                        "R",
                        qudit_index,
                        [line[gate_index].lev_a, line[gate_index].lev_b, line[gate_index].theta, new_phi],
                        dimension,
                    )
                )
            elif isinstance(line[gate_index], VirtRz):
                z_angles[line[gate_index].lev_a] = pi_mod(z_angles[line[gate_index].lev_a] + line[gate_index].phi)
        if back:
            list_of_x_yrots.reverse()

        zseq = []
        zseq.extend([
            gates.VirtRz(circuit, "VRz", qudit_index, [e_lev, z_angles[e_lev]], dimension) for e_lev in z_angles
        ])
        combined_seq = zseq + list_of_x_yrots if back else list_of_x_yrots + zseq
        return [(tag, gate) for gate in combined_seq]

    """
    @staticmethod
    def find_intervals_with_same_target_qudits(instructions: list[Gate]) -> list[tuple[int, ...]]:
        intervals: list[tuple[int, ...]] = []
        current_interval: list[int] = []
        current_target_qudits: list[int] = []
        target_qudits: list[int] = []

        for i, instruction in enumerate(instructions):
            if isinstance(instruction.target_qudits, int):
                target_qudits = [instruction.target_qudits]
            elif isinstance(instruction.target_qudits, list):
                target_qudits = instruction.target_qudits

            if len(current_target_qudits) == 0 or target_qudits == current_target_qudits:
                # If it's the first gate_matrix or the target qudits are the same, add to the current interval
                current_interval.append(i)
            else:
                # If the target qudits are different, save the current interval and start a new one
                intervals.append(tuple(current_interval))
                current_interval = [i]

            current_target_qudits = target_qudits

        # Save the last interval if it exists
        if current_interval:
            intervals.append(tuple(current_interval))

        return intervals

    def remove_z(self, original_circuit: QuantumCircuit, back: bool = True) -> QuantumCircuit:
        circuit = original_circuit.copy()
        new_instructions: list[Gate] = copy.deepcopy(circuit.instructions)
        intervals: list[tuple[int, ...]] = self.find_intervals_with_same_target_qudits(circuit.instructions)

        for interval in intervals:
            if len(interval) > 1:
                from ....quantum_circuit.gates import R, VirtRz

                sequence = cast(list[Union[R, VirtRz]], circuit.instructions[interval[0]: interval[-1] + 1])
                sequence = cast("list[Union[R, VirtRz]]", circuit.instructions[interval[0] : interval[-1] + 1])
                fixed_seq: list[R] = []
                z_tail: list[VirtRz] = []
                combined_seq = self.propagate_z(circuit, sequence, back)

                # combined_seq = z_tail + fixed_seq if back else fixed_seq + z_tail
                new_instructions[interval[0]: interval[-1] + 1] = []
                new_instructions.extend(combined_seq)

        return circuit.set_instructions(new_instructions)
    """
