from __future__ import annotations

import copy

from ....quantum_circuit import gates
from ... import CompilerPass
from ...compilation_minitools import pi_mod


class ZPropagationPass(CompilerPass):
    def __init__(self, backend, back=True) -> None:
        super().__init__(backend)
        self.back = back

    def transpile(self, circuit):
        return self.remove_z(circuit, self.back)

    def propagate_z(self, circuit, line, back):
        Z_angles = {}
        # list_of_Zrots = []
        list_of_XYrots = []
        qudit_index = line[0]._target_qudits
        dimension = line[0]._dimensions

        for i in range(dimension):
            Z_angles[i] = 0.0

        if back:
            line.reverse()

        for gate_index in range(len(line)):
            try:
                line[gate_index].lev_b
                # object is R
                if back:
                    new_phi = pi_mod(
                            line[gate_index].phi + Z_angles[line[gate_index].lev_a] - Z_angles[line[gate_index].lev_b]
                    )
                else:
                    new_phi = pi_mod(
                            line[gate_index].phi - Z_angles[line[gate_index].lev_a] + Z_angles[line[gate_index].lev_b]
                    )

                list_of_XYrots.append(
                        gates.R(
                                circuit,
                                "R",
                                qudit_index,
                                [line[gate_index].lev_a, line[gate_index].lev_b, line[gate_index].theta, new_phi],
                                dimension,
                        )
                )
                # list_of_XYrots.append(R(line[gate_index].theta, new_phi,
                # line[gate_index].lev_a, line[gate_index].lev_b, line[gate_index].dimension))
            except AttributeError:
                try:
                    line[gate_index].lev_a
                    # object is VirtRz
                    Z_angles[line[gate_index].lev_a] = pi_mod(Z_angles[line[gate_index].lev_a] + line[gate_index].phi)
                except AttributeError:
                    pass
        if back:
            list_of_XYrots.reverse()

        Zseq = []
        for e_lev in list(Z_angles):
            Zseq.append(gates.VirtRz(circuit, "VRz", qudit_index, [e_lev, Z_angles[e_lev]], dimension))
            # Zseq.append(Rz(Z_angles[e_lev], e_lev, QC.dimension))

        return list_of_XYrots, Zseq

    def find_intervals_with_same_target_qudits(self, instructions):
        intervals = []
        current_interval = []
        current_target_qudits = None

        for i, instruction in enumerate(instructions):
            target_qudits = instruction._target_qudits

            if current_target_qudits is None or target_qudits == current_target_qudits:
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

    def remove_z(self, original_circuit, back=True):
        circuit = original_circuit.copy()
        new_instructions = copy.deepcopy(circuit.instructions)
        intervals = self.find_intervals_with_same_target_qudits(circuit.instructions)

        for interval in intervals:
            if len(interval) > 1:
                sequence = circuit.instructions[interval[0]: interval[-1] + 1]
                fixed_seq, z_tail = self.propagate_z(circuit, sequence, back)
                if back:
                    new_instructions[interval[0]: interval[-1] + 1] = z_tail + fixed_seq
                else:
                    new_instructions[interval[0]: interval[-1] + 1] = fixed_seq + z_tail

        return circuit.set_instructions(new_instructions)
