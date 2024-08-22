from __future__ import annotations

import operator

from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes


class Lanes:
    def __init__(self, circuit) -> None:
        self.fast_lookup = None
        self.consecutive_view = None
        self.circuit = circuit
        self.instructions = circuit.instructions
        self.pre_process_ops()
        self.index_dict = self.create_lanes()

    def pre_process_ops(self) -> None:
        gates = []
        entanglement_counter = 0
        for gate in self.instructions:
            if gate.gate_type != GateTypes.SINGLE:
                entanglement_counter += 1
                gates.append((entanglement_counter, gate))
                # entanglement_counter += 1
            else:
                gates.append((entanglement_counter, gate))
        self.instructions = gates

    def create_lanes(self):
        self.index_dict = {}
        for gate_tuple in self.instructions:
            gate = gate_tuple[1]
            if gate.gate_type == GateTypes.SINGLE:
                index = gate._target_qudits
                if index not in self.index_dict:
                    self.index_dict[index] = []
                self.index_dict[index].append(gate_tuple)
            elif gate.gate_type in {GateTypes.TWO, GateTypes.MULTI}:
                indices = gate._target_qudits
                for index in indices:
                    if index not in self.index_dict:
                        self.index_dict[index] = []
                    self.index_dict[index].append(gate_tuple)

        self.consecutive_view = self.find_consecutive_singles()
        return self.index_dict

    def extract_instructions(self):
        combined_list = []
        seen_ids = set()

        for line in sorted(self.index_dict.keys()):  # Iterate over keys in sorted order
            for gate_tuple in self.index_dict[line]:
                gate = gate_tuple[1]
                obj_id = id(gate)
                if obj_id not in seen_ids:
                    combined_list.append(gate_tuple)
                    seen_ids.add(obj_id)

        sorted_list = sorted(combined_list, key=operator.itemgetter(0))
        self.instructions = []
        for gate_tuple in sorted_list:
            self.instructions.append(gate_tuple[1])

        return self.instructions

    def extract_lane(self, qudit_line):
        lane_gates = []
        for gate_tuple in self.index_dict[qudit_line]:
            lane_gates.append(gate_tuple[1])

        return lane_gates

    def find_consecutive_singles(self, gates=None):
        if gates is None:
            gates = self.instructions
        from collections import defaultdict

        consecutive_groups = defaultdict(list)
        for gate_tuple in gates:
            gate = gate_tuple[1]
            if gate.gate_type == GateTypes.SINGLE:
                if consecutive_groups[gate._target_qudits]:
                    consecutive_groups[gate._target_qudits][-1].append(gate_tuple)
                else:
                    consecutive_groups[gate._target_qudits] = [[gate_tuple]]
            else:
                for qudit in gate._target_qudits:
                    consecutive_groups[qudit].append([gate_tuple])
                    consecutive_groups[qudit].append([])
        consecutive_groups = {
            key: [sublist for sublist in value if sublist] for key, value in consecutive_groups.items()
        }

        self.consecutive_view = consecutive_groups

        self.fast_lookup = {}
        # Build the index
        for line_number, line_groups in consecutive_groups.items():
            for group_number, group in enumerate(line_groups):
                for gate_number, gate_tuple in enumerate(group):
                    gate = gate_tuple[1]
                    self.fast_lookup[gate] = (line_number, group_number, gate_number)

        return consecutive_groups

    def replace_gates_in_lane(self, line, start_index, end_index, new_gate) -> None:
        # Find the list associated with the line
        if line in self.index_dict:
            gates_of_line = self.index_dict[line]
        else:
            return  # Exit if line not found in index_dict

        # Remove objects within the specified interval [start_index, end_index]
        ordering_id = gates_of_line[start_index][0]
        objects_to_remove = []
        for i in range(start_index, end_index + 1):
            if i < len(gates_of_line):
                objects_to_remove.append(gates_of_line[i])

        for obj in objects_to_remove:
            gates_of_line.remove(obj)

        # Add new_gate at the start_index
        gates_of_line.insert(start_index, (ordering_id, new_gate))

    def next_is_local(self, gate):
        line_number, group_number, gate_number = self.fast_lookup.get(gate)
        return len(self.consecutive_view[line_number][group_number]) - 1 != gate_number
