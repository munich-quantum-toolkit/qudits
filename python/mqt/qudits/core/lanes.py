from __future__ import annotations

import operator
from typing import TYPE_CHECKING, cast

from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate

    LaneLookUp = dict[Gate, tuple[int, int, int]]
    CircuitGroupView = dict[int, list[list[tuple[int, Gate]]]]
    LineView = list[tuple[int, Gate]]
    CircuitView = dict[int, LineView]


class Lanes:
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.fast_lookup: LaneLookUp = {}
        self.consecutive_view: CircuitGroupView = {}
        self.circuit: QuantumCircuit = circuit
        self.instructions: LineView = self.pre_process_ops(circuit.instructions)
        self.index_dict: CircuitView = self.create_lanes()

    @staticmethod
    def pre_process_ops(instructions: list[Gate]) -> LineView:
        gates: LineView = []
        entanglement_counter = 0
        for gate in instructions:
            if gate.gate_type != GateTypes.SINGLE:
                entanglement_counter += 1
                gates.append((entanglement_counter, gate))
                # entanglement_counter += 1
            else:
                gates.append((entanglement_counter, gate))
        return gates

    def create_lanes(self) -> CircuitView:
        self.index_dict = {}
        for gate_tuple in self.instructions:
            gate = gate_tuple[1]
            if gate.gate_type == GateTypes.SINGLE:
                index: int = cast("int", gate.target_qudits)
                if index not in self.index_dict:
                    self.index_dict[index] = []
                self.index_dict[index].append(gate_tuple)
            elif gate.gate_type in {GateTypes.TWO, GateTypes.MULTI}:
                indices: list[int] = cast("list[int]", gate.target_qudits)
                for index in indices:
                    if index not in self.index_dict:
                        self.index_dict[index] = []
                    self.index_dict[index].append(gate_tuple)

        self.consecutive_view = self.find_consecutive_singles()
        return self.index_dict

    def extract_instructions(self) -> list[Gate]:
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
        instructions_as_circuit: list[Gate] = [gate_tuple[1] for gate_tuple in sorted_list]

        return instructions_as_circuit

    def extract_lane(self, qudit_line: int) -> list[Gate]:
        return [gate_tuple[1] for gate_tuple in self.index_dict[qudit_line]]

    def find_consecutive_singles(self, gates: LineView | None = None) -> CircuitGroupView:
        if gates is None:
            gates = self.instructions
        from collections import defaultdict

        consecutive_groups: CircuitGroupView = defaultdict(list)
        for gate_tuple in gates:
            gate = gate_tuple[1]
            if gate.gate_type == GateTypes.SINGLE:
                target_qudits = cast("int", gate.target_qudits)
                if consecutive_groups[target_qudits]:
                    consecutive_groups[target_qudits][-1].append(gate_tuple)
                else:
                    consecutive_groups[target_qudits] = [[gate_tuple]]
            else:
                qudits_targeted: list[int] = cast("list[int]", gate.target_qudits)
                for qudit in qudits_targeted:
                    consecutive_groups[qudit].append([gate_tuple])
                    consecutive_groups[qudit].append([])
        consecutive_groups = {
            key: [sublist for sublist in value if sublist] for key, value in consecutive_groups.items()
        }

        # Build the index
        for line_number, line_groups in consecutive_groups.items():
            for group_number, group in enumerate(line_groups):
                for gate_number, gate_tuple in enumerate(group):
                    gate = gate_tuple[1]
                    self.fast_lookup[gate] = (line_number, group_number, gate_number)

        return consecutive_groups

    def replace_gates_in_lane(self, line: int, start_index: int, end_index: int, new_gate: Gate) -> None:
        # Find the list associated with the line
        if line in self.index_dict:
            gates_of_line = self.index_dict[line]
        else:
            return  # Exit if line not found in index_dict

        # Remove objects within the specified interval [start_index, end_index]
        ordering_id = gates_of_line[start_index][0]
        objects_to_remove = [gates_of_line[i] for i in range(start_index, min(end_index + 1, len(gates_of_line)))]

        for obj in objects_to_remove:
            gates_of_line.remove(obj)

        # Add new_gate at the start_index
        gates_of_line.insert(start_index, (ordering_id, new_gate))

    def next_is_local(self, gate: Gate) -> bool:
        try:
            line_number, group_number, gate_number = self.fast_lookup[gate]
            return len(self.consecutive_view[line_number][group_number]) - 1 != gate_number
        except KeyError as err:
            msg = f"Gate {gate} not found in fast_lookup"
            raise KeyError(msg) from err
