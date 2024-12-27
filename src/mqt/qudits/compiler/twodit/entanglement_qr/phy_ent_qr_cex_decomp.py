from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.core.custom_python_utils import append_to_front
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit

        self.circuit = QuantumCircuit()

    def __transpile_local_ops(self, gate: Gate) -> list[Gate]:
        from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyQrDecomp

        energy_graph_i = self.backend.energy_level_graphs[cast("int", gate.target_qudits)]
        qr = PhyQrDecomp(gate, energy_graph_i, not_stand_alone=False)
        decomp, _algorithmic_cost, _total_cost = qr.execute()
        return decomp

    @staticmethod
    def __transpile_two_ops(backend: Backend, gate: Gate) -> tuple[bool, list[Gate]]:
        assert gate.gate_type == GateTypes.TWO
        from mqt.qudits.compiler.twodit.transpile.phy_two_control_transp import PhyEntSimplePass

        phy_two_simple = PhyEntSimplePass(backend)
        transpiled = phy_two_simple.transpile_gate(gate)
        return (len(transpiled) > 0), transpiled

    def transpile_gate(self, orig_gate: Gate) -> list[Gate]:
        simple_gate, simple_gate_decomp = self.__transpile_two_ops(self.backend, orig_gate)
        if simple_gate:
            return simple_gate_decomp

        eqr = EntangledQRCEX(orig_gate)
        decomp, _countcr, _countpsw = eqr.execute()

        # Full sequence of logical operations to be implemented to reconstruct
        # the logical operation on the device
        full_logical_sequence = [op.dag() for op in reversed(decomp)]

        # Actual implementation of the gate in the device based on the mapping
        physical_sequence: list[Gate] = []
        for gate in reversed(full_logical_sequence):
            if gate.gate_type == GateTypes.SINGLE:
                loc_gate = self.__transpile_local_ops(gate)
                append_to_front(physical_sequence, [op.dag() for op in reversed(loc_gate)])
            elif gate.gate_type == GateTypes.TWO:
                _, ent_gate = self.__transpile_two_ops(self.backend, gate)
                append_to_front(physical_sequence, ent_gate)
            elif gate.gate_type == GateTypes.MULTI:
                msg = "Multi not supposed to be in decomposition!"
                raise RuntimeError(msg)

        return physical_sequence

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions: list[Gate] = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                # new_instructions.extend(gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
                # new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)
