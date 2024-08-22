from __future__ import annotations

import gc

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.onedit import PhyLocAdaPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import Perm


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.circuit = None

    def transpile_gate(self, gate):
        energy_graph_c = self.backend.energy_level_graphs[gate._target_qudits[0]]
        energy_graph_t = self.backend.energy_level_graphs[gate._target_qudits[1]]
        lp_map_0 = [lev for lev in energy_graph_c.log_phy_map if lev < gate._dimensions[gate._target_qudits[0]]]
        lp_map_1 = [lev for lev in energy_graph_t.log_phy_map if lev < gate._dimensions[gate._target_qudits[1]]]

        perm_0 = Perm(gate.parent_circuit, "Pm_ent_0", gate._target_qudits[0], lp_map_0, gate._dimensions[0])
        perm_1 = Perm(gate.parent_circuit, "Pm_ent_1", gate._target_qudits[1], lp_map_1, gate._dimensions[1])
        perm_0_dag = Perm(gate.parent_circuit, "Pm_ent_0", gate._target_qudits[0], lp_map_0, gate._dimensions[0]).dag()
        perm_1_dag = Perm(gate.parent_circuit, "Pm_ent_1", gate._target_qudits[1], lp_map_1, gate._dimensions[1]).dag()

        phyloc = PhyLocAdaPass(self.backend)
        perm_0_seq = phyloc.transpile_gate(perm_0)
        perm_1_seq = phyloc.transpile_gate(perm_1)
        perm_0_d_seq = phyloc.transpile_gate(perm_0_dag)
        perm_1_d_seq = phyloc.transpile_gate(perm_1_dag)

        eqr = EntangledQRCEX(gate)
        decomp, _countcr, _countpsw = eqr.execute()
        perm_0_d_seq.extend(perm_1_d_seq)
        perm_0_d_seq.extend(decomp)
        perm_0_d_seq.extend(perm_0_seq)
        perm_0_d_seq.extend(perm_1_seq)

        return [op.dag() for op in reversed(decomp)]

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                new_instructions.extend(gate_trans)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)
