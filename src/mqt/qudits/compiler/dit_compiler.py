from __future__ import annotations

from ..core.lanes import Lanes
from ..quantum_circuit.components.extensions.gate_types import GateTypes
from .naive_local_resynth import NaiveLocResynthOptPass
from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationOptPass, ZRemovalOptPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass


class QuditCompiler:
    passes_enabled = {
        "PhyLocQRPass": PhyLocQRPass,
        "PhyLocAdaPass": PhyLocAdaPass,
        "LocQRPass": PhyLocQRPass,
        "LocAdaPass": PhyLocAdaPass,
        "LogLocQRPass": LogLocQRPass,
        "ZPropagationOptPass": ZPropagationOptPass,
        "ZRemovalOptPass": ZRemovalOptPass,
        "LogEntQRCEXPass": LogEntQRCEXPass,
        "PhyEntQRCEXPass": PhyEntQRCEXPass,
        "NaiveLocResynthOptPass": NaiveLocResynthOptPass,
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend, circuit, passes_names):
        passes_dict = {}
        new_instr = []
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass]
            decomposition = compiler_pass(backend)
            if "Loc" in str(compiler_pass):
                passes_dict[GateTypes.SINGLE] = decomposition
            elif "Ent" in str(compiler_pass):
                passes_dict[GateTypes.TWO] = decomposition
            elif "Multi" in str(compiler_pass):
                passes_dict[GateTypes.MULTI] = decomposition
        for gate in circuit.instructions:
            decomposer = passes_dict.get(gate.gate_type)
            new_instructions = decomposer.transpile_gate(gate)
            new_instr.extend(new_instructions)

        circuit.set_instructions(new_instr)
        circuit.set_mapping([graph.log_phy_map for graph in backend.energy_level_graphs])

        return circuit

    def compile_O0(self, backend, circuit):
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        compiled = self.compile(backend, circuit, passes)

        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        compiled.set_mapping(mappings)
        return compiled

    def compile_O1(self, backend, circuit):
        phyloc = PhyLocAdaPass(backend)
        phyent = PhyEntQRCEXPass(backend)

        lanes = Lanes(circuit)
        new_instructions = []
        for gate in circuit.instructions:
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate, lanes.next_is_local(gate))
                new_instructions.extend(ins)
            else:
                ins = phyent.transpile_gate(gate)
                new_instructions.extend(ins)
        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instructions)
