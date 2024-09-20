from __future__ import annotations

import typing
from typing import Optional

from ..core.lanes import Lanes
from ..quantum_circuit.components.extensions.gate_types import GateTypes
from . import CompilerPass
from .naive_local_resynth import NaiveLocResynthOptPass
from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationOptPass, ZRemovalOptPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass

if typing.TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit
    from ..quantum_circuit.gate import Gate
    from ..simulation.backends.backendv2 import Backend


class QuditCompiler:
    passes_enabled: typing.ClassVar = {
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

    def compile(self, backend: Backend, circuit: QuantumCircuit, passes_names: list[str]) -> QuantumCircuit:
        passes_dict = {}
        new_instr = []
        # Instantiate and execute created classes
        for compiler_pass_name in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass_name]
            decomposition = compiler_pass(backend)
            if "Loc" in str(compiler_pass):
                passes_dict[GateTypes.SINGLE] = decomposition
            elif "Ent" in str(compiler_pass):
                passes_dict[GateTypes.TWO] = decomposition
            elif "Multi" in str(compiler_pass):
                passes_dict[GateTypes.MULTI] = decomposition
        for gate in circuit.instructions:
            decomposer = typing.cast(Optional[CompilerPass], passes_dict.get(gate.gate_type))
            if decomposer is not None:
                new_instructions = decomposer.transpile_gate(gate)
                new_instr.extend(new_instructions)
            else:
                new_instr.append(gate)

        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instr)

    def compile_O0(self, backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        compiled = self.compile(backend, circuit, passes)

        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        compiled.set_mapping(mappings)
        return compiled

    @staticmethod
    def compile_O1(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        phyloc = PhyLocQRPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)

        circuit = resynth.transpile(circuit)
        new_instructions = []
        for gate in circuit.instructions:
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                new_instructions.extend(ins)
            else:
                ins = phyent.transpile_gate(gate)
                new_instructions.extend(ins)
        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instructions)

    @staticmethod
    def compile_O2(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        phyent = PhyEntQRCEXPass(backend)

        lanes = Lanes(circuit)
        new_instructions = []
        for gate in circuit.instructions:
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                phyloc = PhyLocAdaPass(backend, lanes.next_is_local(gate))
                ins = phyloc.transpile_gate(gate)
                new_instructions.extend(ins)
            else:
                ins = phyent.transpile_gate(gate)
                new_instructions.extend(ins)
        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instructions)
