from __future__ import annotations

import typing
from typing import Optional

from ..core.custom_python_utils import append_to_front
from ..quantum_circuit.components.extensions.gate_types import GateTypes
from . import CompilerPass
from .multidit.transpile.phy_multi_control_transp import PhyMultiSimplePass
from .naive_local_resynth import NaiveLocResynthOptPass
from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationOptPass, ZRemovalOptPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass
from .twodit.transpile.phy_two_control_transp import PhyEntSimplePass

if typing.TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit
    from ..quantum_circuit.gate import Gate
    from ..simulation.backends.backendv2 import Backend
    from . import CompilerPass


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
        "PhyEntSimplePass": PhyEntSimplePass,
        "PhyMultiSimplePass": PhyMultiSimplePass,
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend: Backend, circuit: QuantumCircuit, passes_names: list[str]) -> QuantumCircuit:
        """Method compiles with passes chosen."""
        transpiled_circuit = circuit.copy()
        final_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_final_mappings(final_mappings)

        passes_dict = {}
        new_instr: list[Gate] = []
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

        for gate in reversed(circuit.instructions):
            decomposer = typing.cast("Optional[CompilerPass]", passes_dict.get(gate.gate_type))
            if decomposer is not None:
                new_instructions = decomposer.transpile_gate(gate)
                append_to_front(new_instr, new_instructions)
            else:
                append_to_front(new_instr, gate)
        initial_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_initial_mappings(initial_mappings)
        return transpiled_circuit.set_instructions(new_instr)

    def compile_O0(self, backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """Method compiles with PHY LOC QR and PHY ENT QR CEX with no optimization."""
        final_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])

        passes = ["PhyLocQRPass", "PhyEntQRCEXPass", "PhyMultiSimplePass"]
        compiled = self.compile(backend, circuit, passes)

        initial_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        compiled.set_initial_mappings(initial_mappings)
        compiled.set_final_mappings(final_mappings)
        return compiled

    @staticmethod
    def compile_O1_resynth(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """Method compiles with PHY LOC QR and PHY ENT QR CEX with a resynth steps."""
        phyloc = PhyLocQRPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)
        phymulti = PhyMultiSimplePass(backend)

        transpiled_circuit = circuit.copy()
        final_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_final_mappings(final_mappings)

        circuit = resynth.transpile(circuit)
        new_instructions: list[Gate] = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            elif gate.gate_type is GateTypes.TWO:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phymulti.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        initial_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_initial_mappings(initial_mappings)
        return transpiled_circuit.set_instructions(new_instructions)

    @staticmethod
    def compile_O1_adaptive(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """Method compiles with PHY LOC ADA and PHY ENT QR CEX."""
        phyent = PhyEntQRCEXPass(backend)
        phyloc = PhyLocAdaPass(backend)
        phymulti = PhyMultiSimplePass(backend)
        transpiled_circuit = circuit.copy()
        final_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])

        new_instructions: list[Gate] = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            elif gate.gate_type is GateTypes.TWO:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phymulti.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        transpiled_circuit.set_instructions(new_instructions)
        z_propagation_pass = ZPropagationOptPass(backend=backend, back=False)
        transpiled_circuit = z_propagation_pass.transpile(transpiled_circuit)

        initial_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_initial_mappings(initial_mappings)
        transpiled_circuit.set_final_mappings(final_mappings)

        return transpiled_circuit

    @staticmethod
    def compile_O2(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """Method compiles with PHY LOC ADA and PHY ENT QR CEX with a resynth steps."""
        phyloc = PhyLocAdaPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)
        phymulti = PhyMultiSimplePass(backend)
        transpiled_circuit = circuit.copy()

        final_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                final_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])

        circuit = resynth.transpile(circuit)
        new_instructions: list[Gate] = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            elif gate.gate_type is GateTypes.TWO:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phymulti.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        transpiled_circuit.set_instructions(new_instructions)
        z_propagation_pass = ZPropagationOptPass(backend=backend, back=False)
        transpiled_circuit = z_propagation_pass.transpile(transpiled_circuit)

        initial_mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                initial_mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_initial_mappings(initial_mappings)
        transpiled_circuit.set_final_mappings(final_mappings)
        return transpiled_circuit.set_instructions(new_instructions)
