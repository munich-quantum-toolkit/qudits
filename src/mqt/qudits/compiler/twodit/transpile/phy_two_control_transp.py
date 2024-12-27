from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.compilation_minitools.local_compilation_minitools import check_lev
from mqt.qudits.core.custom_python_utils import append_to_front
from mqt.qudits.quantum_circuit.components.extensions.controls import ControlData
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.core import LevelGraph
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.quantum_circuit.gates import R
    from mqt.qudits.simulation.backends.backendv2 import Backend


class PhyEntSimplePass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit

        self.circuit = QuantumCircuit()

    def __routing(self, gate: R, graph: LevelGraph) -> tuple[list[R], R, list[R]]:
        from mqt.qudits.compiler.onedit.local_operation_swap import cost_calculator, gate_chain_condition
        from mqt.qudits.quantum_circuit.gates import R

        phi = gate.phi
        _, pi_pulses_routing, temp_placement, _, _ = cost_calculator(gate, graph, 0)

        if temp_placement.nodes[gate.lev_a]["lpmap"] > temp_placement.nodes[gate.lev_b]["lpmap"]:
            phi *= -1

        physical_rotation = R(
            self.circuit,
            "R",
            cast("int", gate.target_qudits),
            [temp_placement.nodes[gate.lev_a]["lpmap"], temp_placement.nodes[gate.lev_b]["lpmap"], gate.theta, phi],
            gate.dimensions,
        )

        physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)
        pi_backs = [
            R(
                self.circuit,
                "R",
                cast("int", gate.target_qudits),
                [pi_g.lev_a, pi_g.lev_b, pi_g.theta, -pi_g.phi],
                gate.dimensions,
            )
            for pi_g in reversed(pi_pulses_routing)
        ]

        return pi_pulses_routing, physical_rotation, pi_backs

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        assert gate.gate_type == GateTypes.TWO
        from mqt.qudits.quantum_circuit.gates import CEx, R, Rz

        self.circuit = gate.parent_circuit

        if isinstance(gate.target_qudits, int) and isinstance(gate, (R, Rz)):
            gate_controls = cast("ControlData", gate.control_info["controls"])
            indices = gate_controls.indices
            states = gate_controls.ctrl_states
            target_qudits = [*indices, gate.target_qudits]
            dimensions = [gate.parent_circuit.dimensions[i] for i in target_qudits]
        else:
            target_qudits = cast("list[int]", gate.target_qudits)
            dimensions = cast("list[int]", gate.dimensions)

        energy_graph_c = self.backend.energy_level_graphs[target_qudits[0]]
        energy_graph_t = self.backend.energy_level_graphs[target_qudits[1]]
        lp_map_0 = [check_lev(lev, dimensions[0]) for lev in energy_graph_c.log_phy_map[: dimensions[0]]]

        if isinstance(gate, CEx):
            phi = gate.phi
            ghost_rotation = R(
                self.circuit,
                "R_cex_t" + str(target_qudits[1]),
                target_qudits[1],
                [gate.lev_a, gate.lev_b, np.pi, phi],
                dimensions[1],
                None,
            )
            pi_pulses, rot, pi_backs = self.__routing(ghost_rotation, energy_graph_t)
            new_ctrl_lev = lp_map_0[gate.ctrl_lev]
            new_parameters = [rot.lev_a, rot.lev_b, new_ctrl_lev, rot.phi]
            tcex = CEx(self.circuit, "CEx_t" + str(target_qudits), target_qudits, new_parameters, dimensions, None)
            return [*pi_pulses, tcex, *pi_backs]
        if isinstance(gate, R):
            gate_controls = cast("ControlData", gate.control_info["controls"])
            indices = gate_controls.indices
            states = gate_controls.ctrl_states
            if len(indices) == 1:
                assert len(states) == 1
                ghost_rotation = R(
                    self.circuit,
                    "R_ghost_t" + str(target_qudits[1]),
                    target_qudits[1],
                    [gate.lev_a, gate.lev_b, gate.theta, gate.phi],
                    dimensions[1],
                    None,
                )
                pi_pulses, rot, pi_backs = self.__routing(ghost_rotation, energy_graph_t)
                new_ctrl_lev = lp_map_0[states[0]]
                new_parameters = [rot.lev_a, rot.lev_b, rot.theta, rot.phi]
                newr = R(
                    self.circuit,
                    "Rt" + str(target_qudits),
                    target_qudits[1],
                    new_parameters,
                    dimensions[1],
                    ControlData(indices=indices, ctrl_states=[new_ctrl_lev]),
                )
                return [*pi_pulses, newr, *pi_backs]
        elif isinstance(gate, Rz):
            gate_controls = cast("ControlData", gate.control_info["controls"])
            indices = gate_controls.indices
            states = gate_controls.ctrl_states
            if len(indices) == 1:
                assert len(states) == 1
                ghost_rotation = R(
                    self.circuit,
                    "R_ghost_t" + str(target_qudits[1]),
                    target_qudits[1],
                    [gate.lev_a, gate.lev_b, gate.phi, np.pi],
                    dimensions[1],
                    None,
                )
                pi_pulses, rot, pi_backs = self.__routing(ghost_rotation, energy_graph_t)
                new_ctrl_lev = lp_map_0[states[0]]
                new_parameters = [rot.lev_a, rot.lev_b, rot.theta]
                if (rot.theta * rot.phi) * (gate.phi) < 0:
                    new_parameters = [rot.lev_a, rot.lev_b, -rot.theta]
                newrz = Rz(
                    self.circuit,
                    "Rzt" + str(target_qudits),
                    target_qudits[1],
                    new_parameters,
                    dimensions[1],
                    ControlData(indices=indices, ctrl_states=[new_ctrl_lev]),
                )
                return [*pi_backs, newrz, *pi_pulses]
        return []

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions: list[Gate] = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.MULTI:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                # new_instructions.extend(gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
                # new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)
