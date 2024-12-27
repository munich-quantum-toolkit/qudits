from __future__ import annotations

import copy
import gc
from typing import TYPE_CHECKING, cast

import numpy as np

from ....quantum_circuit import gates
from ....quantum_circuit.components.extensions.gate_types import GateTypes
from ... import CompilerPass
from ...compilation_minitools import new_mod
from ..local_operation_swap import cost_calculator, gate_chain_condition

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ....core import LevelGraph
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....simulation.backends.backendv2 import Backend


class PhyLocQRPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        energy_graph_i = self.backend.energy_level_graphs[cast("int", gate.target_qudits)]
        qr = PhyQrDecomp(gate, energy_graph_i, not_stand_alone=False)
        decomp, _algorithmic_cost, _total_cost = qr.execute()
        return [op.dag() for op in reversed(decomp)]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.SINGLE:
                gate_trans = self.transpile_gate(gate)
                gate_trans = [op.dag() for op in reversed(gate_trans)]
                new_instructions.extend(gate_trans)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class PhyQrDecomp:
    def __init__(self, gate: Gate, graph_orig: LevelGraph, z_prop: bool = False, not_stand_alone: bool = True) -> None:
        self.gate: Gate = gate
        self.circuit: QuantumCircuit = gate.parent_circuit
        self.dimension: int = cast("int", gate.dimensions)
        self.qudit_index: int = cast("int", gate.target_qudits)
        self.U: NDArray = gate.to_matrix(identities=0)
        self.graph: LevelGraph = copy.deepcopy(graph_orig)
        self.phase_propagation: bool = z_prop
        self.not_stand_alone: bool = not_stand_alone

    def execute(self) -> tuple[list[Gate], float, float]:
        decomp: list[Gate] = []
        total_cost = 0.0
        algorithmic_cost = 0.0

        u_ = self.U
        dimension = self.U.shape[0]
        #
        # GRAPH PHASES - REMOVE ANY REMAINING AND SAVE FOR RESTORING AT THE END OF ALGORITHM
        if not self.phase_propagation:
            recover_dict = {}
            inode = self.graph.fst_inode
            if "phase_storage" in self.graph.nodes[inode]:
                for i in range(len(list(self.graph.nodes))):
                    theta_z = new_mod(self.graph.nodes[i]["phase_storage"])
                    if abs(theta_z) > 1.0e-4:
                        phase_gate = gates.VirtRz(
                            self.gate.parent_circuit,
                            "VRz",
                            cast("int", self.gate.target_qudits),
                            [self.graph.nodes[i]["lpmap"], theta_z],
                            cast("int", self.gate.dimensions),
                        )  # (thetaZ, self.graph.nodes[i]['lpmap'], dimension)
                        decomp.append(phase_gate)
                    recover_dict[i] = theta_z

                    # reset the node
                    self.graph.nodes[i]["phase_storage"] = 0

        dim_iterator = list(range(self.U.shape[0]))
        dim_iterator.reverse()

        for c in range(self.U.shape[1]):
            diag_index = dim_iterator.index(c)

            for r in dim_iterator[:diag_index]:
                if abs(u_[r, c]) > 1.0e-8:
                    theta = 2 * np.arctan2(abs(u_[r, c]), abs(u_[r - 1, c]))

                    phi = -(np.pi / 2 + np.angle(u_[r - 1, c]) - np.angle(u_[r, c]))

                    rotation_involved = gates.R(
                        self.circuit, "R", self.qudit_index, [r - 1, r, theta, phi], self.dimension
                    )  # R(theta, phi, r - 1, r, dimension)

                    u_ = rotation_involved.to_matrix(identities=0) @ u_  # matmul(rotation_involved.matrix, U_)

                    non_zeros = np.count_nonzero(abs(u_) > 1.0e-4)

                    estimated_cost, pi_pulses_routing, temp_placement, cost_of_pi_pulses, gate_cost = cost_calculator(
                        rotation_involved, self.graph, non_zeros
                    )

                    decomp += pi_pulses_routing

                    if temp_placement.nodes[r - 1]["lpmap"] > temp_placement.nodes[r]["lpmap"]:
                        phi *= -1

                    physical_rotation = gates.R(
                        self.circuit,
                        "R",
                        self.qudit_index,
                        [temp_placement.nodes[r - 1]["lpmap"], temp_placement.nodes[r]["lpmap"], theta, phi],
                        self.dimension,
                    )
                    # R(theta, phi, temp_placement.nodes[r - 1]['lpmap'], temp_placement.nodes[r]['lpmap'], dimension)
                    physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)

                    decomp.append(physical_rotation)

                    for pi_g in reversed(pi_pulses_routing):
                        decomp.append(
                            gates.R(
                                self.circuit,
                                "R",
                                self.qudit_index,
                                [pi_g.lev_a, pi_g.lev_b, pi_g.theta, -pi_g.phi],
                                self.dimension,
                            )
                        )  # R(pi_g.theta, -pi_g.phi, pi_g.lev_a, pi_g.lev_b, dimension))

                    algorithmic_cost += estimated_cost
                    total_cost += 2 * cost_of_pi_pulses + gate_cost

        diag_u = np.diag(u_)

        for i in range(dimension):
            if abs(np.angle(diag_u[i])) > 1.0e-4:
                phy_n_i = self.graph.nodes[i]["lpmap"]

                phase_gate = gates.VirtRz(
                    self.gate.parent_circuit,
                    "VRz",
                    cast("int", self.gate.target_qudits),
                    [phy_n_i, np.angle(diag_u[i])],
                    cast("int", self.gate.dimensions),
                )  # Rz(np.angle(diag_U[i]), phy_n_i, dimension)

                decomp.append(phase_gate)

        if self.not_stand_alone and not self.phase_propagation:
            inode = self.graph.fst_inode
            if "phase_storage" in self.graph.nodes[inode]:
                for i in range(len(list(self.graph.nodes))):
                    self.graph.nodes[i]["phase_storage"] = recover_dict[i]

        return decomp, algorithmic_cost, total_cost
