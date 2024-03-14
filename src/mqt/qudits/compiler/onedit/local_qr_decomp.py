import gc

import numpy as np

from mqt.qudits.compiler.compiler_pass import CompilerPass
from mqt.qudits.compiler.onedit.local_operation_swap.swap_routine import cost_calculator, gate_chain_condition
from mqt.qudits.compiler.onedit.local_rotation_tools.local_compilation_minitools import new_mod
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.gate_set.r import R
from mqt.qudits.qudit_circuits.components.instructions.gate_set.virt_rz import VirtRz


class LocQRPass(CompilerPass):
    def __init__(self, backend):
        super().__init__(backend)

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for _i, gate in enumerate(instructions):
            if gate.gate_type == GateTypes.SINGLE:
                energy_graph_i = self.backend.energy_level_graphs[gate._target_qudits]
                QR = QrDecomp(gate, energy_graph_i, not_stand_alone=False)
                decomp, algorithmic_cost, total_cost = QR.execute()
                new_instructions += decomp
                gc.collect()
            else:
                new_instructions.append(gate)  # TODO REENCODING
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class QrDecomp:
    def __init__(self, gate, graph_orig, Z_prop=False, not_stand_alone=True):
        self.gate = gate
        self.circuit = gate.parent_circuit
        self.dimension = gate._dimensions
        self.qudit_index = gate._target_qudits
        self.U = gate.to_matrix(identities=0)
        self.graph = graph_orig
        self.phase_propagation = Z_prop
        self.not_stand_alone = not_stand_alone

    def execute(self):
        decomp = []
        total_cost = 0
        algorithmic_cost = 0

        U_ = self.U
        dimension = self.U.shape[0]
        #
        # GRAPH PHASES - REMOVE ANY REMAINING AND SAVE FOR RESTORING AT THE END OF ALGORITHM
        if not self.phase_propagation:
            recover_dict = {}
            inode = self.graph._1stInode
            if "phase_storage" in self.graph.nodes[inode]:
                for i in range(len(list(self.graph.nodes))):
                    thetaZ = new_mod(self.graph.nodes[i]["phase_storage"])
                    if abs(thetaZ) > 1.0e-4:
                        phase_gate = VirtRz(
                            self.gate.parent_circuit,
                            "VRz",
                            self.gate._target_qudits,
                            [self.graph.nodes[i]["lpmap"], thetaZ],
                            self.gate.dimension,
                        )  # (thetaZ, self.graph.nodes[i]['lpmap'], dimension)
                        decomp.append(phase_gate)
                    recover_dict[i] = thetaZ

                    # reset the node
                    self.graph.nodes[i]["phase_storage"] = 0
        #

        l = list(range(self.U.shape[0]))
        l.reverse()

        for c in range(self.U.shape[1]):
            diag_index = l.index(c)

            for r in l[:diag_index]:
                if abs(U_[r, c]) > 1.0e-8:
                    theta = 2 * np.arctan2(abs(U_[r, c]), abs(U_[r - 1, c]))

                    phi = -(np.pi / 2 + np.angle(U_[r - 1, c]) - np.angle(U_[r, c]))

                    rotation_involved = R(
                        self.circuit, "R", self.qudit_index, [r - 1, r, theta, phi], self.dimension
                    )  # R(theta, phi, r - 1, r, dimension)

                    U_ = rotation_involved.to_matrix(identities=0) @ U_  # matmul(rotation_involved.matrix, U_)

                    non_zeros = np.count_nonzero(abs(U_) > 1.0e-4)

                    estimated_cost, pi_pulses_routing, temp_placement, cost_of_pi_pulses, gate_cost = cost_calculator(
                        rotation_involved, self.graph, non_zeros
                    )

                    decomp += pi_pulses_routing

                    if temp_placement.nodes[r - 1]["lpmap"] > temp_placement.nodes[r]["lpmap"]:
                        phi = phi * -1

                    physical_rotation = R(
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
                            R(
                                self.circuit,
                                "R",
                                self.qudit_index,
                                [pi_g.lev_a, pi_g.lev_b, pi_g.theta, -pi_g.phi],
                                self.dimension,
                            )
                        )  # R(pi_g.theta, -pi_g.phi, pi_g.lev_a, pi_g.lev_b, dimension))
                    pi_g = None

                    algorithmic_cost += estimated_cost
                    total_cost += 2 * cost_of_pi_pulses + gate_cost

        diag_U = np.diag(U_)

        for i in range(dimension):
            if abs(np.angle(diag_U[i])) > 1.0e-4:
                phy_n_i = self.graph.nodes[i]["lpmap"]

                phase_gate = VirtRz(
                    self.gate.parent_circuit,
                    "VRz",
                    self.gate._target_qudits,
                    [phy_n_i, np.angle(diag_U[i])],
                    self.gate._dimensions,
                )  # Rz(np.angle(diag_U[i]), phy_n_i, dimension)

                decomp.append(phase_gate)

        if self.not_stand_alone and not self.phase_propagation:
            inode = self.graph._1stInode
            if "phase_storage" in self.graph.nodes[inode]:
                for i in range(len(list(self.graph.nodes))):
                    self.graph.nodes[i]["phase_storage"] = recover_dict[i]

        return decomp, algorithmic_cost, total_cost
