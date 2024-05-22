from __future__ import annotations

import gc

import numpy as np

from ....core import NAryTree
from ....exceptions import SequenceFoundException
from ....quantum_circuit import gates
from ....quantum_circuit.components.extensions.gate_types import GateTypes
from ... import CompilerPass
from ..mapping_aware_transpilation import PhyQrDecomp

np.seterr(all="ignore")


class LogLocAdaPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.SINGLE:
                energy_graph_i = self.backend.energy_level_graphs[gate._target_qudits]

                QR = PhyQrDecomp(gate, energy_graph_i)

                _decomp, algorithmic_cost, total_cost = QR.execute()

                Adaptive = LogAdaptiveDecomposition(
                    gate, energy_graph_i, (algorithmic_cost, total_cost), gate._dimensions
                )

                (
                    matrices_decomposed,
                    _best_cost,
                    self.backend.energy_level_graphs[gate._target_qudits],
                ) = Adaptive.execute()

                new_instructions += matrices_decomposed

                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class LogAdaptiveDecomposition:
    def __init__(self, gate, graph_orig, cost_limit=(0, 0), dimension=-1, Z_prop=False) -> None:
        self.circuit = gate.parent_circuit
        self.U = gate.to_matrix(identities=0)
        self.qudit_index = gate._target_qudits
        self.graph = graph_orig
        self.graph.phase_storing_setup()
        self.cost_limit = cost_limit
        self.dimension = dimension
        self.phase_propagation = Z_prop
        self.TREE = NAryTree()

    def execute(self):
        self.TREE.add(
            0,
            gates.CustomOne(
                self.circuit, "CUo", self.qudit_index, np.identity(self.dimension, dtype="complex"), self.dimension
            ),
            self.U,
            self.graph,
            0,
            0,
            self.cost_limit,
            [],
        )
        try:
            self.DFS(self.TREE.root)
        except SequenceFoundException:
            pass
        finally:
            matrices_decomposed, best_cost, final_graph = self.TREE.retrieve_decomposition(self.TREE.root)

            if matrices_decomposed != []:
                matrices_decomposed, final_graph = self.z_extraction(
                    matrices_decomposed, final_graph, self.phase_propagation
                )
            else:
                pass

            self.TREE.print_tree(self.TREE.root, "TREE: ")

            return matrices_decomposed, best_cost, final_graph

    def z_extraction(self, decomposition, placement, phase_propagation):
        matrices = []

        for d in decomposition[1:]:
            # exclude the identity matrix coming from the root of the tree of solutions which is just for correctness
            matrices += d.PI_PULSES
            matrices = [*matrices, d.rotation]

        U_ = decomposition[-1].U_of_level  # take U of last elaboration which should be the diagonal matrix found

        # check if close to diagonal
        Ucopy = U_.copy()

        # check if the diagonal is made only of noise
        valid_diag = any(abs(np.diag(Ucopy)) > 1.0e-4)  # > 1.0e-4

        # are the non diagonal entries zeroed-out
        filtered_Ucopy = abs(Ucopy) > 1.0e-4
        np.fill_diagonal(filtered_Ucopy, 0)

        not_diag = filtered_Ucopy.any()

        if not_diag or not valid_diag:  # if is diagonal enough then somehow signal end of algorithm
            msg = "Matrix isn't close to diagonal!"
            raise Exception(msg)
        diag_U = np.diag(U_)
        dimension = U_.shape[0]

        for i in range(dimension):
            if abs(np.angle(diag_U[i])) > 1.0e-4:
                phase_gate = gates.VirtRz(
                    self.circuit, "VRz", self.qudit_index, [i, np.angle(diag_U[i])], self.dimension
                )  # old version: VirtRz(np.angle(diag_U[i]), phy_n_i, dimension)
                U_ = phase_gate.to_matrix(identities=0) @ U_
                matrices.append(phase_gate)

        return matrices, placement

    def DFS(self, current_root, level=0) -> None:
        # check if close to diagonal
        Ucopy = current_root.U_of_level.copy()

        # is the diagonal noisy?
        valid_diag = any(abs(np.diag(Ucopy)) > 1.0e-4)

        # are the non diagonal entries zeroed-out?
        filtered_Ucopy = abs(Ucopy) > 1.0e-4
        np.fill_diagonal(filtered_Ucopy, 0)

        not_diag = filtered_Ucopy.any()

        # if is diagonal enough then somehow signal end of algorithm
        if (not not_diag) and valid_diag:
            current_root.finished = True

            raise SequenceFoundException(current_root.key)

        ################################################
        ###############
        #########

        # BEGIN SEARCH

        U_ = current_root.U_of_level

        dimension = U_.shape[0]

        for c in range(dimension):
            for r in range(c, dimension):
                for r2 in range(r + 1, dimension):
                    if abs(U_[r2, c]) > 1.0e-8 and (abs(U_[r, c]) > 1.0e-18 or abs(U_[r, c]) == 0):
                        theta = 2 * np.arctan2(abs(U_[r2, c]), abs(U_[r, c]))
                        phi = -(np.pi / 2 + np.angle(U_[r, c]) - np.angle(U_[r2, c]))

                        rotation_involved = gates.R(
                            self.circuit, "R", self.qudit_index, [r, r2, theta, phi], self.dimension
                        )  # R(theta, phi, r, r2, dimension)

                        U_temp = rotation_involved.to_matrix(identities=0) @ U_  # matmul(rotation_involved.matrix, U_)

                        decomp_next_step_cost = rotation_involved.cost + current_root.current_decomp_cost

                        branch_condition = current_root.max_cost[1] - decomp_next_step_cost

                        if branch_condition > 0 or abs(branch_condition) < 1.0e-12:
                            # if cost is better can be only candidate otherwise try them all

                            self.TREE.global_id_counter += 1
                            new_key = self.TREE.global_id_counter

                            current_root.add(
                                new_key,
                                rotation_involved,
                                U_temp,
                                None,  # new_placement,
                                0,  # next_step_cost,
                                decomp_next_step_cost,
                                current_root.max_cost,
                                [],
                            )

        # ===============CONTINUE SEARCH ON CHILDREN========================================
        if current_root.children is not None:
            for child in current_root.children:
                self.DFS(child, level + 1)
        # ===================================================================================

        # END OF RECURSION#
