from __future__ import annotations

import contextlib
import gc
import itertools
from random import shuffle
from typing import TYPE_CHECKING, cast

import numpy as np

from ....core import NAryTree
from ....core.custom_python_utils import append_to_front
from ....exceptions import SequenceFoundError
from ....quantum_circuit import gates
from ....quantum_circuit.components.extensions.gate_types import GateTypes
from ... import CompilerPass
from ...compilation_minitools import new_mod
from ..local_operation_swap import (
    cost_calculator,
    gate_chain_condition,
    graph_rule_ongate,
    graph_rule_update,
)
from ..mapping_aware_transpilation.phy_local_qr_decomp import PhyQrDecomp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ....core import LevelGraph
    from ....core.dfs_tree import Node as TreeNode
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....simulation.backends.backendv2 import Backend

np.seterr(all="ignore")


class PhyLocAdaPass(CompilerPass):
    def __init__(self, backend: Backend, vrz_prop: bool = False) -> None:
        super().__init__(backend)
        self.vrz_prop = vrz_prop

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        energy_graph_i = self.backend.energy_level_graphs[cast("int", gate.target_qudits)]

        qr = PhyQrDecomp(gate, energy_graph_i)

        _decomp, algorithmic_cost, total_cost = qr.execute()

        adaptive = PhyAdaptiveDecomposition(
            gate, energy_graph_i, (algorithmic_cost, total_cost), cast("int", gate.dimensions), z_prop=self.vrz_prop
        )
        (matrices_decomposed, _best_cost, new_energy_level_graph) = adaptive.execute()

        self.backend.energy_level_graphs[cast("int", gate.target_qudits)] = new_energy_level_graph
        return [op.dag() for op in reversed(matrices_decomposed)]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit: QuantumCircuit = circuit
        instructions: list[Gate] = circuit.instructions
        new_instructions: list[Gate] = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.SINGLE:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class PhyAdaptiveDecomposition:
    def __init__(
        self,
        gate: Gate,
        graph_orig: LevelGraph,
        cost_limit: tuple[float, float] | None = (0, 0),
        dimension: int | None = -1,
        z_prop: bool | None = False,
    ) -> None:
        self.circuit: QuantumCircuit = gate.parent_circuit
        self.U: NDArray = gate.to_matrix(identities=0)
        self.qudit_index: int = cast("int", gate.target_qudits)
        self.graph: LevelGraph = graph_orig
        self.graph.phase_storing_setup()
        self.cost_limit: tuple[float, float] = cast("tuple[float, float]", cost_limit)
        self.dimension: int = cast("int", dimension)
        self.phase_propagation: bool = cast("bool", z_prop)
        self.TREE: NAryTree = NAryTree()

    def execute(self) -> tuple[list[Gate], tuple[float, float], LevelGraph]:
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

        with contextlib.suppress(SequenceFoundError):
            self.dfs(self.TREE.root)

        matrices_decomposed, best_cost, final_graph = self.TREE.retrieve_decomposition(self.TREE.root)
        matrices_decomposed_m: list[Gate] = []
        if matrices_decomposed:
            matrices_decomposed_m, final_graph = self.z_extraction(
                matrices_decomposed, final_graph, self.phase_propagation
            )

        return matrices_decomposed_m, best_cost, final_graph

    def z_extraction(
        self, decomposition: list[TreeNode], placement: LevelGraph, phase_propagation: bool
    ) -> tuple[list[Gate], LevelGraph]:
        matrices: list[Gate] = []

        for d in decomposition[1:]:
            # exclude the identity matrix coming from the root of the tree of solutions which is just for correctness
            matrices += d.PI_PULSES
            matrices = [*matrices, d.rotation]

        u_ = decomposition[-1].u_of_level  # take U of last elaboration which should be the diagonal matrix found
        u_.round(4)
        # check if close to diagonal
        ucopy = u_.copy()

        # check if the diagonal is made only of noise
        valid_diag = any(abs(np.diag(ucopy)) > 1.0e-4)  # > 1.0e-4

        # are the non diagonal entries zeroed-out
        filtered_ucopy = abs(ucopy) > 1.0e-4
        np.fill_diagonal(filtered_ucopy, 0)

        not_diag = filtered_ucopy.any()

        if not_diag or not valid_diag:  # if is diagonal enough then somehow signal end of algorithm
            msg = "Matrix isn't close to diagonal!"
            raise RuntimeError(msg)
        diag_u = np.diag(u_)
        dimension = u_.shape[0]

        for i in range(dimension):
            if abs(np.angle(diag_u[i])) > 1.0e-4:
                if phase_propagation:
                    inode = placement.fst_inode
                    if "phase_storage" in placement.nodes[inode]:
                        placement.nodes[i]["phase_storage"] += np.angle(diag_u[i])
                        placement.nodes[i]["phase_storage"] = new_mod(placement.nodes[i]["phase_storage"])
                else:
                    phy_n_i = placement.nodes[i]["lpmap"]
                    angle_phy = new_mod(np.angle(diag_u[i]))
                    phase_gate = gates.VirtRz(
                        self.circuit, "VRz", self.qudit_index, [phy_n_i, angle_phy], self.dimension
                    )  # old version: VirtRz(np.angle(diag_U[i]), phy_n_i, dimension)
                    log_phase_gate = gates.VirtRz(self.circuit, "VRz", self.qudit_index, [i, angle_phy], self.dimension)
                    log_phase_gate.to_matrix(identities=0)
                    u_ = log_phase_gate.to_matrix(identities=0) @ u_
                    u_.round(4)

                    matrices.append(phase_gate)

        if not phase_propagation:
            inode = placement.fst_inode
            if "phase_storage" in placement.nodes[inode]:
                for i in range(len(list(placement.nodes))):
                    theta_z = new_mod(placement.nodes[i]["phase_storage"])
                    if abs(theta_z) > 1.0e-4:
                        phase_gate = gates.VirtRz(
                            self.circuit,
                            "VRz",
                            self.qudit_index,
                            [placement.nodes[i]["lpmap"], theta_z],
                            self.dimension,
                        )  # VirtRz(thetaZ, placement.nodes[i]['lpmap'],
                        # dimension)
                        matrices.append(phase_gate)
                    # reset the node
                    placement.nodes[i]["phase_storage"] = 0

        return matrices, placement

    def dfs(self, current_root: TreeNode, level: int = 0) -> None:
        # check if close to diagonal
        ucopy = current_root.u_of_level.copy()

        current_placement = current_root.graph

        # is the diagonal noisy?
        valid_diag = any(abs(np.diag(ucopy)) > 1.0e-4)

        # are the non diagonal entries zeroed-out?
        filtered_ucopy = abs(ucopy) > 1.0e-4
        np.fill_diagonal(filtered_ucopy, 0)

        not_diag = filtered_ucopy.any()

        # if is diagonal enough then somehow signal end of algorithm
        if (not not_diag) and valid_diag:
            current_root.finished = True

            raise SequenceFoundError(current_root.key)

        u_ = current_root.u_of_level

        dimension = u_.shape[0]
        for c, r, r2 in itertools.product(range(dimension), range(dimension), range(dimension)):
            if r < c or r2 <= r:
                continue
            if abs(u_[r2, c]) > 1.0e-8 and (abs(u_[r, c]) > 1.0e-18 or abs(u_[r, c]) == 0):
                theta = 2 * np.arctan2(abs(u_[r2, c]), abs(u_[r, c]))

                phi = -(np.pi / 2 + np.angle(u_[r, c]) - np.angle(u_[r2, c]))

                rotation_involved = gates.R(
                    self.circuit, "R", self.qudit_index, [r, r2, theta, phi], self.dimension
                )  # R(theta, phi, r, r2, dimension)

                u_temp = rotation_involved.to_matrix(identities=0) @ u_  # matmul(rotation_involved.matrix, U_)
                u_temp.round(4)
                non_zeros = np.count_nonzero(abs(u_temp) > 1.0e-4)

                (
                    estimated_cost,
                    pi_pulses_routing,
                    new_placement,
                    cost_of_pi_pulses,
                    gate_cost,
                ) = cost_calculator(rotation_involved, current_placement, non_zeros)

                next_step_cost = estimated_cost + current_root.current_cost
                decomp_next_step_cost = cost_of_pi_pulses + gate_cost + current_root.current_decomp_cost

                branch_condition = current_root.max_cost[1] - decomp_next_step_cost  # SECOND POSITION IS PHYSICAL COST
                if branch_condition > 0 or abs(branch_condition) < 1.0e-12:
                    # if cost is better can be only candidate otherwise try them all
                    self.TREE.global_id_counter += 1
                    new_key = self.TREE.global_id_counter

                    if new_placement.nodes[r]["lpmap"] > new_placement.nodes[r2]["lpmap"]:
                        phi *= -1
                    physical_rotation = gates.R(
                        self.circuit,
                        "R",
                        self.qudit_index,
                        [new_placement.nodes[r]["lpmap"], new_placement.nodes[r2]["lpmap"], theta, phi],
                        self.dimension,
                    )
                    physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)
                    physical_rotation = graph_rule_ongate(physical_rotation, new_placement)

                    p_backs = [
                        gates.R(
                            self.circuit,
                            "R",
                            self.qudit_index,
                            [ppulse.lev_a, ppulse.lev_b, ppulse.theta, -ppulse.phi],
                            self.dimension,
                        )
                        for ppulse in pi_pulses_routing
                    ]

                    for p_back in p_backs:
                        graph_rule_update(p_back, new_placement)

                    current_root.add(
                        new_key,
                        physical_rotation,
                        u_temp,
                        new_placement,
                        next_step_cost,
                        decomp_next_step_cost,
                        current_root.max_cost,
                        pi_pulses_routing,
                    )

        def calculate_sparsity(matrix: NDArray) -> float:
            total_elements = matrix.size
            non_zero_elements = np.count_nonzero(np.abs(matrix) > 1e-8)  # np.count_nonzero(matrix)
            return cast("float", non_zero_elements / total_elements)

        def change_kids(lst: list[TreeNode]) -> list[TreeNode]:
            # Check if the list is non-empty
            if not lst:
                return lst
            vals = [calculate_sparsity(obj.u_of_level) for obj in lst]
            # Calculate the range of values in the list
            min_val, _max_val = min(vals), max(vals)
            min_from_one = 1 - min_val
            dim = lst[0].u_of_level.shape[0]
            threshold = (dim - 3) ** 2 / dim**2
            # If the spread is below the threshold, shuffle the list
            if min_from_one < threshold and len(lst) > dim**2 * 0.6:
                shuffle(lst)
            else:
                lst = sorted(lst, key=lambda obj: calculate_sparsity(obj.u_of_level))

            return lst

        if current_root.children is not None:
            new_kids = change_kids(current_root.children)
            for child in new_kids:
                self.dfs(child, level + 1)
