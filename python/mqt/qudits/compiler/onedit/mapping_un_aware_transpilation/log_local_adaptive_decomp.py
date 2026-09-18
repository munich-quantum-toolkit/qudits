# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import contextlib
import gc
import itertools
from typing import TYPE_CHECKING, cast

import numpy as np

from ....core import NAryTree
from ....exceptions import SequenceFoundError
from ....quantum_circuit import gates
from ....quantum_circuit.components.extensions.gate_types import GateTypes
from ... import CompilerPass
from .log_local_qr_decomp import QrDecomp

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from ....core import LevelGraph
    from ....core.dfs_tree import Node as TreeNode
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....simulation.backends.backendv2 import Backend

np.seterr(all="ignore")


class LogLocAdaPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        energy_graph_i = self.backend.energy_level_graphs[cast("int", gate.target_qudits)]

        qr = QrDecomp(gate, energy_graph_i)

        qr_decomp, algorithmic_cost, total_cost = qr.execute()

        adaptive = LogAdaptiveDecomposition(
            gate, energy_graph_i, (algorithmic_cost, total_cost), cast("int", gate.dimensions)
        )

        matrices_decomposed, best_cost, new_graph = adaptive.execute()
        if not np.isfinite(best_cost[1]):
            matrices_decomposed = qr_decomp
            new_graph = energy_graph_i
            for node in new_graph.nodes:
                new_graph.nodes[node]["phase_storage"] = 0
        if new_graph is not None:
            self.backend.energy_level_graphs[cast("int", gate.target_qudits)] = new_graph

        return matrices_decomposed

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.SINGLE:
                new_instructions += self.transpile_gate(gate)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class LogAdaptiveDecomposition:
    def __init__(
        self,
        gate: Gate,
        graph_orig: LevelGraph,
        cost_limit: tuple[float, float] = (0, 0),
        dimension: int = -1,
        z_prop: bool = False,
        max_nodes: int = 1000,
    ) -> None:
        """Initialize a search capped at max_nodes generated nodes, excluding the root.

        A zero budget only checks whether the input is already diagonal. If no
        solution is found within the node limit, execute returns an empty sequence
        and infinite costs so the compiler pass can use its QR decomposition.
        """
        if max_nodes < 0:
            msg = "max_nodes must be non-negative."
            raise ValueError(msg)
        self.max_nodes = max_nodes
        self.circuit: QuantumCircuit = gate.parent_circuit
        self.U: NDArray[np.complex128] = gate.to_matrix(identities=0)
        self.qudit_index: int = cast("int", gate.target_qudits)
        self.graph: LevelGraph = graph_orig
        self.graph.phase_storing_setup()
        self.cost_limit: tuple[float, float] = cost_limit
        self.dimension = dimension
        self.phase_propagation: bool = bool(z_prop)
        self.TREE: NAryTree = NAryTree()

    def execute(self) -> tuple[list[Gate], tuple[float, float], LevelGraph | None]:
        self.TREE.global_id_counter = 0
        self.TREE.add(
            0,
            gates.CustomOne(
                self.circuit, "CUo", self.qudit_index, np.identity(self.dimension, dtype=np.complex128), self.dimension
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
            matrices_decomposed_m, final_graph = self.z_extraction(matrices_decomposed, final_graph)

        return matrices_decomposed_m, best_cost, final_graph

    def z_extraction(
        self, decomposition: list[TreeNode], placement: LevelGraph | None
    ) -> tuple[list[Gate], LevelGraph | None]:
        matrices: list[Gate] = []

        for d in decomposition[1:]:
            # exclude the identity matrix coming from the root of the tree of solutions which is just for correctness
            matrices += d.PI_PULSES
            matrices = [*matrices, d.rotation]

        u_ = decomposition[-1].u_of_level  # take U of last elaboration which should be the diagonal matrix found

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
                phase_gate = gates.VirtRz(
                    self.circuit, "VRz", self.qudit_index, [i, np.angle(diag_u[i])], self.dimension
                )  # old version: VirtRz(np.angle(diag_U[i]), phy_n_i, dimension)
                u_ = phase_gate.to_matrix(identities=0) @ u_
                matrices.append(phase_gate)

        return matrices, placement

    def dfs(self, current_root: TreeNode) -> None:
        lower_triangle = np.tri(current_root.u_of_level.shape[0], k=-1, dtype=np.bool_)
        stack = [self._children(current_root, lower_triangle)]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
            else:
                stack.append(self._children(child, lower_triangle))

    def _children(self, current_root: TreeNode, lower_triangle: NDArray[np.bool_]) -> Iterator[TreeNode]:
        # check if close to diagonal
        ucopy = current_root.u_of_level.copy()

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

        # BEGIN SEARCH

        u_ = current_root.u_of_level

        dimension = u_.shape[0]

        subdiagonal_support = (np.abs(u_) > 1.0e-8) & lower_triangle
        column_support_sizes = np.count_nonzero(subdiagonal_support, axis=0)
        for c in range(dimension - 1):
            if not column_support_sizes[c]:
                continue
            for r, r2 in itertools.combinations(range(c, dimension), 2):
                if self.TREE.global_id_counter >= self.max_nodes:
                    break
                if not subdiagonal_support[r2, c]:
                    continue
                theta = 2 * np.arctan2(abs(u_[r2, c]), abs(u_[r, c]))
                phi = -(np.pi / 2 + np.angle(u_[r, c]) - np.angle(u_[r2, c]))

                rotation_involved = gates.R(
                    self.circuit, "R", self.qudit_index, [r, r2, theta, phi], self.dimension
                )  # R(theta, phi, r, r2, dimension)

                u_temp = rotation_involved.to_matrix(identities=0) @ u_  # matmul(rotation_involved.matrix, U_)

                # Reduce this column without reopening zeros in earlier columns.
                # Later columns may gain entries, as in a cyclic permutation.
                next_support = (np.abs(u_temp) > 1.0e-8) & lower_triangle
                if np.any(next_support[:, :c] & ~subdiagonal_support[:, :c]) or (
                    np.count_nonzero(next_support[:, c]) >= column_support_sizes[c]
                ):
                    continue

                decomp_next_step_cost = rotation_involved.cost + current_root.current_decomp_cost

                branch_condition = current_root.max_cost[1] - decomp_next_step_cost

                if branch_condition > 0 or abs(branch_condition) < 1.0e-12:
                    # if cost is better can be only candidate otherwise try them all

                    self.TREE.global_id_counter += 1
                    new_key = self.TREE.global_id_counter

                    current_root.add(
                        new_key,
                        rotation_involved,
                        u_temp,
                        None,
                        0,  # next_step_cost,
                        decomp_next_step_cost,
                        current_root.max_cost,
                        [],
                    )
                    # Explore before generating siblings that consume the node budget.
                    yield current_root.children[-1]
