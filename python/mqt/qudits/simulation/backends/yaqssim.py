# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import opt_einsum as oe
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.digital.utils.qudit_dag_utils import QuditOpNode


def apply_single_qudit_gate(state: MPS, node: QuditOpNode) -> None:
    site = node.target_qudits[0]
    U = node.gate.to_matrix()
    state.tensors[site] = oe.contract("ab,bcd->acd", U, state.tensors[site])

def apply_two_qudit_gate(state: MPS, node: QuditOpNode) -> None:
    left_site = node.target_qudits[0]
    right_site = node.target_qudits[1]
    d_left = node.dimensions[0]
    d_right = node.dimensions[1]
    U = node.gate.to_matrix()
    U = U.reshape(d_left, d_right, d_left, d_right)
    merged = merge_two_site(state.tensors[left_site], state.tensors[right_site])
    theta  = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = oe.contract("ijkl,klab->ijab", U, theta)
    merged_new = theta_new.reshape(d_left * d_right, merged.shape[1], merged.shape[2])
    new_left, new_right = split_two_site(
        merged_new,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode="discarded_weight",
        threshold=0.0,
        max_bond_dim=None,
        min_bond_dim=1,
    )
    state.tensors[left_site] = new_left
    state.tensors[right_site] = new_right


if __name__ == "__main__":
    import numpy as np
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.digital.utils.qudit_dag_utils import circuit_to_dag
    from mqt.qudits.quantum_circuit import QuantumCircuit

    d = 3 
    circuit = QuantumCircuit(2, [d, d])
    circuit.h(0) 
    dag = circuit_to_dag(circuit)

    state = MPS(length=2, physical_dimensions=[d, d], state="zeros")

    print("=== Tensoren vor dem Gate ===")
    for i, t in enumerate(state.tensors):
        print(f"Tensor[{i}]  Form: {t.shape}")
        print(t.squeeze(), "\n")  
    node = dag.front_layer()[0]
    print(f"Gate: '{node.op_name}' auf Qudit {node.target_qudits[0]}")
    print(f"Gate-Matrix:\n{np.round(node.gate.to_matrix(), 3)}\n")

    apply_single_qudit_gate(state, node)

    print("=== Tensoren nach dem Gate ===")
    for i, t in enumerate(state.tensors):
        print(f"Tensor[{i}]  Form: {t.shape}")
        print(np.round(t.squeeze(), 3), "\n")

    t0 = state.tensors[0][:, 0, :]
    t1 = state.tensors[1][:, :, 0]
    sv = np.einsum("ij,kj->ik", t0, t1).reshape(-1)
    print(f"Zustandsvektor: {np.round(sv, 3)}")