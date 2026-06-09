# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
from mqt.yaqs.digital.utils.qudit_dag_utils import circuit_to_dag
from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from mqt.qudits.quantum_circuit import QuantumCircuit
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

def simulate_circuit(state: MPS, circuit: QuantumCircuit) -> None:
    dag = circuit_to_dag(circuit)
    while dag.op_nodes():
        for node in dag.front_layer():
            if len(node.target_qudits) == 1:
                apply_single_qudit_gate(state, node)
            else:
                apply_two_qudit_gate(state, node)
            dag.remove_op_node(node)

def mps_to_statevector(state: MPS) -> NDArray[np.complex128]:
    result = state.tensors[0][:, 0, :]
    for i in range(1, state.length):
        t = state.tensors[i]
        chi_i = result.shape[-1]
        result = result.reshape(-1, chi_i)
        result = np.einsum("ij,kjl->ikl", result, t).reshape(-1, t.shape[2])
    return result[:, 0]

class YAQSSim(Backend):
    def __init__(self, provider, name=None, description=None, **fields):
        super().__init__(provider, name=name, description=description, **fields)
    
    def run(self, circuit: QuantumCircuit, **options) -> Job:
        job = Job(self)
        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 50)
        self.memory = self._options.get("memory", False)
        job.set_result(JobResult(state_vector=self.execute(circuit), counts=[]))
        return job
    
    def execute(self, circuit: QuantumCircuit, noise_model=None) -> NDArray[np.complex128]:
        dims = circuit.dimensions
        state = MPS(length=len(dims), physical_dimensions=dims, state="zeros")
        simulate_circuit(state, circuit)
        sv = mps_to_statevector(state)
        return sv.reshape(1, len(sv))

if __name__ == "__main__":
    import numpy as np
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.digital.utils.qudit_dag_utils import circuit_to_dag
    from mqt.qudits.quantum_circuit import QuantumCircuit

    d = 2 
    circuit = QuantumCircuit(2, [d, d])
    circuit.h(0) 
    dag = circuit_to_dag(circuit)

    state = MPS(length=2, physical_dimensions=[d, d], state="zeros")
    print(f"State {state}")
    for i, t in enumerate(state.tensors):
        print(f"Tensor[{i}]  Form: {t.shape}")
        print(t.squeeze())  
    node = dag.front_layer()[0]
    print(f"Gate-Matrix:\n{np.round(node.gate.to_matrix(), 3)}\n")

    apply_single_qudit_gate(state, node)

    for i, t in enumerate(state.tensors):
        print(f"Tensor[{i}]  Form: {t.shape}")
        print(np.round(t.squeeze(), 3), "\n")

    t0 = state.tensors[0][:, 0, :]
    t1 = state.tensors[1][:, :, 0]
    sv = np.einsum("ij,kj->ik", t0, t1).reshape(-1)
    print(f"Zustandsvektor: {np.round(sv, 3)}")