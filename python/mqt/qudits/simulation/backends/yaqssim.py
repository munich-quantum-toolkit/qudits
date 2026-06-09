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
from mqt.yaqs.core.data_structures.noise_model import NoiseModel as YAQSNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import WeakSimParams
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site
from mqt.yaqs.core.methods.dissipation import apply_dissipation
from mqt.yaqs.core.methods.stochastic_process import stochastic_process
from mqt.yaqs.digital.utils.qudit_dag_utils import circuit_to_dag

from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from mqt.yaqs.digital.utils.qudit_dag_utils import QuditOpNode
    from numpy.typing import NDArray

    from mqt.qudits.quantum_circuit import QuantumCircuit


def apply_single_qudit_gate(state: MPS, node: QuditOpNode) -> None:
    site = node.target_qudits[0]
    U = node.gate.to_matrix()
    state.tensors[site] = oe.contract("ab,bcd->acd", U, state.tensors[site])


def _apply_adjacent_two_qudit_gate(
    state: MPS, left_site: int, right_site: int, d_left: int, d_right: int, U: np.ndarray
) -> None:
    merged = merge_two_site(state.tensors[left_site], state.tensors[right_site])
    theta = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
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


def _apply_swap(state: MPS, site: int) -> None:
    d_left = state.tensors[site].shape[0]
    d_right = state.tensors[site + 1].shape[0]
    SWAP = np.zeros((d_left * d_right, d_left * d_right), dtype=complex)
    for k in range(d_left):
        for l in range(d_right):
            SWAP[l * d_left + k, k * d_right + l] = 1.0
    SWAP = SWAP.reshape(d_right, d_left, d_left, d_right)
    merged = merge_two_site(state.tensors[site], state.tensors[site + 1])
    theta = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = oe.contract("ijkl,klab->ijab", SWAP, theta)
    merged_new = theta_new.reshape(d_left * d_right, merged.shape[1], merged.shape[2])
    new_left, new_right = split_two_site(
        merged_new,
        [d_right, d_left],
        svd_distribution="right",
        trunc_mode="discarded_weight",
        threshold=0.0,
        max_bond_dim=None,
        min_bond_dim=1,
    )
    state.tensors[site] = new_left
    state.tensors[site + 1] = new_right


def apply_two_qudit_gate(state: MPS, node: QuditOpNode) -> None:
    left_site = node.target_qudits[0]
    right_site = node.target_qudits[1]
    d_left = node.dimensions[0]
    d_right = node.dimensions[1]
    U = node.gate.to_matrix().reshape(d_left, d_right, d_left, d_right)
    if right_site - left_site == 1:
        _apply_adjacent_two_qudit_gate(state, left_site, right_site, d_left, d_right, U)
    else:
        for s in range(left_site, right_site - 1):
            _apply_swap(state, s)
        _apply_adjacent_two_qudit_gate(state, right_site - 1, right_site, d_left, d_right, U)
        for s in range(right_site - 2, left_site - 1, -1):
            _apply_swap(state, s)


def simulate_circuit(state: MPS, circuit: QuantumCircuit, noise_model=None) -> MPS:
    sim_params = WeakSimParams(shots=1, preset="exact")
    dag = circuit_to_dag(circuit)
    while dag.op_nodes():
        for node in dag.front_layer():
            if len(node.target_qudits) == 1:
                apply_single_qudit_gate(state, node)
            else:
                apply_two_qudit_gate(state, node)

            local_noise = build_local_yaqs_noise(noise_model, node.op_name, node.target_qudits, node.dimensions)
            if local_noise is not None:
                apply_dissipation(state, local_noise, dt=1, sim_params=sim_params)
                state = stochastic_process(state, local_noise, dt=1, sim_params=sim_params, rng=None)

            dag.remove_op_node(node)
    return state


def mps_to_statevector(state: MPS) -> NDArray[np.complex128]:
    result = state.tensors[0][:, 0, :]
    for i in range(1, state.length):
        t = state.tensors[i]
        chi_i = result.shape[-1]
        result = result.reshape(-1, chi_i)
        result = np.einsum("ij,kjl->ikl", result, t).reshape(-1, t.shape[2])
    return result[:, 0]


def build_local_yaqs_noise(mqt_noise_model, gate_name, sites, dimensions):
    if mqt_noise_model is None:
        return None
    errors = mqt_noise_model.quantum_errors
    if gate_name not in errors:
        return None

    processes = []
    gate_errors = errors[gate_name]
    noise = gate_errors.get("local") or gate_errors.get("all")
    if noise:
        for site, d in zip(sites, dimensions, strict=False):
            if noise.probability_depolarizing > 0:
                X = np.zeros((d, d), dtype=complex)
                for j in range(d):
                    X[(j + 1) % d, j] = 1.0
                processes.append({
                    "name": "x",
                    "sites": [site],
                    "strength": noise.probability_depolarizing,
                    "matrix": X,
                })
            if noise.probability_dephasing > 0:
                omega = np.exp(2j * np.pi / d)
                Z = np.diag([omega**j for j in range(d)])
                processes.append({"name": "z", "sites": [site], "strength": noise.probability_dephasing, "matrix": Z})
    return YAQSNoiseModel(processes) if processes else None


class YAQSSim(Backend):
    def __init__(self, provider, name=None, description=None, **fields) -> None:
        super().__init__(provider, name=name, description=description, **fields)

    def run(self, circuit: QuantumCircuit, **options) -> Job:
        job = Job(self)
        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 50)
        self.memory = self._options.get("memory", False)
        sv, rho = self.execute(circuit, self.noise_model)
        job.set_result(JobResult(state_vector=sv, counts=[], density_matrix=rho))
        return job

    def execute(self, circuit: QuantumCircuit, noise_model=None) -> tuple:
        dims = circuit.dimensions
        dim_total = int(np.prod(dims))

        if noise_model is None or self.shots == 1:
            state = MPS(length=len(dims), physical_dimensions=dims, state="zeros")
            state = simulate_circuit(state, circuit, noise_model)
            sv = mps_to_statevector(state)
            rho = np.outer(sv, sv.conj())
            return sv.reshape(1, len(sv)), rho

        rho = np.zeros((dim_total, dim_total), dtype=complex)
        for _ in range(self.shots):
            state = MPS(length=len(dims), physical_dimensions=dims, state="zeros")
            state = simulate_circuit(state, circuit, noise_model)
            sv = mps_to_statevector(state)
            rho += np.outer(sv, sv.conj())
        rho /= self.shots
        return None, rho


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
    for _i, _t in enumerate(state.tensors):
        pass
    node = dag.front_layer()[0]

    apply_single_qudit_gate(state, node)

    for _i, _t in enumerate(state.tensors):
        pass

    t0 = state.tensors[0][:, 0, :]
    t1 = state.tensors[1][:, :, 0]
    sv = np.einsum("ij,kj->ik", t0, t1).reshape(-1)
