from __future__ import annotations

import numpy as np

"""def calculate_q0_q1(lev, dim):
    q1 = lev % dim  # Calculate the remainder
    q0 = (lev - q1) // dim  # Calculate the quotient
    return q0, q1"""


def on1(gate, other_size):
    return np.kron(np.identity(other_size, dtype="complex"), gate)


def on0(gate, other_size):
    return np.kron(gate, np.identity(other_size, dtype="complex"))


def gate_expand_to_circuit(gate, circuits_size, target, dims=None):
    if dims is None:
        dims = [2, 2]
    if circuits_size < 1:
        msg = "integer circuits_size must be larger or equal to 1"
        raise ValueError(msg)

    if target >= circuits_size:
        msg = "target must be integer < integer circuits_size"
        raise ValueError(msg)

    upper = [np.identity(dims[i], dtype="complex") for i in range(circuits_size - target - 1)]
    lower = [np.identity(dims[j], dtype="complex") for j in range(target)]
    circ = [*upper, gate, *lower]
    res = circ[-1]

    for i in reversed(list(range(1, len(circ)))):
        res = np.kron(circ[i - 1], res)

    return res


def apply_gate_to_tlines(gate_matrix, circuits_size=2, targets=None, dims=None):
    if dims is None:
        dims = [2, 2]
    if targets is None:
        targets = range(circuits_size)

    if isinstance(targets, int):
        targets = [targets]

    subset_gate = 0
    for i in targets:
        subset_gate += gate_expand_to_circuit(gate_matrix, circuits_size, i, dims)
    return subset_gate
