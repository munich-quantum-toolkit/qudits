#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def on1(gate: NDArray[np.complex128, np.complex128], other_size: int) -> NDArray[np.complex128, np.complex128]:
    return np.kron(np.identity(other_size, dtype=np.complex128), gate)


def on0(gate: NDArray[np.complex128, np.complex128], other_size: int) -> NDArray[np.complex128, np.complex128]:
    return np.kron(gate, np.identity(other_size, dtype=np.complex128))


def gate_expand_to_circuit(
    gate: NDArray[np.complex128, np.complex128], circuits_size: int, target: int, dims: Sequence[int] | None = None
) -> NDArray[np.complex128]:
    if dims is None:
        dims = [2, 2]
    if circuits_size < 1:
        msg = "integer circuits_size must be larger or equal to 1"
        raise ValueError(msg)

    if target >= circuits_size:
        msg = "target must be integer < integer circuits_size"
        raise ValueError(msg)

    upper = [np.identity(dims[i], dtype="complex") for i in range(target + 1, circuits_size)]
    lower = [np.identity(dims[j], dtype="complex") for j in range(target)]
    circ = [*lower, gate, *upper]
    res = circ[-1]

    for i in reversed(list(range(1, len(circ)))):
        res = np.kron(circ[i - 1], res)

    return res


def apply_gate_to_tlines(
    gate_matrix: NDArray[np.complex128],
    circuits_size: int = 2,
    targets: int | list[int] | None = None,
    dims: list[int] | None = None,
) -> NDArray[np.complex128]:
    if dims is None:
        dims = [2] * circuits_size
    if targets is None:
        targets = list(range(circuits_size))
    elif isinstance(targets, int):
        targets = [targets]

    subset_gate = np.zeros((2**circuits_size, 2**circuits_size), dtype=np.complex128)
    for i in targets:
        subset_gate += gate_expand_to_circuit(gate_matrix, circuits_size, i, dims)
    return subset_gate
