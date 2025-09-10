# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import operator
from collections import Counter
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

from .plot_information import state_labels

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..quantum_circuit import QuantumCircuit


def get_density_matrix_from_counts(
    results: list[int] | NDArray[np.int_], circuit: QuantumCircuit
) -> NDArray[np.complex128]:
    num_kets = reduce(operator.mul, circuit.dimensions)
    number_counts = Counter(results)
    probabilities = [(number_counts[num] / len(results)) for num in range(num_kets)]
    kets: list[NDArray[np.complex128]] = [
        from_dirac_to_basis([int(char) for char in state], circuit.dimensions) for state in state_labels(circuit)
    ]
    density_matrix: NDArray[np.complex128] = np.zeros((num_kets, num_kets)).astype(np.complex128)
    for k, p in zip(kets, probabilities, strict=False):
        density_matrix += p * np.outer(k, k.conj())

    return density_matrix


def partial_trace(
    rho: NDArray[np.complex128], qudits2keep: list[int], dims: list[int], optimize: bool = False
) -> NDArray[np.complex128]:
    """Calculate the partial trace.

    p_a = Tr_b(p)

    Args:
        rho: Matrix to trace
        qudits2keep: An array of indices of the spaces to keep after being traced.
            For instance, if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
        dims: An array of the dimensions of each space.
            For instance, if the space is A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D]
        optimize: Whether to optimize the einsum operation. Defaults to False.

    Returns:
        The traced matrix
    """
    qudits2keep_array: NDArray[np.int_] = np.asarray(qudits2keep, dtype=np.int_)
    dims_array: NDArray[np.int_] = np.asarray(dims, dtype=np.int_)
    ndim: int = dims_array.size
    nkeep: int = int(np.prod(dims_array[qudits2keep_array]))

    letters = [chr(ord("a") + i) for i in range(2 * ndim)]
    subscripts_in = ""
    subscripts_out = ""
    for i in range(ndim):
        if i in qudits2keep_array:
            subscripts_in += letters[i] + letters[ndim + i]
            subscripts_out += letters[i]
        else:
            subscripts_in += letters[i] + letters[i]
    einsum_str = f"{subscripts_in}->{subscripts_out}"

    rho_reshaped = rho.reshape(np.tile(dims_array, 2))
    rho_traced: NDArray[np.complex128] = np.einsum(einsum_str, rho_reshaped, optimize=optimize)
    return rho_traced.reshape(nkeep, nkeep)
