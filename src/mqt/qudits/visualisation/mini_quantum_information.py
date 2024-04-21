from __future__ import annotations

import operator
from collections import Counter
from functools import reduce

import numpy as np

from ..quantum_circuit.matrix_factory import from_dirac_to_basis
from .plot_information import state_labels


def get_density_matrix_from_counts(results, circuit):
    num_kets = reduce(operator.mul, circuit.dimensions)
    number_counts = Counter(results)
    probabilities = [(number_counts[num] / len(results)) for num in range(num_kets)]
    kets = [from_dirac_to_basis([int(char) for char in state], circuit.dimensions) for state in state_labels(circuit)]
    density_matrix = np.zeros((num_kets, num_kets))
    for k, p in zip(kets, probabilities):
        density_matrix += p * np.outer(k, k.conj())

    return density_matrix


def partial_trace(rho, qudits2keep, dims, optimize=False):
    """Calculate the partial trace

    p_a = Tr_b(p)

    Parameters
    ----------
    p : 2D array
        Matrix to trace
    qudits2keep : array
        An array of indices of the spaces to keep after
        being traced. For instance, if the space is
        A x B x C x D and we want to trace out B and D,
        keep = [0,2]
    dims : array
        An array of the dimensions of each space.
        For instance, if the space is A x B x C x D,
        dims = [dim_A, dim_B, dim_C, dim_D]

    Returns
    -------
    p_a : 2D array
        Traced matrix
    """
    qudits2keep = np.asarray(qudits2keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[qudits2keep])

    idx1 = list(range(Ndim))
    idx2 = [Ndim + i if i in qudits2keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)
