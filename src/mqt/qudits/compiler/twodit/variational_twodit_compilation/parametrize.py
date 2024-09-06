#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm  # type: ignore[import-not-found]

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import reindex
from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

if TYPE_CHECKING:
    from numpy.typing import NDArray


def params_splitter(params: list[float] | NDArray[np.float64], dims: list[int]) -> list[list[float]]:
    """Split a list of parameters into sublists based on given dimensions.

    Args:
    params (Union[List[float], np.ndarray]): The input parameters to be split.
    dims (Tuple[int, int]): A tuple of two integers representing the dimensions.

    Returns:
    List[Union[List[float], np.ndarray]]: A list of sublists of split parameters.

    Raises:
    ValueError: If the length of params is not compatible with the given dimensions.
    """
    if len(dims) != 2:
        msg = "dims must be a tuple of two integers"
        raise ValueError(msg)

    n, m = dims[0] ** 2 - 1, dims[1] ** 2 - 1
    step_size = n + m

    if len(params) % step_size != 0:
        msg = f"Length of params ({len(params)}) is not compatible with the given dimensions"
        raise ValueError(msg)

    split_params = []
    for i in range(0, len(params), step_size):
        split_params.extend([params[i : i + n], params[i + n : i + step_size]])

    return split_params


def generic_sud(params: list[float] | NDArray[np.float64], dimension: int) -> NDArray[np.complex128]:
    # required well-structured d2 -1 params
    c_unitary = np.identity(dimension, dtype="complex")

    for diag_index in range(dimension - 1):
        l_vec = from_dirac_to_basis([diag_index], dimension)
        d_vec = from_dirac_to_basis([dimension - 1], dimension)

        zld = np.outer(np.array(l_vec), np.array(l_vec).T.conj()) - np.outer(np.array(d_vec), np.array(d_vec).T.conj())

        c_unitary = np.matmul(c_unitary, expm(1j * params[reindex(diag_index, diag_index, dimension)] * zld))

    for m in range(dimension - 1):
        for n in range(m + 1, dimension):
            m_vec = from_dirac_to_basis([m], dimension)
            n_vec = from_dirac_to_basis([n], dimension)

            zmn = np.outer(np.array(m_vec), np.array(m_vec).T.conj()) - np.outer(
                np.array(n_vec), np.array(n_vec).T.conj()
            )

            ymn = -1j * np.outer(np.array(m_vec), np.array(n_vec).T.conj()) + 1j * np.outer(
                np.array(n_vec), np.array(m_vec).T.conj()
            )

            c_unitary = np.matmul(c_unitary, expm(1j * params[reindex(n, m, dimension)] * zmn))

            c_unitary = np.matmul(c_unitary, expm(1j * params[reindex(m, n, dimension)] * ymn))

    return c_unitary
