from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import reindex
from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis


def params_splitter(params, dims):
    ret = []
    n = -1 + dims[0] ** 2
    m = -1 + dims[1] ** 2
    step_size = n + m
    # Iterate over the list with the step size
    for i in range(0, len(params), step_size):
        # Append the first sublist of n elements
        ret.append(params[i : i + n])
        # Append the second sublist of m elements
        ret.append(params[i + n : i + step_size])
    return ret


def generic_sud(params, dimension) -> np.ndarray:  # required well-structured d2 -1 params
    c_unitary = np.identity(dimension, dtype="complex")

    for diag_index in range(dimension - 1):
        l_vec = from_dirac_to_basis([diag_index], dimension)
        d_vec = from_dirac_to_basis([dimension - 1], dimension)

        zld = np.outer(np.array(l_vec), np.array(l_vec).T.conj()) - np.outer(np.array(d_vec), np.array(d_vec).T.conj())

        c_unitary = c_unitary @ expm(1j * params[reindex(diag_index, diag_index, dimension)] * zld)  # noqa

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

            c_unitary = c_unitary @ expm(1j * params[reindex(n, m, dimension)] * zmn)  # noqa

            c_unitary = c_unitary @ expm(1j * params[reindex(m, n, dimension)] * ymn)  # noqa

    return c_unitary
