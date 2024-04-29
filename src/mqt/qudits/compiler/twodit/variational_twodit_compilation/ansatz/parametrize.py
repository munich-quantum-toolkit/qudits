from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import from_dirac_to_basis

CUSTOM_PRIMITIVE = None  # numpy array


def params_splitter(params, dims):
    ret = []
    n = -1 + dims[0] ** 2
    m = -1 + dims[1] ** 2
    for i in range(0, len(params), n + m):
        ret.append(params[i : i + n])
        if i + n < len(params):
            ret.append(params[i + n : i + n + m])
    return ret


def reindex(ir, jc, num_col):
    return ir * num_col + jc


bound_1 = [0, np.pi]
bound_2 = [0, np.pi / 2]
bound_3 = [0, 2 * np.pi]


def generic_sud(params, dimension) -> np.ndarray:  # required well-structured d2 -1 params
    c_unitary = np.identity(dimension, dtype="complex")

    for diag_index in range(dimension - 1):
        l_vec = from_dirac_to_basis([diag_index], dimension)
        d_vec = from_dirac_to_basis([dimension - 1], dimension)

        zld = np.outer(np.array(l_vec), np.array(l_vec).T.conj()) - np.outer(np.array(d_vec), np.array(d_vec).T.conj())
        c_unitary @= expm(1j * params[reindex(diag_index, diag_index, dimension)] * zld)

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

            c_unitary @= expm(1j * params[reindex(n, m, dimension)] * zmn)

            c_unitary @= expm(1j * params[reindex(m, n, dimension)] * ymn)

    return c_unitary
