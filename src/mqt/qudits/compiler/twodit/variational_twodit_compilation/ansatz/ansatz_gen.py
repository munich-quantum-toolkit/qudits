from __future__ import annotations

import numpy as np

from .....quantum_circuit import gates
from ....compilation_minitools import gate_expand_to_circuit
from .parametrize import CUSTOM_PRIMITIVE, generic_sud, params_splitter


def prepare_ansatz(u, params, dims):
    counter = 0

    unitary = gate_expand_to_circuit(np.identity(dims[0], dtype=complex), circuits_size=2, target=0, dims=dims)

    for i in range(len(params)):
        if counter == 2:
            counter = 0
            unitary = unitary @ u

        unitary = unitary @ gate_expand_to_circuit(
            generic_sud(params[i], dims[counter]), circuits_size=2, target=counter, dims=dims
        )
        counter += 1

    return unitary


def cu_ansatz(P, dims):
    params = params_splitter(P, dims)
    cu = CUSTOM_PRIMITIVE
    return prepare_ansatz(cu, params, dims)


def ms_ansatz(P, dims):
    params = params_splitter(P, dims)
    ms = gates.MS(
        None,
        "MS",
        None,
        [np.pi / 2],
        dims,
        None,
    ).to_matrix()  # ms_gate(np.pi / 2, dim)

    return prepare_ansatz(ms, params, dims)


def ls_ansatz(P, dims):
    params = params_splitter(P, dims)

    if 2 in dims:
        theta = np.pi / 2
    elif 3 in dims:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = gates.LS(
        None,
        "LS",
        None,
        [theta],
        dims,
        None,
    ).to_matrix()  # ls_gate(theta, dim)

    return prepare_ansatz(ls, params, dims)
