from __future__ import annotations

import numpy as np

from .....quantum_circuit import gates
from ....compilation_minitools import gate_expand_to_circuit
from .parametrize import CUSTOM_PRIMITIVE, generic_sud, params_splitter


def ansatz_decompose(u, params, dims):
    decomposition = []
    counter = 0

    for i in range(len(params)):
        if counter == 2:
            counter = 0
            decomposition.append(u)

        decomposition.append(
            gate_expand_to_circuit(generic_sud(params[i], dims[counter]), n=2, target=counter, dim=dims)
        )

        counter += 1

    return decomposition


def create_cu_instance(P, dims):
    params = params_splitter(P, dims)
    cu = CUSTOM_PRIMITIVE
    return ansatz_decompose(cu, params, dims)


def create_ms_instance(P, dims):
    params = params_splitter(P, dims)
    ms = gates.MS(
        None,
        "MS",
        None,
        [np.pi / 2],
        dims,
        None,
    ).to_matrix()  # ms_gate(np.pi / 2, dim)

    return ansatz_decompose(ms, params, dims)


def create_ls_instance(P, dims):
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

    return ansatz_decompose(ls, params, dims)
