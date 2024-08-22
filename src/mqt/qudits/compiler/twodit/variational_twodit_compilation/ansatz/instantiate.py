from __future__ import annotations

import numpy as np

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import Primitive
from mqt.qudits.compiler.twodit.variational_twodit_compilation.parametrize import generic_sud, params_splitter
from mqt.qudits.quantum_circuit import gates
from mqt.qudits.quantum_circuit.gates import CustomOne


def ansatz_decompose(circuit, u, params, dims):
    counter = 0
    decomposition = []
    for i in range(len(params)):
        if counter == 2:
            counter = 0
            decomposition.append(u)

        decomposition.append(
            CustomOne(circuit, "CUo_SUD", counter, generic_sud(params[i], dims[counter]), dims[counter])
        )

        counter += 1

    return decomposition


def create_cu_instance(circuit, P, dims):
    params = params_splitter(P, dims)
    cu = Primitive.CUSTOM_PRIMITIVE
    return ansatz_decompose(circuit, cu, params, dims)


def create_ms_instance(circuit, P, dims):
    params = params_splitter(P, dims)
    ms = gates.MS(circuit, "MS", [0, 1], [np.pi / 2], dims)  # ms_gate(np.pi / 2, dim)

    return ansatz_decompose(circuit, ms, params, dims)


def create_ls_instance(circuit, P, dims):
    params = params_splitter(P, dims)

    if 2 in dims:
        theta = np.pi / 2
    elif 3 in dims:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = gates.LS(circuit, "LS", [0, 1], [theta], dims)  # ls_gate(theta, dim)

    return ansatz_decompose(circuit, ls, params, dims)
