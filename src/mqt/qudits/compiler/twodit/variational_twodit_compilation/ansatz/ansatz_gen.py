#!/usr/bin/env python3
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.compiler.compilation_minitools import gate_expand_to_circuit
from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import Primitive
from mqt.qudits.compiler.twodit.variational_twodit_compilation.parametrize import generic_sud, params_splitter
from mqt.qudits.quantum_circuit import QuantumCircuit, gates

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.qudits.quantum_circuit.gate import Gate


def prepare_ansatz(u: Gate | None, params: list[list[float]], dims: list[int]) -> NDArray[np.complex128, np.complex128]:
    counter = 0
    unitary = gate_expand_to_circuit(np.identity(dims[0], dtype=complex), circuits_size=2, target=0, dims=dims)

    for i in range(len(params)):
        if counter == 2:
            counter = 0

            unitary = np.matmul(unitary, u)

        unitary = np.matmul(
            unitary,
            gate_expand_to_circuit(generic_sud(params[i], dims[counter]), circuits_size=2, target=counter, dims=dims),
        )

        counter += 1

    return unitary


def cu_ansatz(p: list[float], dims: list[int]) -> NDArray[np.complex128, np.complex128]:
    params = params_splitter(p, dims)
    cu = Primitive.CUSTOM_PRIMITIVE
    return prepare_ansatz(cu, params, dims)


def ms_ansatz(p: list[float], dims: list[int]) -> NDArray[np.complex128, np.complex128]:
    params = params_splitter(p, dims)
    ms = gates.MS(QuantumCircuit(2, dims, 0), "MS", [0, 1], [np.pi / 2], dims).to_matrix(
        identities=0
    )  # ms_gate(np.pi / 2, dim)

    return prepare_ansatz(ms, params, dims)


def ls_ansatz(p: list[float], dims: list[int]) -> NDArray[np.complex128, np.complex128]:
    params = params_splitter(p, dims)

    if 2 in dims:
        theta = np.pi / 2
    elif 3 in dims:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = gates.LS(
        QuantumCircuit(2, dims, 0),
        "LS",
        [0, 1],
        [theta],
        dims,
        None,
    ).to_matrix()  # ls_gate(theta, dim)

    return prepare_ansatz(ls, params, dims)
