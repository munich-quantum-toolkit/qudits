import numpy as np
from mqt.qudits.compiler.compilation_minitools.numerical_ansatz_utils import gate_expand_to_circuit
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation import (
    variational_customize_vars,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.ansatz.parametrize import (
    generic_sud,
    params_splitter,
)
from mqt.qudits.qudit_circuits.components.instructions.gate_set.ls import LS
from mqt.qudits.qudit_circuits.components.instructions.gate_set.ms import MS


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
    cu = variational_customize_vars.CUSTOM_PRIMITIVE
    return prepare_ansatz(cu, params, dims)


def ms_ansatz(P, dims):
    params = params_splitter(P, dims)
    ms = MS(
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

    ls = LS(
        None,
        "LS",
        None,
        [theta],
        dims,
        None,
    ).to_matrix()  # ls_gate(theta, dim)

    return prepare_ansatz(ls, params, dims)
