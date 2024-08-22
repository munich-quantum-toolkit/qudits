from __future__ import annotations

import copy
from operator import itemgetter

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz import (
    create_cu_instance,
    create_ls_instance,
    create_ms_instance,
)
from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import Primitive
from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_solve_n_search import binary_search_compile
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import Optimizer


def variational_compile(target, tolerance, ansatz_type, layers, custom_primitive=None):
    dim_0, dim_1 = itemgetter(*target.reference_lines)(target.parent_circuit.dimensions)
    Primitive.set_class_variables(custom_primitive)
    Optimizer.set_class_variables(target.to_matrix(), tolerance, dim_0, dim_1, layers)
    _best_layer, _best_error, parameters = binary_search_compile(layers, ansatz_type)

    circuit = copy.deepcopy(target.parent_circuit)
    if ansatz_type == "MS":  # MS is 0
        gates = create_ms_instance(circuit, parameters, [dim_0, dim_1])
    elif ansatz_type == "LS":  # LS is 1
        gates = create_ls_instance(circuit, parameters, [dim_0, dim_1])
    elif ansatz_type == "CU":
        gates = create_cu_instance(circuit, parameters, [dim_0, dim_1])
    else:
        gates = None
    circuit.set_instructions(gates)
    return circuit
