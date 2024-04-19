import queue
import threading

import numpy as np
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation import (
    variational_customize_vars,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.ansatz.parametrize import (
    bound_1,
    bound_2,
    bound_3,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.opt.optimizer import (
    bounds_assigner,
    solve_anneal,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.variational_customize_vars import (
    SINGLE_DIM_0,
    SINGLE_DIM_1,
)


def interrupt_function():
    variational_customize_vars.timer_var = True


def binary_search_compile(max_num_layer, ansatz_type):
    if max_num_layer < 0:
        raise Exception

    counter = 0
    low = 0
    high = max_num_layer

    tol = variational_customize_vars.OBJ_FIDELITY

    best_layer, best_error, best_xi = (low + (high - low) // 2, np.inf, [])
    mid, error, xi = (low + (high - low) // 2, np.inf, [])

    # Repeat until the pointers low and high meet each other
    while low <= high:
        print("counter :", counter)
        mid = low + (high - low) // 2
        print("number layer: ", mid)

        error, xi = run(mid, ansatz_type)

        if error > tol:
            low = mid + 1
        else:
            high = mid - 1
            best_layer, best_error, best_xi = (mid, error, xi)

        counter += 1

    return best_layer, best_error, best_xi


def run(num_layer, ansatz_type):
    num_params_single_unitary_line_0 = -1 + variational_customize_vars.SINGLE_DIM_0**2
    num_params_single_unitary_line_1 = -1 + variational_customize_vars.SINGLE_DIM_1**2

    bounds_line_0 = bounds_assigner(bound_1, bound_2, bound_3, num_params_single_unitary_line_0**2, SINGLE_DIM_0) * (
        num_layer + 1
    )
    bounds_line_1 = bounds_assigner(bound_1, bound_2, bound_3, num_params_single_unitary_line_1**2, SINGLE_DIM_1) * (
        num_layer + 1
    )

    bounds = [
        bounds_line_0[i] if i % 2 == 0 else bounds_line_1[i] for i in range(max(len(bounds_line_0), len(bounds_line_1)))
    ]

    duration = 3600 * (variational_customize_vars.SINGLE_DIM_0 * variational_customize_vars.SINGLE_DIM_1 / 4)

    result_queue = queue.Queue()

    thread = threading.Thread(target=solve_anneal, args=(bounds, ansatz_type, result_queue))
    thread.start()

    timer = threading.Timer(duration, interrupt_function)
    timer.start()

    thread.join()
    f, x = result_queue.get()
    # f, x = solve_anneal(bounds, ansatz_type)

    return f, x
