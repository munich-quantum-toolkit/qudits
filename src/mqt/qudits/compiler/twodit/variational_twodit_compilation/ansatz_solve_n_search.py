from __future__ import annotations

import queue
import threading

import numpy as np

from .ansatz.parametrize import (
    bound_1,
    bound_2,
    bound_3,
)
from .opt import Optimizer


def interrupt_function() -> None:
    Optimizer.timer_var = True


def binary_search_compile(max_num_layer, ansatz_type):
    if max_num_layer < 0:
        raise Exception

    counter = 0
    low = 0
    high = max_num_layer

    tol = Optimizer.OBJ_FIDELITY

    best_layer, best_error, best_xi = (low + (high - low) // 2, np.inf, [])
    mid, error, xi = (low + (high - low) // 2, np.inf, [])

    # Repeat until the pointers low and high meet each other
    while low <= high:
        mid = low + (high - low) // 2

        error, xi = run(mid, ansatz_type)

        if error > tol:
            low = mid + 1
        else:
            high = mid - 1
            best_layer, best_error, best_xi = (mid, error, xi)

        counter += 1

    return best_layer, best_error, best_xi


def run(num_layer, ansatz_type):
    num_params_single_unitary_line_0 = -1 + Optimizer.SINGLE_DIM_0**2
    num_params_single_unitary_line_1 = -1 + Optimizer.SINGLE_DIM_1**2

    bounds_line_0 = Optimizer.bounds_assigner(
        bound_1, bound_2, bound_3, num_params_single_unitary_line_0**2, Optimizer.SINGLE_DIM_0
    ) * (num_layer + 1)
    bounds_line_1 = Optimizer.bounds_assigner(
        bound_1, bound_2, bound_3, num_params_single_unitary_line_1**2, Optimizer.SINGLE_DIM_1
    ) * (num_layer + 1)

    bounds = [
        bounds_line_0[i] if i % 2 == 0 else bounds_line_1[i] for i in range(max(len(bounds_line_0), len(bounds_line_1)))
    ]

    duration = 3600 * (Optimizer.SINGLE_DIM_0 * Optimizer.SINGLE_DIM_1 / 4)

    result_queue = queue.Queue()

    thread = threading.Thread(target=Optimizer.solve_anneal, args=(bounds, ansatz_type, result_queue))
    thread.start()

    timer = threading.Timer(duration, interrupt_function)
    timer.start()

    thread.join()
    f, x = result_queue.get()
    # f, x = solve_anneal(bounds, ansatz_type)

    return f, x
