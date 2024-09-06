#!/usr/bin/env python3
from __future__ import annotations

import queue
import threading
from typing import Literal

import numpy as np

from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import Optimizer


def interrupt_function() -> None:
    Optimizer.timer_var = True


def binary_search_compile(max_num_layer: int, ansatz_type: Literal["MS", "LS", "CU"]) -> tuple[int, float, list[float]]:
    if max_num_layer < 0:
        raise ValueError

    counter = 0
    low = 0
    high = max_num_layer

    tol = Optimizer.OBJ_FIDELITY

    best_layer, best_error, best_xi = (low + (high - low) // 2, np.inf, [])
    mid: int = (low + (high - low)) // 2
    error: float = np.inf
    xi: list[float] = []
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


def run(num_layer: int, ansatz_type: Literal["MS", "LS", "CU"]) -> tuple[float, list[float]]:
    bounds = Optimizer.return_bounds(num_layer)

    duration = 3600 * (Optimizer.SINGLE_DIM_0 * Optimizer.SINGLE_DIM_1 / 4)

    result_queue: queue.Queue[tuple[float, list[float]]] = queue.Queue()

    thread = threading.Thread(target=Optimizer.solve_anneal, args=(bounds, ansatz_type, result_queue))
    thread.start()

    timer = threading.Timer(duration, interrupt_function)
    timer.start()

    thread.join()
    f, x = result_queue.get()

    return f, x
