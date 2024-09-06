#!/usr/bin/env python3
from __future__ import annotations

import typing

import numpy as np
from scipy.optimize import dual_annealing  # type: ignore[import-not-found]

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz import cu_ansatz, ls_ansatz, ms_ansatz, reindex
from mqt.qudits.exceptions import FidelityReachError

from ..ansatz.ansatz_gen_utils import bound_1, bound_2, bound_3
from .distance_measures import fidelity_on_unitares

if typing.TYPE_CHECKING:
    import multiprocessing

    from numpy.typing import NDArray


class Optimizer:
    timer_var: bool = False
    OBJ_FIDELITY: float = 1e-4
    SINGLE_DIM_0: int = 0
    SINGLE_DIM_1: int = 0
    TARGET_GATE: NDArray[typing.Any, typing.Any] = np.ndarray([])
    MAX_NUM_LAYERS: int = 0  # (2 * SINGLE_DIM_0 * SINGLE_DIM_1)
    X_SOLUTION: typing.ClassVar = []
    FUN_SOLUTION: typing.ClassVar = []

    @classmethod
    def set_class_variables(
        cls,
        target: NDArray[np.complex128, np.complex128],
        obj_fid: float = 1e-4,
        dim_0: int = 0,
        dim_1: int = 0,
        layers: int = 0,
    ) -> None:
        cls.OBJ_FIDELITY = obj_fid
        cls.SINGLE_DIM_0 = dim_0
        cls.SINGLE_DIM_1 = dim_1
        cls.TARGET_GATE = target
        cls.MAX_NUM_LAYERS = layers if layers > 0 else (2 * dim_0 * dim_1 if dim_0 > 0 and dim_1 > 0 else 0)
        cls.X_SOLUTION = []
        cls.FUN_SOLUTION = []

    @staticmethod
    def bounds_assigner(
        b1: tuple[float, float], b2: tuple[float, float], b3: tuple[float, float], num_params_single: int, d: int
    ) -> list[tuple[float, float]]:
        assignment: list[tuple[float, float]] = [(0.0, 0.0)] * (num_params_single + 1)

        for m in range(d):
            for n in range(d):
                if m == n:
                    assignment[reindex(m, n, d)] = b3
                elif m > n:
                    assignment[reindex(m, n, d)] = b1
                else:
                    assignment[reindex(m, n, d)] = b2

        return assignment[:-1]  # dont return last eleement which is just a global phase

    @classmethod
    def return_bounds(cls, num_layer_search: int = 1) -> list[tuple[float, float]]:
        num_params_single_unitary_line_0 = -1 + Optimizer.SINGLE_DIM_0**2
        num_params_single_unitary_line_1 = -1 + Optimizer.SINGLE_DIM_1**2

        bounds_line_0 = Optimizer.bounds_assigner(
            bound_1, bound_2, bound_3, num_params_single_unitary_line_0, Optimizer.SINGLE_DIM_0
        )
        bounds_line_1 = Optimizer.bounds_assigner(
            bound_1, bound_2, bound_3, num_params_single_unitary_line_1, Optimizer.SINGLE_DIM_1
        )

        # Determine the length of the longest bounds list
        # max_length = max(len(bounds_line_0), len(bounds_line_1))

        # Create a new list by alternating elements from bounds_line_0 and bounds_line_1
        bounds = []
        num_layer = num_layer_search
        for _i in range(num_layer + 1 + 1):
            bounds += bounds_line_0
            bounds += bounds_line_1

        return bounds

    @classmethod
    def obj_fun_core(cls, ansatz: NDArray[np.complex128, np.complex128], lambdas: list[float]) -> float:
        if (1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)) < cls.OBJ_FIDELITY:
            cls.X_SOLUTION = lambdas
            cls.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)

            raise FidelityReachError
        if cls.timer_var:
            raise TimeoutError

        return 1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)

    @classmethod
    def objective_fnc_ms(cls, lambdas: list[float]) -> float:
        ansatz = ms_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def objective_fnc_ls(cls, lambdas: list[float]) -> float:
        ansatz = ls_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def objective_fnc_cu(cls, lambdas: list[float]) -> float:
        ansatz = cu_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def solve_anneal(
        cls,
        bounds: list[tuple[float, float]],
        ansatz_type: str,
        result_queue: multiprocessing.Queue[tuple[float, list[float]]],
    ) -> None:
        try:
            if ansatz_type == "MS":  # MS is 0
                opt = dual_annealing(cls.objective_fnc_ms, bounds=bounds)
            elif ansatz_type == "LS":  # LS is 1
                opt = dual_annealing(cls.objective_fnc_ls, bounds=bounds)
            elif ansatz_type == "CU":
                opt = dual_annealing(cls.objective_fnc_cu, bounds=bounds)
            else:
                opt = None

            x = opt.x
            fun = opt.fun

            result_queue.put((fun, x))

        except FidelityReachError:
            result_queue.put((cls.FUN_SOLUTION, cls.X_SOLUTION))

        except TimeoutError:
            result_queue.put((cls.FUN_SOLUTION, cls.X_SOLUTION))
