from __future__ import annotations

from scipy.optimize import dual_annealing

from .....exceptions import FidelityReachException
from ..ansatz import (
    cu_ansatz,
    ls_ansatz,
    ms_ansatz,
    reindex,
)
from .distance_measures import fidelity_on_unitares


class Optimizer:
    OBJ_FIDELITY = 1e-4

    SINGLE_DIM_0 = None
    SINGLE_DIM_1 = None

    TARGET_GATE = None
    MAX_NUM_LAYERS = None  # (2 * SINGLE_DIM_0 * SINGLE_DIM_1)

    X_SOLUTION = []
    FUN_SOLUTION = []

    timer_var = False

    @staticmethod
    def bounds_assigner(b1, b2, b3, num_params_single, d):
        assignment = [None] * (num_params_single + 1)

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
    def obj_fun_core(cls, ansatz, lambdas):
        if (1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)) < cls.OBJ_FIDELITY:
            cls.X_SOLUTION = lambdas
            cls.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)

            raise FidelityReachException
        if cls.timer_var:
            raise TimeoutError

        return 1 - fidelity_on_unitares(ansatz, cls.TARGET_GATE)

    @classmethod
    def objective_fnc_ms(cls, lambdas):
        ansatz = ms_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def objective_fnc_ls(cls, lambdas):
        ansatz = ls_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def objective_fnc_cu(cls, lambdas):
        ansatz = cu_ansatz(lambdas, [cls.SINGLE_DIM_0, cls.SINGLE_DIM_1])
        return cls.obj_fun_core(ansatz, lambdas)

    @classmethod
    def solve_anneal(cls, bounds, ansatz_type, result_queue) -> None:
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

        except FidelityReachException:
            result_queue.put((cls.FUN_SOLUTION, cls.X_SOLUTION))

        except TimeoutError:
            result_queue.put((cls.FUN_SOLUTION, cls.X_SOLUTION))
