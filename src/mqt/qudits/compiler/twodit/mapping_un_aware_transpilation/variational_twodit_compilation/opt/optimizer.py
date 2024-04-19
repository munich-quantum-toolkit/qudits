from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation import (
    variational_customize_vars,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.ansatz.ansatz_gen import (
    cu_ansatz,
    ls_ansatz,
    ms_ansatz,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.ansatz.parametrize import (
    reindex,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.opt.distance_measures import (
    fidelity_on_unitares,
)
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.variational_twodit_compilation.variational_customize_vars import (
    OBJ_FIDELITY,
    SINGLE_DIM_0,
    SINGLE_DIM_1,
    TARGET_GATE,
    timer_var,
)
from mqt.qudits.exceptions.compilerexception import FidelityReachException
from scipy.optimize import dual_annealing


def bounds_assigner(b1, b2, b3, num_params_single, d):
    assignment = [None] * (num_params_single + 1)

    for m in range(0, d):
        for n in range(0, d):
            if m == n:
                assignment[reindex(m, n, d)] = b3
            elif m > n:
                assignment[reindex(m, n, d)] = b1
            else:
                assignment[reindex(m, n, d)] = b2

    return assignment[:-1]  # dont return last eleement which is just a global phase


def obj_fun_core(ansatz, lambdas):
    print(1 - fidelity_on_unitares(ansatz, TARGET_GATE))

    if (1 - fidelity_on_unitares(ansatz, TARGET_GATE)) < OBJ_FIDELITY:
        variational_customize_vars.X_SOLUTION = lambdas

        variational_customize_vars.FUN_SOLUTION = 1 - fidelity_on_unitares(ansatz, TARGET_GATE)

        raise FidelityReachException
    if timer_var:
        raise TimeoutError

    return 1 - fidelity_on_unitares(ansatz, TARGET_GATE)


def objective_fnc_ms(lambdas):
    ansatz = ms_ansatz(lambdas, [SINGLE_DIM_0, SINGLE_DIM_1])
    return obj_fun_core(ansatz, lambdas)


def objective_fnc_ls(lambdas):
    ansatz = ls_ansatz(lambdas, [SINGLE_DIM_0, SINGLE_DIM_1])
    return obj_fun_core(ansatz, lambdas)


def objective_fnc_cu(lambdas):
    ansatz = cu_ansatz(lambdas, [SINGLE_DIM_0, SINGLE_DIM_1])
    return obj_fun_core(ansatz, lambdas)


def solve_anneal(bounds, ansatz_type, result_queue):
    try:
        if ansatz_type == "MS":  # MS is 0
            opt = dual_annealing(objective_fnc_ms, bounds=bounds)
        elif ansatz_type == "LS":  # LS is 1
            opt = dual_annealing(objective_fnc_ls, bounds=bounds)
        elif ansatz_type == "CU":
            opt = dual_annealing(objective_fnc_cu, bounds=bounds)
        else:
            opt = None

        x = opt.x
        fun = opt.fun

        result_queue.put((fun, x))

    except FidelityReachException as e:
        print("FidelityReachException ", e)
        result_queue.put((variational_customize_vars.FUN_SOLUTION, variational_customize_vars.X_SOLUTION))

    except TimeoutError as e:
        print("Execution Time Out", e)
        result_queue.put((variational_customize_vars.FUN_SOLUTION, variational_customize_vars.X_SOLUTION))
