from __future__ import annotations

import copy
from itertools import starmap
from random import uniform

import numpy as np
from scipy.optimize import minimize

from mqt.qudits.compiler.compilation_minitools import gate_expand_to_circuit
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import Optimizer
from mqt.qudits.compiler.twodit.variational_twodit_compilation.parametrize import generic_sud, params_splitter
from mqt.qudits.quantum_circuit.gates import CustomOne


def apply_rotations(M, params, dims):
    params = params_splitter(params, dims)
    R1 = gate_expand_to_circuit(generic_sud(params[0], dims[0]), circuits_size=2, target=0, dims=dims)
    R2 = gate_expand_to_circuit(generic_sud(params[1], dims[1]), circuits_size=2, target=1, dims=dims)
    R3 = gate_expand_to_circuit(generic_sud(params[2], dims[0]), circuits_size=2, target=0, dims=dims)
    R4 = gate_expand_to_circuit(generic_sud(params[3], dims[1]), circuits_size=2, target=1, dims=dims)

    return R1 @ R2 @ M @ R3 @ R4


def instantiate_rotations(circuit, gate, params):
    gate = copy.deepcopy(gate)
    gate.parent_circuit = circuit
    dims = gate._dimensions
    params = params_splitter(params, dims)

    decomposition = []

    decomposition.extend((
        CustomOne(circuit, "CUo_SUD", 0, generic_sud(params[0], dims[0]), dims[0]),
        CustomOne(circuit, "CUo_SUD", 1, generic_sud(params[1], dims[1]), dims[1]),
        gate,
        CustomOne(circuit, "CUo_SUD", 0, generic_sud(params[2], dims[0]), dims[0]),
        CustomOne(circuit, "CUo_SUD", 1, generic_sud(params[3], dims[1]), dims[1]),
    ))

    return decomposition


def density(M_prime):
    non_zero_elements = M_prime[M_prime > 1e-8]
    if len(non_zero_elements) == 0:
        return 0
    return non_zero_elements.size / M_prime.size


def manhattan_norm(matrix):
    return np.sum(np.abs(matrix))


def frobenius_norm(matrix):
    return np.sqrt(np.sum(np.abs(matrix) ** 2))


def compute_F(X):
    # Hoyer's sparsity measure on matrices
    # 0<=H<=1 , 0 is non sparse, 1 is very sparse
    # sparsity is then 1 when non sparse , 0 when sparse
    # Create an all-ones matrix J with the same shape as X
    J = np.ones_like(X)

    # Compute the Manhattan norm of X
    norm_X1 = manhattan_norm(X)

    # Compute the Frobenius norm of X and J
    norm_X2 = frobenius_norm(X)
    norm_J2 = frobenius_norm(J)

    # Compute the numerator and denominator
    numerator = norm_J2 * norm_X2 - norm_X1
    denominator = norm_J2 * norm_X2 - norm_X2

    # Handle the potential division by zero
    if denominator == 0:
        msg = "Denominator is zero, which will cause division by zero."
        raise ValueError(msg)

    # Compute F(X)
    F_X = numerator / denominator
    return 1 - F_X


def objective_function(thetas, M, dims):
    """
    Objective function for promoting sparsity and low variance in the transformed matrix M'.

    Args:
        thetas (list of float): List of rotation angles.
        M (numpy.ndarray): Input matrix (NxN).

    Returns:
        float: The value of the objective function.
    """
    M_prime = apply_rotations(M, thetas, dims)

    # Separate the real and imaginary parts
    real_part = np.real(M_prime)
    imag_part = np.imag(M_prime)
    M_ghost = np.abs(real_part) + np.abs(imag_part)

    # Variance of non-zero elements
    den = density(M_ghost)

    return compute_F(M_ghost) * den


def sparsify(gate, tol=0.1):
    M = gate.to_matrix()
    dims = gate._dimensions

    Optimizer.set_class_variables(M, tol, dims[0], dims[1])
    bounds = Optimizer.return_bounds()

    initial_thetas = np.array(list(starmap(uniform, bounds)))

    # Optimize the rotation angles
    result = minimize(objective_function, initial_thetas, args=(M, dims), bounds=bounds)
    # result = dual_annealing(objective_function, args=(M, dims), bounds=bounds)
    optimal_thetas = result.x
    # f = result.fun

    circuit = copy.deepcopy(gate.parent_circuit)
    gates = instantiate_rotations(circuit, gate, optimal_thetas)
    circuit.set_instructions(gates)
    return circuit
