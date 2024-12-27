from __future__ import annotations

import copy
import typing
from itertools import starmap
from random import uniform

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-not-found]
from scipy.stats import unitary_group  # type: ignore[import-not-found]

from mqt.qudits.compiler.compilation_minitools import gate_expand_to_circuit
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import Optimizer
from mqt.qudits.compiler.twodit.variational_twodit_compilation.parametrize import generic_sud, params_splitter
from mqt.qudits.quantum_circuit.gates import CustomOne

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate


def random_unitary_matrix(n: int) -> NDArray[np.complex128, np.complex128]:
    return unitary_group.rvs(n)


def random_sparse_unitary(n: int, density: float = 0.4) -> NDArray:
    """Generate a random sparse-like complex unitary matrix as a numpy array.

    Parameters:
    -----------
    n : int
        Size of the matrix (n x n)
    density : float
        Approximate density of non-zero elements (between 0 and 1)

    Returns:
    --------
    numpy.ndarray
        A complex unitary matrix with approximate sparsity
    """
    # Create a random complex matrix with mostly zeros
    mat_a = np.zeros((n, n), dtype=complex)

    # Calculate number of non-zero elements
    nnz = int(density * n * n)

    rng = np.random.default_rng()
    # Generate random positions for non-zero elements
    positions = rng.choice(n * n, size=nnz, replace=False)
    rows, cols = np.unravel_index(positions, (n, n))

    values = (rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)) * np.sqrt(n)
    mat_a[rows, cols] = values

    # Add a small random perturbation to all elements to avoid pure identity results
    perturbation = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.01
    mat_a += perturbation

    # Perform QR decomposition to get unitary matrix
    q, _r = np.linalg.qr(mat_a)

    # Make Q more sparse by zeroing out small elements
    mask = np.abs(q) < np.sqrt(density)  # Adaptive threshold
    q[mask] = 0

    # Ensure unitarity by performing another QR
    q_, _ = np.linalg.qr(q)

    return q_


def apply_rotations(
    m: NDArray[np.complex128, np.complex128], params_list: list[float], dims: list[int]
) -> NDArray[np.complex128, np.complex128]:
    params = params_splitter(params_list, dims)
    r1 = gate_expand_to_circuit(generic_sud(params[0], dims[0]), circuits_size=2, target=0, dims=dims)
    r2 = gate_expand_to_circuit(generic_sud(params[1], dims[1]), circuits_size=2, target=1, dims=dims)
    r3 = gate_expand_to_circuit(generic_sud(params[2], dims[0]), circuits_size=2, target=0, dims=dims)
    r4 = gate_expand_to_circuit(generic_sud(params[3], dims[1]), circuits_size=2, target=1, dims=dims)

    result: NDArray[np.complex128, np.complex128] = np.matmul(np.matmul(np.matmul(np.matmul(r1, r2), m), r3), r4)
    return result


def instantiate_rotations(circuit: QuantumCircuit, gate: Gate, params_list: list[float]) -> list[Gate]:
    gate = copy.deepcopy(gate)
    gate.parent_circuit = circuit
    dims = typing.cast("list[int]", gate.dimensions)
    params = params_splitter(params_list, dims)

    decomposition: list[Gate] = []

    decomposition.extend((
        CustomOne(circuit, "CUo_SUD", 0, generic_sud(params[0], dims[0]), dims[0]),
        CustomOne(circuit, "CUo_SUD", 1, generic_sud(params[1], dims[1]), dims[1]),
        gate,
        CustomOne(circuit, "CUo_SUD", 0, generic_sud(params[2], dims[0]), dims[0]),
        CustomOne(circuit, "CUo_SUD", 1, generic_sud(params[3], dims[1]), dims[1]),
    ))

    return decomposition


def density(m_prime: NDArray[np.float64, np.float64]) -> float:
    non_zero_elements = m_prime[m_prime > 1e-8]
    if len(non_zero_elements) == 0:
        return 0
    return typing.cast("float", non_zero_elements.size / m_prime.size)


def manhattan_norm(matrix: NDArray[np.complex128, np.complex128]) -> float:
    return np.sum(np.abs(matrix))


def frobenius_norm(matrix: NDArray[np.complex128, np.complex128]) -> float:
    return typing.cast("float", np.sqrt(np.sum(np.abs(matrix) ** 2)))


def compute_f(x: NDArray[np.complex128, np.complex128]) -> float:
    # Hoyer's sparsity measure on matrices
    # 0<=H<=1 , 0 is non sparse, 1 is very sparse
    # sparsity is then 1 when non sparse , 0 when sparse
    # Create an all-ones matrix J with the same shape as X
    j = np.ones_like(x)

    # Compute the Manhattan norm of X
    norm_x1 = manhattan_norm(x)

    # Compute the Frobenius norm of X and J
    norm_x2 = frobenius_norm(x)
    norm_j2 = frobenius_norm(j)

    # Compute the numerator and denominator
    numerator = norm_j2 * norm_x2 - norm_x1
    denominator = norm_j2 * norm_x2 - norm_x2

    # Handle the potential division by zero
    if denominator == 0:
        msg = "Denominator is zero, which will cause division by zero."
        raise ValueError(msg)

    # Compute F(X)
    f_x = numerator / denominator
    return 1 - f_x


def objective_function(thetas: list[float], m: NDArray[np.complex128, np.complex128], dims: list[int]) -> float:
    """Objective function for promoting sparsity and low variance in the transformed matrix M'.

    Args:
        thetas: List of rotation angles.
        m: Input matrix (NxN).
        dims: Dimensions of the qudits.

    Returns:
        The value of the objective function.
    """
    m_prime = apply_rotations(m, thetas, dims)

    # Separate the real and imaginary parts
    real_part = np.real(m_prime)
    imag_part = np.imag(m_prime)
    m_ghost = np.abs(real_part) + np.abs(imag_part)

    # Variance of non-zero elements
    den = density(m_ghost)

    return compute_f(m_ghost) * den


def sparsify(gate: Gate, tol: float = 0.1) -> QuantumCircuit:
    m = gate.to_matrix()
    dims = typing.cast("list[int]", gate.dimensions)

    Optimizer.set_class_variables(m, tol, dims[0], dims[1])
    bounds = Optimizer.return_bounds()

    initial_thetas = np.array(list(starmap(uniform, bounds)))

    # Optimize the rotation angles
    result = minimize(objective_function, initial_thetas, args=(m, dims), bounds=bounds)
    # result = dual_annealing(objective_function, args=(M, dims), bounds=bounds)
    optimal_thetas = result.x
    # f = result.fun

    circuit = copy.deepcopy(gate.parent_circuit)
    gates = instantiate_rotations(circuit, gate, optimal_thetas)
    circuit.set_instructions(gates)
    return circuit
