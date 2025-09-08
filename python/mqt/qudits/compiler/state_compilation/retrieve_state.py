from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def verify_normalized_state(quantum_state: ArrayLike) -> bool:
    squared_magnitudes = np.abs(quantum_state) ** 2
    sum_squared_magnitudes = float(np.sum(squared_magnitudes))

    # Check if the sum is approximately equal to one
    tolerance = 1e-6
    return bool(np.isclose(sum_squared_magnitudes, 1.0, atol=tolerance))


def generate_random_quantum_state(cardinalities: list[int]) -> ArrayLike:
    length = reduce(operator.mul, cardinalities)
    # Generate random complex numbers with real and imaginary parts
    rng = np.random.default_rng()
    real_parts = rng.standard_normal(length)
    imag_parts = rng.standard_normal(length)
    complex_nums = real_parts + 1j * imag_parts

    # Normalize the array
    return complex_nums / np.linalg.norm(complex_nums)


def generate_all_combinations(dimensions: list[int]) -> list[list[int]]:
    if len(dimensions) == 0:
        return [[]]

    all_combinations = []

    for i in range(dimensions[0]):
        sub_combinations = generate_all_combinations(dimensions[1:])
        for sub_combination in sub_combinations:
            all_combinations.append([i, *sub_combination])  # noqa: PERF401

    return all_combinations


def generate_ghz_entries(dimensions: list[int]) -> list[list[int]]:
    min_d = min(dimensions)
    entries = []

    for i in range(min_d):
        entry = [i] * len(dimensions)
        entries.append(entry)

    return entries


def generate_qudit_w_entries(dimensions: list[int]) -> list[list[int]]:
    num_positions = len(dimensions)
    entries = []

    for j, d in enumerate(dimensions):
        for i in range(1, d):
            entry = [0] * num_positions
            entry[j] = i
            entries.append(entry)

    return entries


def generate_embedded_w_entries(dimensions: list[int]) -> list[list[int]]:
    num_positions = len(dimensions)
    entries = []

    for j in range(len(dimensions)):
        entry = [0] * num_positions
        entry[j] = 1
        entries.append(entry)

    return entries


def find_entries_indices(input_list: list[list[int]], sublist: list[list[int]]) -> list[int]:
    indices = []

    for state in sublist:
        idf = True
        for i in range(len(input_list)):
            for j in range(len(input_list[i])):
                if input_list[i][j] != state[j]:
                    idf = False
                    break
            if idf:
                indices.append(i)
            idf = True

    indices.sort()
    return indices


def generate_uniform_state(dimensions: list[int], state: str) -> ArrayLike:
    all_entries = generate_all_combinations(dimensions)

    if state == "qudit-w-state":
        state_entries = generate_qudit_w_entries(dimensions)
    elif state == "embedded-w-state":
        state_entries = generate_embedded_w_entries(dimensions)
    elif state == "ghz":
        state_entries = generate_ghz_entries(dimensions)
    else:
        msg = "Input chose is wrong"
        raise ValueError(msg)

    complex_ = np.sqrt(1.0 / len(state_entries))
    state_vector = np.array([0.0] * len(all_entries), dtype=complex)

    for i in find_entries_indices(all_entries, state_entries):
        state_vector[i] = complex_

    return state_vector
