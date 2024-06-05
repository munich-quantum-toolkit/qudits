from __future__ import annotations

import operator
from functools import reduce

import numpy as np


def verify_normalized_state(quantum_state):
    squared_magnitudes = np.abs(quantum_state) ** 2
    sum_squared_magnitudes = np.sum(squared_magnitudes)

    # Check if the sum is approximately equal to one
    tolerance = 1e-6
    return np.isclose(sum_squared_magnitudes, 1.0, atol=tolerance)


def generate_random_quantum_state(cardinalities):
    length = reduce(operator.mul, cardinalities)
    # Generate random complex numbers with real and imaginary parts
    real_parts = np.random.randn(length)
    imag_parts = np.random.randn(length)
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
            all_combinations.append([i, *sub_combination])

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
        id = True
        for i in range(len(input_list)):
            for j in range(len(input_list[i])):
                if input_list[i][j] != state[j]:
                    id = False
                    break
            if id:
                indices.append(i)
            id = True

    indices.sort()
    return indices


def generate_uniform_state(dimensions: list[int], state: str) -> np.array:
    all_entries = generate_all_combinations(dimensions)

    if state == "qudit-w-state":
        # print("qudit-w-state")
        state_entries = generate_qudit_w_entries(dimensions)
    elif state == "embedded-w-state":
        # print("embedded-w-state")
        state_entries = generate_embedded_w_entries(dimensions)
    elif state == "ghz":
        state_entries = generate_ghz_entries(dimensions)
        # print("GHZ")
    else:
        print("Input chose is wrong")
        raise Exception

    complex_ = np.sqrt(1.0 / len(state_entries))
    state_vector = np.array([0.0] * len(all_entries), dtype=complex)

    for i in find_entries_indices(all_entries, state_entries):
        state_vector[i] = complex_

    return state_vector
