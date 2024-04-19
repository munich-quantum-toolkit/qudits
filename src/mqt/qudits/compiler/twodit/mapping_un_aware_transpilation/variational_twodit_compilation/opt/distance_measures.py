import typing

import numpy as np

"""FROM An alternative quantum fidelity for mixed states of qudits Xiaoguang Wang, 1, 2, * Chang-Shui Yu, 3 and x. x.
Yi 3 """


def size_check(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape == b.shape and a.shape[0] == a.shape[1]:
        return True

    return False


def fidelity_on_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    adag = a.T.conj().copy()
    bdag = b.T.conj().copy()
    numerator = np.abs(np.trace(adag @ b))
    denominator = np.sqrt(np.trace(a @ adag) * np.trace(b @ bdag))

    return typing.cast(float, (numerator / denominator))


def fidelity_on_unitares(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    dimension = a.shape[0]

    return typing.cast(float, np.abs(np.trace(a.T.conj() @ b)) / dimension)


def fidelity_on_density_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        raise Exception

    numerator = np.abs(np.trace(a @ b))
    denominator = np.sqrt(np.trace(a @ a) * np.trace(b @ b))

    return typing.cast(float, (numerator / denominator))


def density_operator(state_vector) -> np.ndarray:
    if isinstance(state_vector, list):
        state_vector = np.array(state_vector)

    return np.outer(state_vector, state_vector.conj())


def frobenius_dist(x, y):
    a = x - y
    return np.sqrt(np.trace(np.abs(a.T.conj() @ a)))
