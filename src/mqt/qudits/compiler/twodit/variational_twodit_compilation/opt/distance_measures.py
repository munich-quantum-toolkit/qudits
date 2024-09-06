#!/usr/bin/env python3
"""FROM An alternative quantum fidelity for mixed states of qudits https://doi.org/10.1016/j.physleta.2008.10.083."""

from __future__ import annotations

import typing

import numpy as np

from mqt.qudits.exceptions.circuiterror import ShapeMismatchError

if typing.TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def size_check(a: NDArray[np.complex128, np.complex128], b: NDArray[np.complex128, np.complex128]) -> bool:
    return bool(a.shape == b.shape and a.shape[0] == a.shape[1])


def fidelity_on_operator(a: NDArray[np.complex128, np.complex128], b: NDArray[np.complex128, np.complex128]) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    adag = a.T.conj().copy()
    bdag = b.T.conj().copy()
    numerator = np.abs(np.trace(adag @ b))
    denominator = np.sqrt(np.trace(a @ adag) * np.trace(b @ bdag))

    return typing.cast(float, (numerator / denominator))


def fidelity_on_unitares(a: NDArray[np.complex128, np.complex128], b: NDArray[np.complex128, np.complex128]) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    dimension = a.shape[0]

    return typing.cast(float, np.abs(np.trace(a.T.conj() @ b)) / dimension)


def fidelity_on_density_operator(
    a: NDArray[np.complex128, np.complex128], b: NDArray[np.complex128, np.complex128]
) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    numerator = np.abs(np.trace(a @ b))
    denominator = np.sqrt(np.trace(a @ a) * np.trace(b @ b))

    return typing.cast(float, (numerator / denominator))


def naive_state_fidelity(state1: ArrayLike, state2: ArrayLike) -> float:
    """Calculates fidelity between two state vectors."""
    # Ensure both states have the same dimension
    if state1.shape != state2.shape:
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    # Inner product of the two states
    inner_product = np.conj(state1).dot(state2)

    # Fidelity calculation
    return float(np.abs(inner_product) ** 2)
