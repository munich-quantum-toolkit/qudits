from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from mqt.qudits.core import LevelGraph
    from mqt.qudits.quantum_circuit.gates import R

T = TypeVar("T")


def check_lev(lev: int, dim: int) -> int:
    if lev < dim:
        return lev
    msg = "Mapping Not Compatible with Circuit."
    raise IndexError(msg)


def swap_elements(list_nodes: list[T], i: int, j: int) -> list[T]:
    a = list_nodes[i]
    b = list_nodes[j]
    list_nodes[i] = b
    list_nodes[j] = a
    return list_nodes


def new_mod(a: float, b: float = 2 * np.pi) -> float:
    res = np.mod(a, b)
    return float(res if not res else res - b if a < 0 else res)


def pi_mod(a: float) -> float:
    a = new_mod(a)
    if a > 0 and a > np.pi:
        a -= 2 * np.pi
    elif a < 0 and abs(a) > np.pi:
        a = 2 * np.pi + a
    return a


def regulate_theta(angle: float) -> float:
    theta_in_units_of_pi: float = np.mod(abs(angle / np.pi), 4)
    if angle < 0:
        theta_in_units_of_pi *= -1
    if abs(theta_in_units_of_pi) < 0.2:
        theta_in_units_of_pi += 4.0

    return theta_in_units_of_pi * np.pi


def phi_cost(theta: float) -> float:
    theta_on_units = theta / np.pi

    return abs(theta_on_units) * 1e-04


def theta_cost(theta: float) -> float:
    theta_on_units = theta / np.pi
    return float(4 * abs(theta_on_units) + abs(np.mod(abs(theta_on_units) + 0.25, 0.5) - 0.25)) * 1e-04


def rotation_cost_calc(gate: R, placement: LevelGraph) -> float:
    source = gate.original_lev_a
    target = gate.original_lev_b

    gate_cost = gate.cost

    if placement.is_irnode(source) or placement.is_irnode(target):
        sp_penalty = (
            min(
                placement.distance_nodes(placement.fst_inode, source),
                placement.distance_nodes(placement.fst_inode, target),
            )
            + 1
        )

        gate_cost = sp_penalty * theta_cost(gate.theta)

    return gate_cost
