from __future__ import annotations

import copy
import typing

import numpy as np

from mqt.qudits.core.micro_dd import (
    create_decision_tree,
    cut_branches,
    dd_reduction_aggregation,
    dd_reduction_hashing,
    get_node_contributions,
    normalize_all,
)
from mqt.qudits.quantum_circuit.gates import R

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.qudits.core.micro_dd import MicroDDNode, NodeContribution
    from mqt.qudits.quantum_circuit import QuantumCircuit


def find_complex_number(x: complex, c: complex) -> complex:
    a = x.real  # Real part of x
    b = x.imag  # Imaginary part of x

    # Calculate z
    real_part = (c.real - b * c.imag) / (a**2 + b**2)
    imag_part = (c.imag + b * c.real) / (a**2 + b**2)
    return complex(real_part, imag_part)


def get_angles(from_: complex, to_: complex) -> tuple[float, float]:
    theta = 2 * np.arctan2(abs(from_), abs(to_))
    phi = typing.cast("float", -(np.pi / 2 + np.angle(to_) - np.angle(from_)))

    return theta, phi


class Operation:
    def __init__(
        self, controls: list[tuple[int, int]], qudit: int, levels: tuple[int, int], angles: tuple[float, float]
    ) -> None:
        self._controls = controls
        self._qudit = qudit
        self._levels = levels
        self._angles = angles

    def is_z(self) -> bool:
        return self._levels == (-1, 0)

    @property
    def controls(self) -> list[tuple[int, int]]:
        return self._controls

    @controls.setter
    def controls(self, value: list[tuple[int, int]]) -> None:
        self._controls = value

    def get_control_nodes(self) -> list[int]:
        return [c[0] for c in self._controls]

    def get_control_levels(self) -> list[int]:
        return [c[1] for c in self._controls]

    @property
    def qudit(self) -> int:
        return self._qudit

    @qudit.setter
    def qudit(self, value: int) -> None:
        self._qudit = value

    @property
    def levels(self) -> tuple[int, int]:
        return self._levels

    @levels.setter
    def levels(self, value: tuple[int, int]) -> None:
        self._levels = value

    def get_angles(self) -> tuple[float, float]:
        return self._angles

    @property
    def theta(self) -> float:
        return self._angles[0]

    @property
    def phi(self) -> float:
        return self._angles[1]

    def __str__(self) -> str:
        return (
            f"QuantumOperation(controls={self._controls}, qudit={self._qudit},"
            f" levels={self._levels}, angles={self._angles})"
        )


class StatePrep:
    def __init__(self, quantum_circuit: QuantumCircuit, state: NDArray[np.complex128], approx: bool = False) -> None:
        self.circuit = quantum_circuit
        self.state = state
        self.approximation = approx

    def retrieve_local_sequence(
        self, fweight: complex, children: list[MicroDDNode]
    ) -> dict[tuple[int, int], tuple[float, float]]:
        size = len(children)
        qudit = int(children[0].value)
        aplog = {}

        coef = np.array([c.weight for c in children])

        for i in reversed(range(size - 1)):
            a, p = get_angles(coef[i + 1], coef[i])
            gate = R(self.circuit, "R", qudit, [i, i + 1, a, p], self.circuit.dimensions[qudit], None).to_matrix()
            coef = np.dot(gate, coef)
            aplog[i, i + 1] = (-a, p)

        phase_2 = float(np.angle(find_complex_number(fweight, coef[0])))
        aplog[-1, 0] = (-phase_2 * 2.0, 0.0)

        return aplog

    def synthesis(
        self,
        labels: list[int],
        cardinalities: list[int],
        node: MicroDDNode,
        circuit_meta: list[Operation],
        controls: list[tuple[int, int]] | None = None,
        depth: int = 0,
    ) -> None:
        if controls is None:
            controls = []
        if node.terminal:
            return
        if node.weight is not None:
            rotations = self.retrieve_local_sequence(node.weight, node.children)
            circuit_meta.extend([
                Operation(controls, labels[depth], key, rotations[key]) for key in sorted(rotations.keys())
            ])

        if not node.reduced:
            for i in range(cardinalities[depth]):
                controls_track = copy.deepcopy(controls)
                controls_track.append((labels[depth], i))
                if len(node.children_index) == 0:
                    self.synthesis(labels, cardinalities, node.children[i], circuit_meta, controls_track, depth + 1)
                else:
                    self.synthesis(
                        labels,
                        cardinalities,
                        node.children[node.children_index[i]],
                        circuit_meta,
                        controls_track,
                        depth + 1,
                    )
        else:
            controls_track = copy.deepcopy(controls)
            self.synthesis(
                labels, cardinalities, node.children[node.children_index[0]], circuit_meta, controls_track, depth + 1
            )

    def compile_state(self) -> QuantumCircuit:
        final_state = self.state
        cardinalities = self.circuit.dimensions
        labels = list(range(len(self.circuit.dimensions)))
        ops: list[Operation] = []
        decision_tree, _number_of_nodes = create_decision_tree(labels, cardinalities, final_state)
        if self.approximation:
            contributions: NodeContribution = get_node_contributions(decision_tree, labels)
            cut_branches(contributions, 0.01)
            normalize_all(decision_tree, cardinalities)

        dd_reduction_hashing(decision_tree, cardinalities)
        dd_reduction_aggregation(decision_tree, cardinalities)
        self.synthesis(labels, cardinalities, decision_tree, ops, [], 0)

        new_circuit = copy.deepcopy(self.circuit)
        for op in ops:
            if abs(op.theta) > 1e-5:
                nodes = op.get_control_nodes()
                levels = op.get_control_levels()
                if op.is_z():
                    new_circuit.rz(op.qudit, [0, 1, op.theta]).control(nodes, levels)
                else:
                    new_circuit.r(op.qudit, [op.levels[0], op.levels[1], op.theta, op.phi]).control(nodes, levels)

        return new_circuit
