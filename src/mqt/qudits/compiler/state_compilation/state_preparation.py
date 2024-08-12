from __future__ import annotations

import copy

import numpy as np

from mqt.qudits.core.micro_dd import (
    create_decision_tree,
    cut_branches,
    dd_reduction_aggregation,
    dd_reduction_hashing,
    getNodeContributions,
    normalize_all,
)
from mqt.qudits.quantum_circuit.gates import R


def find_complex_number(x, c):
    a = x.real  # Real part of x
    b = x.imag  # Imaginary part of x

    # Calculate z
    real_part = (c.real - b * c.imag) / (a**2 + b**2)
    imag_part = (c.imag + b * c.real) / (a**2 + b**2)
    return complex(real_part, imag_part)


def getAngles(from_, to_):
    theta = 2 * np.arctan2(abs(from_), abs(to_))
    phi = -(np.pi / 2 + np.angle(to_) - np.angle(from_))

    return theta, phi


class Operation:
    def __init__(self, controls, qudit, levels, angles) -> None:
        self._controls = controls
        self._qudit = qudit
        self._levels = levels
        self._angles = angles

    def is_z(self):
        return self._levels == (-1, 0)

    @property
    def controls(self):
        return self._controls

    @controls.setter
    def controls(self, value) -> None:
        self._controls = value

    def get_control_nodes(self):
        return [c[0] for c in self._controls]

    def get_control_levels(self):
        return [c[1] for c in self._controls]

    @property
    def qudit(self):
        return self._qudit

    @qudit.setter
    def qudit(self, value) -> None:
        self._qudit = value

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value) -> None:
        self._levels = value

    def get_angles(self):
        return self._angles

    @property
    def theta(self):
        return self._angles[0]

    @property
    def phi(self):
        return self._angles[1]

    def __str__(self) -> str:
        return (
            f"QuantumOperation(controls={self._controls}, qudit={self._qudit},"
            f" levels={self._levels}, angles={self._angles})"
        )


class StatePrep:
    def __init__(self, quantum_circuit, state, approx=False) -> None:
        self.circuit = quantum_circuit
        self.state = state
        self.approximation = approx

    def retrieve_local_sequence(self, fweight, children):
        size = len(children)
        qudit = children[0].value
        aplog = {}

        coef = np.array([c.weight for c in children])

        for i in reversed(range(size - 1)):
            a, p = getAngles(coef[i + 1], coef[i])
            gate = R(self.circuit, "R", qudit, [i, i + 1, a, p], self.circuit.dimensions[qudit], None).to_matrix()
            coef = np.dot(gate, coef)
            aplog[i, i + 1] = (-a, p)

        phase_2 = np.angle(find_complex_number(fweight, coef[0]))
        aplog[-1, 0] = (-phase_2 * 2, 0)

        return aplog

    def synthesis(self, labels, cardinalities, node, circuit_meta, controls=None, depth=0) -> None:
        if controls is None:
            controls = []
        if node.terminal:
            return

        rotations = self.retrieve_local_sequence(node.weight, node.children)

        for key in sorted(rotations.keys()):
            circuit_meta.append(Operation(controls, labels[depth], key, rotations[key]))

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

    def compile_state(self):
        final_state = self.state
        cardinalities = self.circuit.dimensions
        labels = list(range(len(self.circuit.dimensions)))
        ops = []
        decision_tree, _number_of_nodes = create_decision_tree(labels, cardinalities, final_state)

        if self.approximation:
            contributions = getNodeContributions(decision_tree, labels)
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
