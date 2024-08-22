from __future__ import annotations

import operator
from functools import reduce
from random import randint
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.state_compilation.retrieve_state import generate_random_quantum_state, generate_uniform_state
from mqt.qudits.compiler.state_compilation.state_preparation import StatePrep
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt.distance_measures import naive_state_fidelity
from mqt.qudits.quantum_circuit import QuantumCircuit


def mini_sim(circuit):
    size = reduce(operator.mul, circuit.dimensions)
    zero = np.array(size * [0])
    zero[0] = 1
    for gate in circuit.instructions:
        zero = np.dot(gate.to_matrix(identities=2), zero)
    return zero


class TestStatePrep(TestCase):
    def test_compile_state(self):
        for length in range(2, 4):
            cardinalities = [randint(2, 7) for _ in range(length)]

            w = generate_uniform_state(cardinalities, "qudit-w-state")
            circuit = QuantumCircuit(length, cardinalities, 0)
            preparation = StatePrep(circuit, w)
            new_circuit = preparation.compile_state()
            state = mini_sim(new_circuit)
            assert np.allclose(state, w)

            ghz = generate_uniform_state(cardinalities, "ghz")
            circuit = QuantumCircuit(length, cardinalities, 0)
            preparation = StatePrep(circuit, ghz)
            new_circuit = preparation.compile_state()
            state = mini_sim(new_circuit)
            assert np.allclose(state, ghz)

            wemb = generate_uniform_state(cardinalities, "embedded-w-state")
            circuit = QuantumCircuit(length, cardinalities, 0)
            preparation = StatePrep(circuit, wemb)
            new_circuit = preparation.compile_state()
            state = mini_sim(new_circuit)
            assert np.allclose(state, wemb)

            circuit = QuantumCircuit(length, cardinalities, 0)
            final_state = generate_random_quantum_state(cardinalities)
            preparation = StatePrep(circuit, final_state)
            new_circuit = preparation.compile_state()
            state = mini_sim(new_circuit)
            assert np.allclose(state, final_state)

            circuit = QuantumCircuit(length, cardinalities, 0)
            final_state = generate_random_quantum_state(cardinalities)
            preparation = StatePrep(circuit, final_state, True)
            new_circuit = preparation.compile_state()
            state = mini_sim(new_circuit)
            assert naive_state_fidelity(state, final_state) > 0.975
