from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_sim
from mqt.qudits.compiler.state_compilation.retrieve_state import generate_random_quantum_state, generate_uniform_state
from mqt.qudits.compiler.state_compilation.state_preparation import StatePrep
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt.distance_measures import naive_state_fidelity
from mqt.qudits.quantum_circuit import QuantumCircuit


class TestStatePrep(TestCase):
    @staticmethod
    def test_compile_state():
        for length in range(2, 4):
            rng = np.random.default_rng()
            cardinalities = [int(rng.integers(2, 7)) for _ in range(length)]
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
