from __future__ import annotations

import random
from typing import cast
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import (
    mini_phy_unitary_sim,
    mini_unitary_sim,
    naive_phy_sim,
)
from mqt.qudits.compiler.state_compilation.retrieve_state import generate_random_quantum_state
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import fidelity_on_unitares
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt.distance_measures import naive_state_fidelity
from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import (
    random_sparse_unitary,
    random_unitary_matrix,
)
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider

rng = np.random.default_rng()


def choice(x: list[bool]) -> bool:
    return cast("bool", rng.choice(x, size=1)[0])


class TestQuditCompiler(TestCase):
    @staticmethod
    def test_compile():
        """!WARNING!

        We are using 1e-6 tolerance due to 17-27k circuit operations on the compiled circuits -
        numerical errors compound across this many operations.
        Once the compiler methods have been improved the threshold should become tighter!

        """
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    if choice([True, False]):
                        circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                    if choice([True, False]):
                        circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions))
        circuit.set_instructions(insts)
        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_transpile():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        final_state = generate_random_quantum_state([3, 4, 5])
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        circuit.set_initial_state(final_state)

        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntSimplePass", "PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, final_state, rtol=1e-6, atol=1e-6)
        assert np.allclose(final_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compile_00():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    if choice([True, False]):
                        circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                    if choice([True, False]):
                        circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O0(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compile_o1_resynth():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    if choice([True, False]):
                        circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                    if choice([True, False]):
                        circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O1_resynth(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compile_o1_adaptive():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    if choice([True, False]):
                        circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                    if choice([True, False]):
                        circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O1_adaptive(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compile_02():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O2(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_random_evo_compile():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")
        final_state = generate_random_quantum_state([3, 4, 5])
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        circuit.set_initial_state(final_state)

        for _k in range(2):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        if choice([True, False]):
                            circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                        if choice([True, False]):
                            circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O2(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        # assert np.allclose(uni_l, uni_cl, rtol=1e-6, atol=1e-6)

        norm_diff = fidelity_on_unitares(uni_l, uni_cl)
        assert (1 - norm_diff) < 1e-6

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        # assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)
        og_state = og_state.reshape(-1)
        norm_diff_s = naive_state_fidelity(og_state, compiled_state)
        assert (1 - norm_diff_s) < 1e-6