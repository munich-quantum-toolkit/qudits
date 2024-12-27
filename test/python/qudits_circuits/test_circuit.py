from __future__ import annotations

from typing import cast
from unittest import TestCase

import numpy as np
import pytest

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_sim, naive_phy_sim
from mqt.qudits.compiler.state_compilation.retrieve_state import generate_random_quantum_state
from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import (
    random_sparse_unitary,
    random_unitary_matrix,
)
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.quantum_circuit.components import ClassicRegister

rng = np.random.default_rng()


def choice(x: list[bool]) -> bool:
    return cast("bool", rng.choice(x, size=1)[0])


class TestQuantumCircuit(TestCase):
    @staticmethod
    def test_to_qasm():
        """Export circuit as QASM program."""
        qreg_field = QuantumRegister("field", 7, [7, 7, 7, 7, 7, 7, 7])
        qreg_matter = QuantumRegister("matter", 2, [2, 2])
        cl_reg = ClassicRegister("classic", 3)

        # Initialize the circuit
        circ = QuantumCircuit(qreg_field)
        circ.append(qreg_matter)
        circ.append_classic(cl_reg)

        # Apply operations
        circ.x(qreg_field[0])
        circ.h(qreg_matter[0])
        circ.cx([qreg_field[0], qreg_field[1]])
        circ.cx([qreg_field[1], qreg_field[2]])
        circ.r(qreg_matter[1], [0, 1, np.pi, np.pi / 2])
        circ.csum([qreg_field[2], qreg_matter[1]])
        circ.pm(qreg_matter[0], [1, 0])
        circ.rh(qreg_field[2], [0, 1])
        circ.ls([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.ms([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.rz(qreg_matter[1], [0, 1, np.pi / 5])
        circ.s(qreg_field[6])
        circ.virtrz(qreg_field[6], [1, np.pi / 5])
        circ.z(qreg_field[4])
        circ.randu([qreg_field[0], qreg_matter[0], qreg_field[1]])
        circ.cu_one(qreg_field[0], np.identity(7))
        circ.cu_two([qreg_field[0], qreg_matter[1]], np.identity(7 * 2))
        circ.cu_multi([qreg_field[0], qreg_matter[1], qreg_matter[0]], np.identity(7 * 2 * 2))

        qasm_program = circ.to_qasm()
        expected_ditqasm = (
            "DITQASM 2.0;qreg field [7][7,7,7,7,7,7,7];qreg matter [2][2,2];creg meas[9];"
            "x field[0];h matter[0];cx (0, 1, 1, 0.0) field[0], field[1];"
            "cx (0, 1, 1, 0.0) field[1], field[2];"
            "rxy (0, 1, 3.141592653589793, 1.5707963267948966) matter[1];csum field[2], matter[1];"
            "pm (1, 0) matter[0];rh (0, 1) field[2];ls (1.0471975511965976) field[2], matter[0];"
            "ms (1.0471975511965976) field[2], matter[0];rz (0, 1, 0.6283185307179586) matter[1];"
            "s field[6];virtrz (1, 0.6283185307179586) field[6];z field[4];"
            "rdu field[0], matter[0], field[1];"
            "cuone (custom_data) field[0];cutwo (custom_data) field[0], matter[1];"
            "cumulti (custom_data) field[0], matter[0], matter[1];measure field[0] -> meas[0];"
            "measure field[1] -> meas[1];measure field[2] -> meas[2];measure field[3] -> meas[3];"
            "measure field[4] -> meas[4];measure field[5] -> meas[5];measure field[6] -> meas[6];"
            "measure matter[0] -> meas[7];measure matter[1] -> meas[8];"
        )

        generated_ditqasm = qasm_program.replace("\n", "")
        assert generated_ditqasm == expected_ditqasm

    @staticmethod
    def test_append():
        qreg_field = QuantumRegister("field", 7, [7, 7])
        QuantumRegister("matter", 2, [2, 2])
        ClassicRegister("classic", 3)

        # Initialize the circuit
        with pytest.raises(
            IndexError, match="Check your Quantum Register to have the right number of lines and number of dimensions"
        ):
            QuantumCircuit(qreg_field)

    @staticmethod
    def test_save_qasm():
        """Export circuit as QASM program."""
        qreg_field = QuantumRegister("field", 7, [7, 7, 7, 7, 7, 7, 7])
        qreg_matter = QuantumRegister("matter", 2, [2, 2])
        cl_reg = ClassicRegister("classic", 3)

        # Initialize the circuit
        circ = QuantumCircuit(qreg_field)
        circ.append(qreg_matter)
        circ.append_classic(cl_reg)

        # Apply operations
        circ.x(qreg_field[0])
        circ.h(qreg_matter[0])
        circ.cx([qreg_field[0], qreg_field[1]])
        circ.cx([qreg_field[1], qreg_field[2]])
        circ.r(qreg_matter[1], [0, 1, np.pi, np.pi / 2])
        circ.csum([qreg_field[2], qreg_matter[1]])
        circ.pm(qreg_matter[0], [1, 0])
        circ.rh(qreg_field[2], [0, 1])
        circ.ls([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.ms([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.rz(qreg_matter[1], [0, 1, np.pi / 5])
        circ.s(qreg_field[6])
        circ.virtrz(qreg_field[6], [1, np.pi / 5])
        circ.z(qreg_field[4])
        circ.randu([qreg_field[0], qreg_matter[0], qreg_field[1]])
        circ.cu_one(qreg_field[0], np.identity(7))
        circ.cu_two([qreg_field[0], qreg_matter[1]], np.identity(7 * 2))
        circ.cu_multi([qreg_field[0], qreg_matter[1], qreg_matter[0]], np.identity(7 * 2 * 2))

        file = circ.save_to_file(file_name="test")
        circ.to_qasm()
        circ_new = QuantumCircuit()
        circ_new.load_from_file(file)

    @staticmethod
    def test_simulate():
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

        og_state = circuit.simulate()
        compiled_state = mini_sim(circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compileo0():
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

        compiled_circuit = circuit.compileO0("faketraps8seven")
        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(compiled_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compileo1_re():
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

        compiled_circuit = circuit.compileO1("faketraps8seven", "resynth")
        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(compiled_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compileo1_ada():
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

        compiled_circuit = circuit.compileO1("faketraps8seven", "adapt")
        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(compiled_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_compileo2():
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

        compiled_circuit = circuit.compileO2("faketraps8seven")
        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(compiled_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)

    @staticmethod
    def test_set_initial_state():
        final_state = generate_random_quantum_state([3, 4, 5])
        circuit = QuantumCircuit(3, [3, 4, 5], 0)
        circuit.set_initial_state(final_state)

        compiled_circuit = circuit.compileO2("faketraps8seven")
        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(compiled_circuit)
        assert np.allclose(og_state, compiled_state, rtol=1e-6, atol=1e-6)
