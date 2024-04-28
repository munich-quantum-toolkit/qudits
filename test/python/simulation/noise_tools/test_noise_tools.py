from __future__ import annotations

from collections import defaultdict
from random import randint
from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel, NoisyCircuitFactory


def rand_0_5():
    return randint(0, 5)


class TestNoisyCircuitFactory(TestCase):
    def setUp(self) -> None:
        local_error = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)
        local_error_rz = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)
        entangling_error = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)
        entangling_error_extra = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)
        entangling_error_on_target = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)
        entangling_error_on_control = Noise(probability_depolarizing=0.999, probability_dephasing=0.999)

        # Add errors to noise_tools model

        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate_matrix
        noise_model.add_all_qudit_quantum_error(local_error, ["csum"])
        noise_model.add_recurrent_quantum_error_locally(local_error, ["csum"], [0])
        # Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_target(entangling_error_on_target, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_control(entangling_error_on_control, ["csum", "cx", "ls", "ms"])
        # Super noisy Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error_extra, ["csum"])
        # Local Gates
        noise_model.add_quantum_error_locally(local_error, ["rh", "h", "rxy", "s", "x", "z"])
        noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])
        self.noise_model = noise_model

    def test_generate_circuit(self):
        qreg_example = QuantumRegister("reg", 6, 6 * [5])
        circ = QuantumCircuit(qreg_example)
        choice = rand_0_5()
        x = circ.x(choice)
        x = x.control([int(np.mod(choice + 1, 5))], [2])
        rz = circ.rz(rand_0_5(), [0, 2, np.pi / 13])
        cx = circ.cx([3, 4], [0, 3, 0, np.pi / 12])
        x = circ.x(rand_0_5()).dag()
        s = circ.s(rand_0_5())
        z = circ.z(rand_0_5())
        csum = circ.csum([5, 1])
        vrz = circ.virtrz(rand_0_5(), [0, np.pi / 13]).dag()
        vrz = circ.virtrz(rand_0_5(), [1, -np.pi / 8])
        csum = circ.csum([2, 5]).dag()
        x = circ.x(rand_0_5()).dag()
        z = circ.z(rand_0_5()).dag()
        h = circ.h(rand_0_5())
        rz = circ.rz(rand_0_5(), [3, 4, np.pi / 13]).dag()
        h = circ.h(rand_0_5()).dag()
        r = circ.r(rand_0_5(), [0, 1, np.pi / 5 + np.pi, np.pi / 7])
        rh = circ.rh(rand_0_5(), [1, 3])
        r = circ.r(rand_0_5(), [0, 4, np.pi, np.pi / 2]).dag()
        r2 = circ.r(rand_0_5(), [0, 3, np.pi / 5, np.pi / 7])
        cx = circ.cx([1, 2], [0, 1, 1, np.pi / 2]).dag()
        csum = circ.csum([0, 1])

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        instructions_new = new_circ.instructions

        tag_counts_list1 = defaultdict(int)
        tag_counts_list2 = defaultdict(int)
        insts = 0
        insts_new = 0
        for gate in instructions_og:
            insts += 1
            tag_counts_list1[gate.qasm_tag] += 1

        for gate in instructions_new:
            insts_new += 1
            tag_counts_list2[gate.qasm_tag] += 1

        keys_to_check = ['x', 'z', 'rxy', 'rz']
        valid_stochasticity = True
        # Iterate over all keys
        for key in tag_counts_list1.keys() | tag_counts_list2.keys():
            if key in keys_to_check:
                if tag_counts_list1.get(key, 0) > tag_counts_list2.get(key, 0):
                    valid_stochasticity = False
            else:
                if tag_counts_list1.get(key, 0) != tag_counts_list2.get(key, 0):
                    valid_stochasticity = False

        self.assertTrue(valid_stochasticity)
        self.assertTrue(insts == circ.number_gates)
        self.assertTrue(insts_new == new_circ.number_gates)
        self.assertTrue(len(instructions_new) > len(instructions_og))

    def test_generate_circuit_isolated(self):
        qreg_example = QuantumRegister("reg", 2, [5, 5])
        circ = QuantumCircuit(qreg_example)
        x = circ.x(0)
        x = x.control([1], [2])

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        instructions_new = new_circ.instructions

        self.assertTrue(1 == circ.number_gates)
        self.assertTrue(5 == new_circ.number_gates)
        self.assertTrue(["x"] == [i.qasm_tag for i in instructions_og])
        self.assertTrue(["x", "x", "x", "z", "z"])
