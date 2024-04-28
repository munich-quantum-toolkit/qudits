from __future__ import annotations

from random import randint
from unittest import TestCase

import numpy as np

from mqt.qudits._qudits.misim import state_vector_simulation
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel


def rand_0_5():
    return randint(0, 5)


def is_quantum_state(state):
    """
    Check if a given NumPy array represents a valid quantum state vector.
    """
    # Check if the input is a NumPy array
    if not isinstance(state, np.ndarray):
        print("Input is not a NumPy array")
        return False

    # Squeeze the array to one dimension
    state = np.squeeze(state)

    # Check if the array is 1-dimensional
    if state.ndim != 1:
        print("Array is not 1-dimensional")
        return False

    # Check if the array is complex
    if not np.issubdtype(state.dtype, np.complexfloating):
        print("Array is not complex")
        return False

    # Check if the array is normalized
    norm = np.linalg.norm(state)
    if not np.isclose(norm, 1.0):
        print("Array is not normalized")
        return False

    return True


class Testpymisim(TestCase):
    def test_state_vector_simulation(self):
        qreg_example = QuantumRegister("reg", 6, 6 * [5])
        circ = QuantumCircuit(qreg_example)
        for _i in range(300):
            circ.rz(rand_0_5(), [0, 2, np.pi / 13])
            circ.x(rand_0_5()).dag()
            circ.s(rand_0_5())
            circ.z(rand_0_5())
            circ.csum([5, 1])
            circ.virtrz(rand_0_5(), [0, np.pi / 13]).dag()
            circ.virtrz(rand_0_5(), [1, -np.pi / 8])
            circ.virtrz(rand_0_5(), [1, -np.pi / 8])
            circ.csum([2, 5]).dag()
            circ.x(rand_0_5()).dag()
            circ.x(rand_0_5()).dag()
            circ.z(rand_0_5())
            circ.z(rand_0_5()).dag()
            circ.h(rand_0_5())
            circ.rz(rand_0_5(), [3, 4, np.pi / 13]).dag()
            circ.h(rand_0_5()).dag()
            circ.r(rand_0_5(), [0, 1, np.pi / 5 + np.pi, np.pi / 7])
            circ.rh(rand_0_5(), [1, 3])

            circ.h(rand_0_5())

            circ.r(rand_0_5(), [0, 4, np.pi, np.pi / 2]).dag()
            circ.r(rand_0_5(), [0, 3, np.pi / 5, np.pi / 7])
            circ.h(rand_0_5())
            choice = rand_0_5()
            x = circ.x(choice)
            x.control([int(np.mod(choice + 1, 5))], [2])

            circ.cx([1, 2], [0, 1, 1, np.pi / 2]).dag()
            circ.cx([3, 4], [0, 3, 0, np.pi / 12])
            circ.csum([0, 1])

        # Depolarizing quantum errors
        local_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        local_error_rz = Noise(probability_depolarizing=0.3, probability_dephasing=0.3)
        entangling_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        entangling_error_extra = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        entangling_error_on_target = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        entangling_error_on_control = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)

        # Add errors to noise_tools model

        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate
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

        print("Start execution")
        state_vec = np.array(state_vector_simulation(circ, noise_model))
        assert len(state_vec) == 5**6
        assert is_quantum_state(state_vec)
