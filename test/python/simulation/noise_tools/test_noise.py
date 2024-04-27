from __future__ import annotations

from unittest import TestCase

from mqt.qudits.simulation.noise_tools import Noise, NoiseModel


class TestNoiseModel(TestCase):
    # Depolarizing quantum errors
    local_error = Noise(probability_depolarizing=0.001, probability_dephasing=0.001)
    local_error_rz = Noise(probability_depolarizing=0.03, probability_dephasing=0.03)
    entangling_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.001)
    entangling_error_extra = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
    entangling_error_on_target = Noise(probability_depolarizing=0.1, probability_dephasing=0.0)
    entangling_error_on_control = Noise(probability_depolarizing=0.01, probability_dephasing=0.0)

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
    noise_model.add_quantum_error_locally(local_error, ["h", "rxy", "s", "x", "z"])
    noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])
