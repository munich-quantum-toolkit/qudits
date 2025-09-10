# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation.noise_tools import NoiseModel

def state_vector_simulation(circuit: QuantumCircuit, noise_model: NoiseModel) -> list[complex]:
    """Simulate the state vector of a quantum circuit with noise model.

    Args:
        circuit: The quantum circuit to simulate
        noise_model: The noise model to apply

    Returns:
        list: The state vector of the quantum circuit
    """

__all__ = ["state_vector_simulation"]
