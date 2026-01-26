# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import mqt.qudits.quantum_circuit
import mqt.qudits.simulation.noise_tools

def state_vector_simulation(
    circuit: mqt.qudits.quantum_circuit.QuantumCircuit, noise_model: mqt.qudits.simulation.noise_tools.NoiseModel
) -> list[complex]:
    """Simulate the state vector of a quantum circuit with noise model.

    Args:
        circuit: The quantum circuit to simulate
        noise_model: The noise model to apply

    Returns:
        list: The state vector of the quantum circuit
    """
