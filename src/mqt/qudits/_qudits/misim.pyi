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
