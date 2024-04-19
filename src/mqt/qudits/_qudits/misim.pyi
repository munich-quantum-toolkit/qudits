from typing import Any


# todo: this need significantly better typing and a better docstring
def state_vector_simulation(circuit: Any, noise_model: dict[str, Any]) -> list[complex]:  # noqa: ANN401
    """Simulate the state vector of a quantum circuit with noise model.

    Args:
        circuit: The quantum circuit to simulate
        noise_model: The noise model to apply

    Returns:
        list: The state vector of the quantum circuit
    """


__all__ = ["state_vector_simulation"]
