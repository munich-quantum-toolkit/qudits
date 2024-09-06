from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Noise:
    """Represents a noise model with depolarizing and dephasing probabilities."""

    probability_depolarizing: float
    probability_dephasing: float


class NoiseModel:
    """Represents a quantum noise model for various gates and qudit configurations."""

    def __init__(self) -> None:
        self.quantum_errors: dict[str, dict[str, Noise]] = {}

    def _add_quantum_error(self, noise: Noise, gates: list[str], mode: str) -> None:
        """Helper method to add quantum errors to the model.

        Args:
            noise (Noise): The noise model to add.
            gates (List[str]): List of gate names to apply the noise to.
            mode (Union[Tuple[int, ...], Literal["local", "all", "nonlocal", "target", "control"]]): The mode or qudit configuration for the noise.
        """
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate][mode] = noise

    def add_quantum_error_locally(self, noise: Noise, gates: list[str]) -> None:
        """Add a quantum error locally to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "local")

    def add_all_qudit_quantum_error(self, noise: Noise, gates: list[str]) -> None:
        """Add a quantum error to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "all")

    def add_nonlocal_quantum_error(self, noise: Noise, gates: list[str]) -> None:
        """Add a nonlocal quantum error for specified gates."""
        self._add_quantum_error(noise, gates, "nonlocal")

    def add_nonlocal_quantum_error_on_target(self, noise: Noise, gates: list[str]) -> None:
        """Add a nonlocal quantum error on target qudits for specified gates."""
        self._add_quantum_error(noise, gates, "target")

    def add_nonlocal_quantum_error_on_control(self, noise: Noise, gates: list[str]) -> None:
        """Add a nonlocal quantum error on control qudits for specified gates."""
        self._add_quantum_error(noise, gates, "control")

    @property
    def basis_gates(self) -> list[str]:
        """Get the list of basis gates in the noise model."""
        return list(self.quantum_errors.keys())

    def __str__(self) -> str:
        """Return a string representation of the NoiseModel."""
        info_str = "NoiseModel Info:\n"
        for gate, errors in self.quantum_errors.items():
            for mode, noise in errors.items():
                info_str += f"Gate: {gate}, Mode: {mode}, Noise: {noise}\n"
        return info_str
