from __future__ import annotations


class Noise:
    """Represents a noise model with depolarizing and dephasing probabilities."""

    def __init__(self, probability_depolarizing: float, probability_dephasing: float) -> None:
        self.probability_depolarizing = probability_depolarizing
        self.probability_dephasing = probability_dephasing

    def __str__(self) -> str:
        return str(self.probability_depolarizing) + " " + str(self.probability_dephasing)


class SubspaceNoise:
    """Represents physical noises for each level transitions."""

    def __init__(
        self,
        probability_depolarizing: float,
        probability_dephasing: float,
        levels: tuple[int, int] | list[tuple[int, int]],
    ) -> None:
        self.subspace_w_probs: dict[tuple[int, int], Noise] = {}

        if isinstance(levels, tuple):
            self.add_noise(
                levels[0],
                levels[1],
                Noise(probability_depolarizing, probability_dephasing),
            )
        elif len(levels) > 0:
            for lev in levels:
                self.add_noise(
                    lev[0],
                    lev[1],
                    Noise(probability_depolarizing, probability_dephasing),
                )
        else:
            # case where you want the subspace noise to be dynamically assigned
            # to the two-dimensional subspace of a Givens derived rotation.
            # The negative values are not physical and we will check only if they are negative.
            self.add_noise(
                -2,
                -1,
                Noise(probability_depolarizing, probability_dephasing),
            )

    def __str__(self) -> str:
        result = ""
        for levs, noise in self.subspace_w_probs.items():
            result += str(levs[0]) + "<->" + str(levs[1]) + ":" + str(noise) + ", "
        return result

    def add_noise(self, lev_a: int, lev_b: int, noise: Noise) -> None:
        if lev_b < lev_a:
            lev_a, lev_b = lev_b, lev_a
        if lev_a == lev_b:
            msg = "The levels in the subspace noise should be different!"
            raise ValueError(msg)
        if (lev_a, lev_b) in self.subspace_w_probs:
            msg = "The same level physical noise is defined for multiple times!"
            raise ValueError(msg)
        if (lev_a < 0 or lev_b < 0) and len(self.subspace_w_probs) > 0:
            msg = (
                "Negative keys are for the dynamic assignment of the subspaces, "
                "therefore you cannot assignment more subspaces!"
            )
            raise ValueError(msg)
        if any(levs[0] < 0 and levs[1] < 0 for levs in self.subspace_w_probs):
            msg = (
                "Negative keys are already present for the dynamic assignment of the subspaces, "
                "therefore you cannot assignment other subspaces!"
            )
            raise ValueError(msg)
        self.subspace_w_probs[lev_a, lev_b] = noise

    def add_noises(self, noises: dict[tuple[int, int], Noise]) -> None:
        for tup, noise in noises.items():
            self.add_noise(tup[0], tup[1], noise)


class NoiseModel:
    """Represents a quantum noise model for various gates and qudit configurations."""

    def __init__(self) -> None:
        self.quantum_errors: dict[str, dict[str, Noise | SubspaceNoise]] = {}

    def _add_quantum_error(self, noise: Noise | SubspaceNoise, gates: list[str], mode: str) -> None:
        """Helper method to add quantum errors to the model.

        Args:
            noise (SubspaceNoise | Noise): The subspace noise model to add.
            gates (List[str]): List of gate names to apply the noise to.
            mode (Union[Tuple[int, ...], Literal["local", "all", "nonlocal", "target", "control"]]): The mode or qudit configuration for the noise.
        """
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            if mode not in self.quantum_errors[gate]:
                self.quantum_errors[gate][mode] = noise  # empty case
            elif isinstance(noise, Noise):
                msg = "Mathematical noise has been defined multiple times!"
                raise ValueError(msg)
            else:
                existing_instance = self.quantum_errors[gate][mode]
                assert isinstance(existing_instance, SubspaceNoise)
                existing_instance.add_noises(
                    noise.subspace_w_probs
                )  # add the noise info to the existing SubspaceNoise instance

    def add_quantum_error_locally(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a quantum error locally to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "local")

    def add_all_qudit_quantum_error(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a quantum error to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "all")

    def add_nonlocal_quantum_error(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a nonlocal quantum error for specified gates."""
        self._add_quantum_error(noise, gates, "nonlocal")

    def add_nonlocal_quantum_error_on_target(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a nonlocal quantum error on target qudits for specified gates."""
        self._add_quantum_error(noise, gates, "target")

    def add_nonlocal_quantum_error_on_control(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
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
            for mode, subnoise in errors.items():
                info_str += f"Gate: {gate}, Mode: {mode}, SubspaceNoise: {subnoise}\n"
        return info_str
