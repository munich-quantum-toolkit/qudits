from __future__ import annotations

import copy
import os
import time
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np

from ...quantum_circuit import QuantumCircuit
from ...quantum_circuit.components.extensions.gate_types import GateTypes
from ...quantum_circuit.gates import CEx, R, Rh, Rz
from .noise import Noise, SubspaceNoise

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator

    from ...quantum_circuit.gate import Gate
    from .noise import NoiseModel


class NoisyCircuitFactory:
    def __init__(self, noise_model: NoiseModel, circuit: QuantumCircuit) -> None:
        self.noise_model: NoiseModel = noise_model
        self.circuit: QuantumCircuit = circuit
        self.rng: Generator = self._initialize_rng()

    @staticmethod
    def _initialize_rng() -> Generator:
        current_time = int(time.time() * 1000)
        seed = hash((os.getpid(), current_time)) % 2**32
        return np.random.default_rng(seed)

    def generate_circuit(self) -> QuantumCircuit:
        noisy_circuit = QuantumCircuit(self.circuit.num_qudits, self.circuit.dimensions, self.circuit.num_cl)
        noisy_circuit.number_gates = 0

        for instruction in self.circuit.instructions:
            copied_instruction = copy.deepcopy(instruction)
            noisy_circuit.instructions.append(copied_instruction)
            noisy_circuit.number_gates += 1

            self._apply_noise(noisy_circuit, instruction)

        return noisy_circuit

    def _dynamic_subspace_noise_info_rectification(self, noise_info: SubspaceNoise, instruction: Gate) -> SubspaceNoise:
        """Corrects negative subspace levels in SubspaceNoise by matching them to physical two-level rotations.

        This function adapts SubspaceNoise dynamically to align with physical local two-level rotations
        (R, Rz, Rh, CEx). If the input noise has negative subspace levels, it maps them to the
        instruction's actual levels.

        Args:
            noise_info: The original SubspaceNoise object to be corrected
            instruction: The Gate instruction containing the target levels
        Returns:
            SubspaceNoise: Corrected noise information with proper subspace levels
        Note:
            Only processes single-subspace noise objects with negative indices.
            Supported gate types are R, Rz, Rh, and CEx.
        """
        # Return original if no correction needed
        if not self._needs_correction(noise_info):
            return noise_info

        # Return original if instruction type not supported
        if not isinstance(instruction, (R, Rz, Rh, CEx)):
            return noise_info

        # Get the noise probabilities from the negative-indexed subspace
        subspace = next(iter(noise_info.subspace_w_probs.keys()))
        noise_probs = noise_info.subspace_w_probs[subspace]

        # Create new noise info with correct levels
        return SubspaceNoise(
            probability_depolarizing=noise_probs.probability_depolarizing,
            probability_dephasing=noise_probs.probability_dephasing,
            levels=(instruction.lev_a, instruction.lev_b),
        )

    @staticmethod
    def _needs_correction(noise_info: SubspaceNoise) -> bool:
        """Determines if the noise info needs level correction.

        Returns True if there is exactly one subspace and it has negative indices.
        """
        if len(noise_info.subspace_w_probs) != 1:
            return False

        subspace = next(iter(noise_info.subspace_w_probs.keys()))
        return subspace[0] < 0 or subspace[1] < 0

    def _apply_noise(self, noisy_circuit: QuantumCircuit, instruction: Gate) -> None:
        if instruction.qasm_tag not in self.noise_model.quantum_errors:
            return

        for mode, noise_info in self.noise_model.quantum_errors[instruction.qasm_tag].items():
            qudits = self._get_affected_qudits(instruction, mode)
            if qudits is None:
                continue  # type: ignore[unreachable]

            if isinstance(noise_info, SubspaceNoise):
                noise_info = self._dynamic_subspace_noise_info_rectification(noise_info, instruction)  # noqa: PLW2901

            self._apply_depolarizing_noise(noisy_circuit, qudits, noise_info)
            self._apply_dephasing_noise(noisy_circuit, qudits, noise_info)

    def _get_affected_qudits(self, instruction: Gate, mode: str) -> list[int]:
        if isinstance(mode, str):
            return self._get_qudits_for_mode(instruction, mode)
        msg = "Something broken is construction of Noise Model."  # type: ignore[unreachable]
        raise ValueError(msg)

    def _get_qudits_for_mode(self, instruction: Gate, mode: str) -> list[int]:
        mode_handlers: dict[str, Callable[[], list[int]]] = {
            "local": lambda: instruction.reference_lines,
            "all": lambda: list(range(instruction.parent_circuit.num_qudits)),
            "nonlocal": partial(self._get_nonlocal_qudits, instruction),
            "control": partial(self._get_control_qudits, instruction),
            "target": partial(self._get_target_qudits, instruction),
        }

        handler = mode_handlers.get(mode)
        if not handler:
            msg = f"Unknown mode: {mode}"
            raise ValueError(msg)

        return handler()

    @staticmethod
    def _get_nonlocal_qudits(instruction: Gate) -> list[int]:
        if instruction.gate_type not in {GateTypes.TWO, GateTypes.MULTI}:
            msg = f"Nonlocal mode not applicable for gate type: {instruction.gate_type}"
            raise ValueError(msg)
        return instruction.reference_lines

    def _get_control_qudits(self, instruction: Gate) -> list[int]:
        self._validate_two_qudit_gate(instruction)
        qudits_targeted = cast("list[int]", instruction.target_qudits)
        return qudits_targeted[:1]

    @staticmethod
    def _get_target_qudits(instruction: Gate) -> list[int]:
        """Returns the target qudits for a gate instruction, excluding control qudits.

        For multi-controlled gates (e.g. controlled-R, controlled-exchange), returns all
        non-control qudits. For single qudit gates, returns a list with just the target qudit.

        Args:
        instruction: The gate instruction to analyze

        Returns:
        List of target qudit indices, excluding control qudits
        """
        if instruction.gate_type != GateTypes.SINGLE:
            qudits_targeted = instruction.reference_lines
            return qudits_targeted[1:]

        return [cast("int", instruction.target_qudits)]

    @staticmethod
    def _validate_two_qudit_gate(instruction: Gate) -> None:
        if instruction.gate_type != GateTypes.TWO:
            msg = f"Gate type {instruction.gate_type} is incompatible for the desired operation."
            raise ValueError(msg)

    def _apply_depolarizing_noise(
        self, noisy_circuit: QuantumCircuit, qudits: list[int], noise_info: Noise | SubspaceNoise
    ) -> None:
        if isinstance(noise_info, Noise):  # Mathematical Description of Depolarizing noise channel
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                prob_each = noise_info.probability_depolarizing / dim / dim  # TODO: ARE WE SURE THIS IS CORRECT?
                noise_combinations = list(product(range(dim), repeat=2))
                probabilities = [1 - prob_each * (dim * dim - 1)] + [prob_each] * (dim * dim - 1)
                power_noise_x, power_noise_z = self.rng.choice(noise_combinations, p=probabilities)
                # TODO: THE FOLLOWING LINES COULD CREATE A LOT OF OVERHEAD IN SIMULATION
                for _ in range(power_noise_x):
                    noisy_circuit.x(dit)
                for _ in range(power_noise_z):
                    noisy_circuit.z(dit)
        elif isinstance(noise_info, SubspaceNoise):  # Physical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                possible_levels = list(range(dim))

                # Validate all subspace levels
                for lev_a, lev_b in noise_info.subspace_w_probs:
                    if lev_a not in possible_levels or lev_b not in possible_levels:
                        msg = (
                            "Subspace levels exceed qudit dimensions. "
                            f"Got levels ({lev_a}, {lev_b}) but dimension is {dim}. "
                            "Check noise model compatibility with circuit."
                        )
                        raise IndexError(msg)

                    # Calculate probabilities for noise operations
                    prob_each = noise_info.subspace_w_probs[lev_a, lev_b].probability_depolarizing / 4

                    # Generate possible noise combinations (X,Z gates)
                    noise_combinations = list(product(range(2), repeat=2))
                    probabilities = [1 - 3 * prob_each] + [prob_each] * 3

                    # Choose noise operation based on probability distribution
                    noise_x, noise_z = self.rng.choice(noise_combinations, p=probabilities)

                    # Apply appropriate noise operation
                    if (noise_x, noise_z) == (1, 0):
                        noisy_circuit.noisex(dit, [lev_a, lev_b])
                    elif (noise_x, noise_z) == (0, 1):
                        noisy_circuit.noisez(dit, lev_b)
                    elif (noise_x, noise_z) == (1, 1):
                        noisy_circuit.noisey(dit, [lev_a, lev_b])

    def _apply_dephasing_noise(
        self, noisy_circuit: QuantumCircuit, qudits: list[int], noise_info: Noise | SubspaceNoise
    ) -> None:
        """Applies dephasing noise to specified qudit levels outside the main depolarizing subspace levels.

        Args:
            noisy_circuit: Circuit to apply noise to
            qudits: List of qudits to apply noise to
            noise_info: Noise model information containing subspace probabilities

        Raises:
            IndexError: If subspace levels are outside qudit dimensions
        """
        if isinstance(noise_info, SubspaceNoise):  # Physical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                possible_levels = set(range(dim))  # Changed to set for efficient operations

                # Validate all subspace levels
                for lev_a, lev_b in noise_info.subspace_w_probs:
                    if lev_a not in possible_levels or lev_b not in possible_levels:
                        msg = (
                            "Subspace levels exceed qudit dimensions. "
                            f"Got levels ({lev_a}, {lev_b}) but dimension is {dim}. "
                            "Check noise model compatibility with circuit."
                        )
                        raise IndexError(msg)

                    # Calculate remaining levels for dephasing
                    subspace_levels = {lev_a, lev_b}
                    dephasing_levels = list(possible_levels - subspace_levels)

                    if not dephasing_levels:  # Check if we have levels for dephasing
                        continue

                    # Calculate probability for each level
                    prob_each = noise_info.subspace_w_probs[lev_a, lev_b].probability_dephasing
                    probs = [prob_each, 1 - prob_each]  # [apply noise, no noise]

                    # Apply dephasing to each level outside subspace
                    for physical_level in dephasing_levels:
                        if self.rng.choice([True, False], p=probs):
                            noisy_circuit.noisez(dit, physical_level)
