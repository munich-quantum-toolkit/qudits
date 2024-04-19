from __future__ import annotations

import copy
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from ...quantum_circuit import QuantumCircuit, gates
from ...quantum_circuit.gate import GateTypes

if TYPE_CHECKING:
    from .noise import NoiseModel


class NoisyCircuitFactory:
    def __init__(self, noise_model: NoiseModel, circuit: QuantumCircuit) -> None:
        self.noise_model = noise_model
        self.circuit = circuit

    def generate_circuit(self):
        current_time = int(time.time() * 1000)
        seed = hash((os.getpid(), current_time)) % 2**32
        gen = np.random.Generator(np.random.PCG64(seed=seed))

        # num_qudits, dimensions_slice, numcl
        noisy_circuit = QuantumCircuit(self.circuit.num_qudits, self.circuit._dimensions, self.circuit._num_cl)
        noisy_circuit.number_gates = 0
        for instruction in self.circuit.instructions:
            # Deep copy the instruction
            copied_instruction = copy.deepcopy(instruction)
            # Append the deep copied instruction to the new circuit
            noisy_circuit.instructions.append(copied_instruction)
            noisy_circuit.number_gates += 1

            # Append an error depending on the prob
            if instruction.qasm_tag in self.noise_model.quantum_errors:
                for mode, noise_info in self.noise_model.quantum_errors[instruction.qasm_tag].items():
                    x_prob = [(1 - noise_info.probability_depolarizing), noise_info.probability_depolarizing]
                    z_prob = [(1 - noise_info.probability_dephasing), noise_info.probability_dephasing]
                    x_choice = gen.choice(a=range(2), p=x_prob)
                    z_choice = gen.choice(a=range(2), p=z_prob)
                    if x_choice == 1 or z_choice == 1:
                        qudits = None
                        if isinstance(mode, (list, tuple)):
                            qudits = list(mode)
                        elif isinstance(mode, str):
                            if mode == "local":
                                qudits = instruction.reference_lines
                            elif mode == "all":
                                qudits = list(range(instruction.parent_circuit.num_qudits))
                            elif mode == "nonlocal":
                                assert instruction.gate_type in {GateTypes.TWO, GateTypes.MULTI}
                                qudits = instruction.reference_lines
                            elif mode == "control":
                                assert instruction.gate_type == GateTypes.TWO
                                qudits = instruction._target_qudits[:1]
                            elif mode == "target":
                                assert instruction.gate_type == GateTypes.TWO
                                qudits = instruction._target_qudits[1:]
                        else:
                            pass

                        if x_choice == 1:
                            if isinstance(instruction, (gates.R, gates.Rz)):
                                for dit in qudits:
                                    noisy_circuit.r(dit, [instruction.lev_a, instruction.lev_b, np.pi, np.pi / 2])
                            else:
                                for dit in qudits:
                                    noisy_circuit.x(dit)
                            noisy_circuit.number_gates += 1
                        if z_choice == 1:
                            if isinstance(instruction, (gates.R, gates.Rz)):
                                for dit in qudits:
                                    noisy_circuit.rz(dit, [instruction.lev_a, instruction.lev_b, np.pi])
                            else:
                                for dit in qudits:
                                    noisy_circuit.z(dit)
                            noisy_circuit.number_gates += 1

        return noisy_circuit
