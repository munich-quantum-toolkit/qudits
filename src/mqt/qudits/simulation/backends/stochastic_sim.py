from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from ..noise_tools import NoiseModel, NoisyCircuitFactory
from ..save_info import save_full_states, save_shots

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...quantum_circuit import QuantumCircuit
    from . import MISim, TNSim
    from .backendv2 import Backend


def generate_seed() -> int:
    """Generate a random seed for numpy random generator."""
    current_time = int(time.time() * 1000)
    return hash((os.getpid(), current_time)) % 2**32


def measure_state(vector_data: NDArray[np.complex128]) -> int:
    """Measure the quantum state and return the result."""
    probabilities = np.abs(vector_data.ravel()) ** 2
    rng = np.random.default_rng(generate_seed())
    return rng.choice(len(probabilities), p=probabilities)


def save_results(backend: Backend, results: list[int | NDArray[np.complex128]]) -> None:
    """Save the simulation results based on backend configuration."""
    if backend.full_state_memory and backend.file_path and backend.file_name:
        save_full_states(results, backend.file_path, backend.file_name)
    elif backend.memory and backend.file_path and backend.file_name:
        save_shots(results, backend.file_path, backend.file_name)


def stochastic_simulation(backend: Backend, circuit: QuantumCircuit) -> NDArray | list[int]:
    noise_model: NoiseModel = NoiseModel()
    if backend.noise_model is not None:
        noise_model = backend.noise_model

    shots: int = backend.shots
    num_processes: int = mp.cpu_count()
    from . import MISim, TNSim

    with mp.Pool(processes=num_processes) as pool:
        if isinstance(backend, TNSim):
            factory = NoisyCircuitFactory(noise_model, circuit)
            args_tn = [(backend, factory) for _ in range(shots)]
            results = pool.map(stochastic_execution_tn, args_tn)
        elif isinstance(backend, MISim):
            args_mis = [(backend, circuit, noise_model) for _ in range(shots)]
            results = pool.map(stochastic_execution_mi, args_mis)
        else:
            msg = "Unsupported backend type"
            raise TypeError(msg)

    save_results(backend, results)
    return results


def stochastic_execution_tn(args: tuple[TNSim, NoisyCircuitFactory]) -> NDArray | int:
    backend, factory = args
    circuit = factory.generate_circuit()
    vector_data = backend.execute(circuit)
    return vector_data if backend.full_state_memory else measure_state(vector_data)


def stochastic_execution_mi(args: tuple[MISim, QuantumCircuit, NoiseModel]) -> NDArray | int:
    backend, circuit, noise_model = args
    vector_data = backend.execute(circuit, noise_model)
    return vector_data if backend.full_state_memory else measure_state(vector_data)
