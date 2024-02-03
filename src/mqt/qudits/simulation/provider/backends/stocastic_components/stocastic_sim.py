import os
import time

import numpy as np
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.simulation.data_log.save_info import save_full_states, save_shots
from mqt.qudits.simulation.provider.backends.backendv2 import Backend
import multiprocessing as mp
from mqt.qudits.simulation.provider.noise_tools.noisy_circuit_factory import NoisyCircuitFactory


def stocastic_simulation(backend: Backend, circuit: QuantumCircuit):
    noise_model = backend.noise_model
    shots = backend.shots
    num_processes = mp.cpu_count()
    factory = NoisyCircuitFactory(noise_model, circuit)

    with mp.Pool(processes=num_processes) as process:
        execution_args = [(backend, factory) for _ in range(shots)]
        results = process.map(stocastic_execution, execution_args)

    if backend.full_state_memory:
        filepath = backend.file_path
        filename = backend.file_name
        save_full_states(results, filepath, filename)
    elif backend.memory:
        filepath = backend.file_path
        filename = backend.file_name
        save_shots(results, filepath, filename)

    return results


def stocastic_execution(args):
    backend, factory = args
    circuit = factory.generate_circuit()
    vector_data = backend.execute(circuit)

    if not backend.full_state_memory:
        current_time = int(time.time() * 1000)
        seed = hash((os.getpid(), current_time)) % 2 ** 32
        gen = np.random.Generator(np.random.PCG64(seed=seed))
        vector_data = np.ravel(vector_data)
        probabilities = [abs(x) ** 2 for x in vector_data]
        idx = gen.choice(a=range(len(probabilities)), p=probabilities)
        return idx

    return vector_data
