"""Example: Using YAQSSim as a backend for MQT Qudits circuits."""

from __future__ import annotations

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel

provider = MQTQuditProvider()
backend = provider.get_backend("yaqssim")

dims = [3, 4, 2, 4, 3]
qreg = QuantumRegister("reg", len(dims), dims)
circuit = QuantumCircuit(qreg)
circuit.h(0)
circuit.h(2)
circuit.csum([0, 1])
circuit.csum([0, 4])
circuit.h(4)

noise_model = NoiseModel()
noise_model.add_quantum_error_locally(Noise(0.1, 0.05), ["h"])
noise_model.add_quantum_error_locally(Noise(0.05, 0.0), ["csum"])

job = backend.run(circuit)
rho = job.result().get_density_matrix()

print(f"Noiseless density matrix: {rho}")

job_noisy = backend.run(circuit, noise_model=noise_model, shots=300)
rho_noisy = job_noisy.result().get_density_matrix()

print(f"Noiseless density matrix: {rho_noisy}")
