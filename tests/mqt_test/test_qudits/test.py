import mqt.misim.pymisim as misim
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.qudit_circuits.components.registers.quantum_register import QuantumRegister
from mqt.qudits.simulation.provider.noise_tools.noise import Noise, NoiseModel

import numpy as np

qreg_example = QuantumRegister("reg", 2, [2, 3])
circ = QuantumCircuit(qreg_example)
h = circ.h(0)
csum = circ.csum([0, 1])
#x = circ.x(1).control([0], [1])
#r = circ.r(0, [0, 1, np.pi/3, -np.pi / 2])
#r2 = circ.r(0, [0, 1, np.pi, np.pi / 2]).control([1], [1])

# Depolarizing quantum errors
local_error = Noise(probability_depolarizing=0.001, probability_dephasing=0.001)
local_error_rz = Noise(probability_depolarizing=0.03, probability_dephasing=0.03)
entangling_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.001)
entangling_error_extra = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
entangling_error_on_target = Noise(probability_depolarizing=0.1, probability_dephasing=0.0)
entangling_error_on_control = Noise(probability_depolarizing=0.01, probability_dephasing=0.0)

# Add errors to noise_tools model

noise_model = NoiseModel()  # We know that the architecture is only two qudits
# Very noisy gate
noise_model.add_all_qudit_quantum_error(local_error, ["csum"])
noise_model.add_recurrent_quantum_error_locally(local_error, ["csum"], [0])
# Entangling gates
noise_model.add_nonlocal_quantum_error(entangling_error, ["cx", "ls", "ms"])
noise_model.add_nonlocal_quantum_error_on_target(entangling_error_on_target, ["cx", "ls", "ms"])
noise_model.add_nonlocal_quantum_error_on_control(entangling_error_on_control, ["csum", "cx", "ls", "ms"])
# Super noisy Entangling gates
noise_model.add_nonlocal_quantum_error(entangling_error_extra, ["csum"])
# Local Gates
noise_model.add_quantum_error_locally(local_error, ["h", "rxy", "s", "x", "z"])
noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])

print("HELLOO")

# num_qudits, dimensions, props = qcread.get_quantum_circuit_properties(circ)
print(misim.state_vector_simulation(circ, NoiseModel()))
