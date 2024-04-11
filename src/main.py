from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.qudit_circuits.components.registers.quantum_register import QuantumRegister
from mqt.qudits.simulation.provider.backends.stocastic_components.stocastic_sim import stocastic_execution_misim
from mqt.qudits.simulation.provider.noise_tools.noise import Noise, NoiseModel
from mqt.qudits.simulation.provider.qudit_provider import MQTQuditProvider
from mqt.qudits.visualisation.plot_information import plot_counts, plot_state
import numpy as np

qreg_example = QuantumRegister("reg", 10, [7, 7, 7, 7, 7, 7, 7, 7, 7, 7])
circ = QuantumCircuit(qreg_example)

rz = circ.rz(1, [0, 2, np.pi / 13])
x = circ.x(0)
s = circ.s(1)
"""
z = circ.z(0)

vrz = circ.virtrz(0, [0, np.pi / 13])
vrz = circ.virtrz(0, [1, -np.pi / 8])
vrz = circ.virtrz(1, [1, -np.pi / 8])

x = circ.x(0)
x = circ.x(0)
z = circ.z(1)
z = circ.z(1)
h = circ.h(1)
rz = circ.rz(1, [3, 4, np.pi / 13])
h = circ.h(0)
r = circ.r(0, [0, 1, np.pi / 5 + np.pi, np.pi / 7])
"""
# h = circ.h(0)
#for i in range(200):
#    r = circ.r(1, [0, 4, np.pi, np.pi / 2])
#r2 = circ.r(0, [0, 5, np.pi / 5, np.pi / 7]).dag()
# h = circ.h(0)
# x = circ.x(0).control([2], [2])
# print(h.to_matrix().round(3))
# print(x.to_matrix().round(3))
cx = circ.cx([1, 0], [0, 1, 1, np.pi / 2])
cx2 = circ.cx([1, 0], [0, 3, 0, np.pi / 12])
csum = circ.csum([0, 1])
# print(csum.to_matrix(2).round(3))
# print(x.to_matrix().round(3) == csum.to_matrix().round(3))
# ------------------------------------------------------
# Noiselse simulation

provider = MQTQuditProvider()
backend = provider.get_backend("misim")
job = backend.run(circ)
result1 = job.result()

plot_state(result1.get_state_vector(), circ)

provider = MQTQuditProvider()
backend = provider.get_backend("tnsim")
#job = backend.run(circ)
#result2 = job.result()
#plot_state(result2.get_state_vector(), circ)

print(" ARE THEY EQUAL??")
print(result1.get_state_vector().round(4) == result2.get_state_vector().round(4))
print(result1.get_state_vector().round(4))
#print(result2.get_state_vector().round(4))

#########################################################################################

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

provider = MQTQuditProvider()
backend = provider.get_backend("misim")
job = backend.run(circ, noise_model=noise_model)
result = job.result()
counts = result.get_counts()
print(counts)
plot_counts(result.get_counts(), circ)

provider = MQTQuditProvider()
backend = provider.get_backend("tnsim")
#job = backend.run(circ, noise_model=noise_model)
#result = job.result()
#counts = result.get_counts()
#print(counts)
#plot_counts(result.get_counts(), circ)

#from mqt.qudits.compiler.dit_manager import QuditManager

backend_ion = provider.get_backend("faketraps2trits", shots=1000)
#qudit_compiler = QuditManager()

passes = ["LocQRPass"]
#compiled_circuit_qr = qudit_compiler.compile(backend_ion, circ, passes)

#print(
#    f"\n Number of operations: {len(compiled_circuit_qr.instructions)}, \n Number of qudits in the circuit: {compiled_circuit_qr.num_qudits}")
# print(r.to_matrix(2))
# plot_state(result.get_state_vector(), circ)


# h3 = circ.h(0)  # , ControlData([0], [1]))#.control([1], [1])
# print(h3.to_matrix().round(2))
# h3 = circ.h(0)
# h3 = circ.h(0)
# for _i in range(200):
# r = circ.r(1, [0, 1, np.pi, np.pi / 2])
# r2 = circ.r(1, [1, 2, np.pi/3, np.pi / 2])
# csum = circ.csum([0, 1])
# rz = circ.rz(0, [0, 2, np.pi / 13]).to_matrix().round(3)
# phase = np.exp(-1j * np.pi / 26)
# ls = circ.ls([0, 1], [np.pi / 4])
# ms = circ.ms([0, 1], [np.pi / 2])
# ru = circ.randu([0, 1])
# p = circ.pm([0], [0, 2, 1])
"""

"""
# cu2 = np.load('/home/k3vn/Documents/Lattice/mult.npy')
# circ.cu_two([0,1],cu2)
# print(circ.to_qasm())

#########################################

# backend_ion = provider.get_backend("faketraps2six", shots=1000)

# job = backend_ion.run(circ)
# result = job.result()
# plot_counts(result.get_counts(), circ)

# Evaluate


# rho = get_density_matrix_from_counts(result.get_counts(), circ)
# print(np.trace(rho @ rho))
# print(partial_trace(rho, [1], [3, 3]))

# WE compile the circuit and notice new differences
# qudit_compiler = QuditManager()
# passes = ["LocAdaPass", "ZPropagationPass", "ZRemovalPass"]
# pulse_level_circuit = qudit_compiler.compile(backend_ion, circ, passes)

# job = backend_ion.run(pulse_level_circuit)
# result = job.result()
# plot_counts(result.get_counts(), circ)
