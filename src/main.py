import numpy as np

from mqt.qudits.compiler.dit_manager import QuditManager
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData
from mqt.qudits.qudit_circuits.components.registers.quantum_register import QuantumRegister
from mqt.qudits.simulation.provider.noise_tools.noise import Noise, NoiseModel
from mqt.qudits.simulation.provider.qudit_provider import MQTQuditProvider
from mqt.qudits.visualisation.plot_information import plot_counts, plot_state
from mqt.qudits.visualisation.mini_quantum_information import get_density_matrix_from_counts, partial_trace

qasm = """
        DITQASM 2.0;
        qreg q [3][3,2,3];
        qreg bob [2][2,2];
        creg meas[3];
        h q[2] ctl q[0] bob[0] q[1] [0,0,0];
        cx q[2],q[1];
        cx q[1],q[0];
        rxy (0, 1, pi, pi/2) q[0] ctl  bob[0] q[1] [0,0];
        barrier q[0],q[1],q[2];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        """
c = QuantumCircuit()
c.from_qasm(qasm)
print(c.to_qasm())

# Program a quantum algorithm in python

qreg_example = QuantumRegister("reg", 2, [3, 3])
circ = QuantumCircuit(qreg_example)

# x = circ.x(1)
# s = circ.s(0)
# z = circ.z(1, ControlData([0], [1]))
h3 = circ.h(0)  # , ControlData([0], [1]))#.control([1], [1])
h3 = circ.h(0)
h3 = circ.h(0)
for i in range(200):
    r = circ.r(1, [0, 1, np.pi, np.pi / 2])
csum = circ.csum([0, 1])
# cx = circ.cx([0, 1],[0, 2, 1, 0.])
# rz = circ.rz(0, [0, 1, np.pi / 2])
# ls = circ.ls([0, 1], [np.pi / 4])
# ms = circ.ms([0, 1], [np.pi / 2])
ru = circ.randu([0, 1])
# p = circ.pm([0], [0, 2, 1])


print(circ.to_qasm())

# ------------------------------------------------------
# Noiselse simulation

provider = MQTQuditProvider()
backend = provider.get_backend("tnsim")

job = backend.run(circ)
result = job.result()
plot_state(result.get_state_vector(), circ)

backend_ion = provider.get_backend("faketraps2trits", shots=1000)

job = backend_ion.run(circ)
result = job.result()
plot_counts(result.get_counts(), circ)

# Evaluate

rho = get_density_matrix_from_counts(result.get_counts(), circ)
print(np.trace(rho @ rho))
print(partial_trace(rho, [1], [3, 3]))

# WE compile the circuit and notice new differences
qudit_compiler = QuditManager()
passes = ["LocAdaPass", "ZPropagationPass", "ZRemovalPass"]
pulse_level_circuit = qudit_compiler.compile(backend_ion, circ, passes)


job = backend_ion.run(pulse_level_circuit)
result = job.result()
plot_counts(result.get_counts(), circ)
