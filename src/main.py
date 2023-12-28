from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.qudit_circuits.components.registers.quantum_register import QuantumRegister
from mqt.qudits.simulation.provider.qudit_provider import MQTQuditProvider
from mqt.qudits.visualisation.run_info import plot_histogram

qasm = """
        DITQASM 2.0;
        qreg q [3][3,2,3];
        qreg bob [2][2,2];
        creg meas[3];
        h q[2];
        cx q[2],q[1];
        cx j[1],q[0];
        rxy ( pi, pi/2 , 0, 1) q[0];
        barrier q[0],q[1],q[2];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        """
# c = QuantumCircuit()
# c.from_qasm(qasm)
# print(QASM().parse_ditqasm2_str(qasm))
qreg_example = QuantumRegister("x", 3, [3, 2, 2])
circ = QuantumCircuit(qreg_example)
# circ.append(QuantumRegister())
# circ.from_qasm(qasm)
h3 = circ.h(0)
# x = circ.x(0)
# csum = circ.csum([0, 2])

# print(h3.to_matrix(identities=1))
# print(csum.to_matrix(identities=1))


provider = MQTQuditProvider()
print(provider.backends("sim"))

backend = provider.get_backend("tnsim")
result = backend.run(circ)

state_size = 1
for s in circ.dimensions:
    state_size *= s

result = result.tensor.reshape(1, state_size)

# takes a vector in
plot_histogram(result, circ)
