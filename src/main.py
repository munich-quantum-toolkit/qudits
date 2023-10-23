from mqt.circuit.circuit import QuantumCircuit
from mqt.circuit.components.registers.quantum_register import QuantumRegister
from mqt.simulation.provider.mqtvider import MQTProvider

qasm = """
        DITQASM 2.0;
        qreg q [3][3,2,3];
        qreg j [2][2,2];
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
qreg_example = QuantumRegister("x", 2, [3, 3])
circ = QuantumCircuit(qreg_example)
# circ.from_qasm(qasm)
h3 = circ.h(0)
print(h3.to_matrix())

provider = MQTProvider()
print(provider.backends("sim"))

backend = provider.get_backend("tnsim")
result = backend.run(circ)

state_size = 1
for s in circ.dimensions:
    state_size *= s

print(result.tensor.reshape(1, state_size))
