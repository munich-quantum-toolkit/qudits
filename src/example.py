from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.qudit_circuits.components.registers.quantum_register import QuantumRegister
from mqt.qudits.simulation.provider.qudit_provider import MQTQuditProvider
from mqt.qudits.visualisation.plot_information import plot_counts, plot_state
from mqt.qudits.visualisation.mini_quantum_information import get_density_matrix_from_counts, partial_trace
from mqt.qudits.compiler.dit_manager import QuditManager

# In[2]:


qasm = """
        DITQASM 2.0;

        qreg fields [3][5,5,5];
        qreg matter [2][2,2];

        creg meas_matter[2];
        creg meas_fields[3];

        h fields[2] ctl matter[0] matter[1] [0,0];
        cx fields[2], matter[0];
        cx fields[2], matter[1];
        rxy (0, 1, pi, pi/2) fields[2];
        barrier q[0],q[1],q[2];

        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        """
# Control syntax: operation   ctl  qudit_line  [list of qudit control levels]

circuit = QuantumCircuit()
circuit.from_qasm(qasm)

print(f"\n Number of operations: {len(circuit.instructions)}, \n Number of qudits in the circuit: {circuit.num_qudits}")

# In[3]:


circuit = QuantumCircuit()

field_reg = QuantumRegister("fields", 1, [3])
ancilla_reg = QuantumRegister("ancillas", 1, [3])

circuit.append(field_reg)
circuit.append(ancilla_reg)

print(f"\n Number of operations: {len(circuit.instructions)}, \n Number of qudits in the circuit: {circuit.num_qudits}")

# In[4]:


h = circuit.h(field_reg[0])

# Syntax for controlled operations is :

# h = circuit.h(field_reg[0], ControlData([control_register], [controls_levels]))
# OR
# h = circuit.h(field_reg[0]).control([control_register], [controls_levels])


# In[5]:


csum = circuit.csum([field_reg[0], ancilla_reg[0]])

# In[6]:


print(f"\n Number of operations: {len(circuit.instructions)}, \n Number of qudits in the circuit: {circuit.num_qudits}")

# In[7]:


print(circuit.to_qasm())

# In[8]:


provider = MQTQuditProvider()
provider.backends("sim")

# In[9]:


backend = provider.get_backend("tnsim")

job = backend.run(circuit)
result = job.result()

state_vector = result.get_state_vector()

plot_state(state_vector, circuit)

# In[10]:


backend_ion = provider.get_backend("faketraps2trits", shots=1000)

job = backend_ion.run(circuit)
result = job.result()
counts = result.get_counts()

plot_counts(counts, circuit)

# In[11]:


rho = get_density_matrix_from_counts(counts, circuit)
print(partial_trace(rho, qudits2keep=[0], dims=[3, 3]))

# In[12]:


# the compiler uses energy level graph of the architecture

qudit_compiler = QuditManager()
passes = ["LocQRPass"]

# In[13]:


compiled_circuit_qr = qudit_compiler.compile(backend_ion, circuit, passes)

print(
        f"\n Number of operations: {len(compiled_circuit_qr.instructions)}, \n Number of qudits in the circuit: {compiled_circuit_qr.num_qudits}")

# In[18]:


job = backend_ion.run(compiled_circuit_qr)

result = job.result()
counts = result.get_counts()

plot_counts(counts, compiled_circuit_qr)

# In[15]:


passes = ["LocAdaPass", "ZPropagationPass","ZRemovalPass"]

compiled_circuit_ada = qudit_compiler.compile(backend_ion, circuit, passes)

print(f"\n Number of operations: {len(compiled_circuit_ada.instructions)}, \n Number of qudits in the circuit: {compiled_circuit_ada.num_qudits}")

# In[17]:


job = backend_ion.run(compiled_circuit_ada)

result = job.result()
counts = result.get_counts()

plot_counts(counts, compiled_circuit_ada)

# In[ ]:


# In[ ]:
