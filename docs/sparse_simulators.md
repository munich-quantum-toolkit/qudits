# Sparse Matrix Simulators

MQT Qudits includes memory-efficient sparse matrix simulators with optional GPU acceleration for large-scale quantum circuit simulation.

## Overview

Sparse matrix simulators use `scipy.sparse` (CPU) or `cupyx.scipy.sparse` (GPU) to represent quantum gates and states efficiently. This is particularly beneficial for:

- **Large circuits** with many qudits (>10)
- **Sparse gate matrices** (gates affecting only a few levels)
- **Memory-constrained environments**
- **GPU-accelerated computation** (when available)

## Available Backends

### SparseStatevecSim

Memory-efficient state vector simulator using sparse matrix operations.

**Features:**

- Simulates quantum state evolution using sparse matrices
- Returns final state vector
- Supports all gate types
- CPU and GPU modes

**Usage:**

```python
from mqt.qudits.simulation import MQTQuditProvider

provider = MQTQuditProvider()
backend = provider.get_backend("sparse_statevec")

# CPU mode (default)
job = backend.run(circuit)
state_vector = job.result().get_state_vector()

# GPU mode (requires CuPy)
job_gpu = backend.run(circuit, use_gpu=True)
state_vector_gpu = job_gpu.result().get_state_vector()
```

### SparseUnitarySim

Memory-efficient unitary matrix constructor for circuit analysis.

**Features:**

- Constructs the full circuit unitary matrix using sparse operations
- Returns the unitary matrix representation
- Useful for circuit analysis and verification
- CPU and GPU modes

**Usage:**

```python
from mqt.qudits.simulation import MQTQuditProvider

provider = MQTQuditProvider()
backend = provider.get_backend("sparse_unitary")

# Construct circuit unitary
job = backend.run(circuit)
unitary = job.result().get_state_vector()

# Verify unitarity
import numpy as np

is_unitary = np.allclose(unitary @ unitary.conj().T, np.eye(unitary.shape[0]))
print(f"Is unitary: {is_unitary}")
```

## GPU Acceleration

### Requirements

- **Hardware:** NVIDIA GPU with CUDA support
- **Software:** CuPy is automatically installed on Linux and Windows systems

### Usage

Simply add `use_gpu=True` to the `run()` method:

```python
# GPU-accelerated sparse state vector simulation
backend = provider.get_backend("sparse_statevec")
job = backend.run(circuit, use_gpu=True)
result = job.result()
```

**Error Handling:**

If GPU is unavailable, the simulator will raise an error:

```python
try:
    job = backend.run(circuit, use_gpu=True)
except (ImportError, RuntimeError) as e:
    print(f"GPU not available: {e}")
    # Fallback to CPU
    job = backend.run(circuit, use_gpu=False)
```

## Performance Considerations

### When to Use Sparse Simulators

**✅ Good for:**

- Circuits with >10 qudits
- Sparse gate matrices (most single-qudit gates)
- Long-range entangling gates
- Memory-constrained systems
- GPU-equipped machines (for large circuits)

**❌ Less optimal for:**

- Small circuits (<10 qudits) - dense matrices may be faster
- Very dense gate matrices
- Systems without sufficient memory even for sparse representation

### CPU vs GPU

| Aspect       | CPU (NumPy/SciPy)              | GPU (CuPy)                   |
| ------------ | ------------------------------ | ---------------------------- |
| **Setup**    | No additional dependencies     | Requires CuPy + NVIDIA GPU   |
| **Memory**   | System RAM                     | GPU VRAM                     |
| **Speed**    | Good for small/medium circuits | Excellent for large circuits |
| **Best for** | <15 qudits                     | >15 qudits                   |

## Comparison with Other Simulators

| Backend           | Data Structure    | Gate Support      | Memory Efficiency | GPU Support |
| ----------------- | ----------------- | ----------------- | ----------------- | ----------- |
| `tnsim`           | Tensor Network    | All gates         | Medium            | ❌          |
| `misim`           | Decision Diagrams | Limited gateset\* | High              | ❌          |
| `sparse_statevec` | Sparse Matrices   | All gates         | High              | ✅          |
| `sparse_unitary`  | Sparse Matrices   | All gates         | High              | ✅          |

\*MISim supports: csum, cx, h, rxy, rz, rh, virtrz, s, x, z

## Examples

### Example 1: Basic Sparse Simulation

```python
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider
import numpy as np

# Create a 3-qudit circuit
qreg = QuantumRegister("test", 3, [3, 3, 3])
circuit = QuantumCircuit(qreg)

# Add gates
circuit.h(0)
circuit.csum([0, 1])
circuit.rz(2, [0, 1, np.pi / 4])

# Simulate with sparse backend
provider = MQTQuditProvider()
backend = provider.get_backend("sparse_statevec")
job = backend.run(circuit)

# Get results
state_vector = job.result().get_state_vector()
print(f"State vector norm: {np.linalg.norm(state_vector)}")  # Should be 1.0
```

### Example 2: Unitary Matrix Analysis

```python
# Analyze circuit unitary
backend_unitary = provider.get_backend("sparse_unitary")
job = backend_unitary.run(circuit)
unitary = job.result().get_state_vector()

# Verify properties
print(f"Unitary shape: {unitary.shape}")
print(
    f"Is unitary: {np.allclose(unitary @ unitary.conj().T, np.eye(unitary.shape[0]))}"
)
print(f"Determinant magnitude: {np.abs(np.linalg.det(unitary))}")  # Should be ~1.0
```

### Example 3: GPU-Accelerated Large Circuit

```python
# Create a larger circuit
qreg_large = QuantumRegister("large", 15, [2] * 15)
circuit_large = QuantumCircuit(qreg_large)

for i in range(14):
    circuit_large.h(i)
    circuit_large.csum([i, i + 1])

# Simulate on GPU
try:
    backend = provider.get_backend("sparse_statevec")
    job = backend.run(circuit_large, use_gpu=True)
    result = job.result()
    print("✅ GPU simulation successful!")
except ImportError:
    print("❌ CuPy not available, falling back to CPU...")
    job = backend.run(circuit_large, use_gpu=False)
    result = job.result()
```

### Example 4: Comparing Sparse vs Dense

```python
# All gates can generate both sparse and dense matrices
gate = circuit.h(0)

# Dense matrix (NumPy array)
dense_matrix = gate.to_matrix(identities=0, sparse=False)
print(f"Dense type: {type(dense_matrix)}")

# Sparse matrix (scipy.sparse.csr_matrix)
sparse_matrix = gate.to_matrix(identities=0, sparse=True)
print(f"Sparse type: {type(sparse_matrix)}")

# Verify equivalence
assert np.allclose(sparse_matrix.toarray(), dense_matrix)
print("✅ Sparse and dense matrices are equivalent")
```

### Example 5: Noisy Simulation

The sparse state vector simulator supports noise models for realistic quantum simulations:

```python
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel, SubspaceNoise

# Create a circuit
qreg = QuantumRegister("noisy", 2, [3, 3])
circuit = QuantumCircuit(qreg)
circuit.h(0)
circuit.cx([0, 1])
circuit.rz(0, [np.pi / 4])

# Mathematical (uniform) noise model
noise_model = NoiseModel()
local_error = Noise(probability_depolarizing=0.01, probability_dephasing=0.01)
entangling_error = Noise(probability_depolarizing=0.05, probability_dephasing=0.02)

noise_model.add_quantum_error_locally(local_error, ["h", "rz"])
noise_model.add_nonlocal_quantum_error(entangling_error, ["cx"])

# Run noisy simulation (requires ≥50 shots)
backend = provider.get_backend("sparse_statevec")
job = backend.run(circuit, noise_model=noise_model, shots=100)
result = job.result()

# Analyze results
print(f"Number of measurement outcomes: {len(result.counts)}")
print(f"Outcome distribution: {set(result.counts)}")
```

### Example 6: Physical Noise with SubspaceNoise

For more realistic simulations, use `SubspaceNoise` to model noise on specific energy level transitions:

```python
# Physical noise model
physical_noise = NoiseModel()

# Noise specific to the 0↔1 transition (ground state operations)
ground_noise = SubspaceNoise(
    probability_depolarizing=0.001, probability_dephasing=0.001, levels=(0, 1)
)

# Noise for higher energy levels (typically more error-prone)
excited_noise = SubspaceNoise(
    probability_depolarizing=0.005, probability_dephasing=0.003, levels=[(1, 2), (2, 3)]
)

# Dynamic noise (automatically assigned to gate's active subspace)
dynamic_noise = SubspaceNoise(
    probability_depolarizing=0.002, probability_dephasing=0.001, levels=[]
)

# Apply to gates
physical_noise.add_quantum_error_locally(ground_noise, ["x", "h"])
physical_noise.add_quantum_error_locally(excited_noise, ["z", "s"])
physical_noise.add_quantum_error_locally(dynamic_noise, ["rz", "rh"])
physical_noise.add_nonlocal_quantum_error(dynamic_noise, ["cx", "ls"])

# Run with physical noise
qreg = QuantumRegister("physical", 3, [4, 4, 4])
circuit = QuantumCircuit(qreg)
for i in range(3):
    circuit.h(i)
circuit.cx([0, 1])
circuit.cx([1, 2])

backend = provider.get_backend("sparse_statevec")
job = backend.run(circuit, noise_model=physical_noise, shots=100)
result = job.result()

print("Physical noise simulation completed!")
```

**Note:** `SparseUnitarySim` does not support noise models (it computes ideal unitaries). If you pass a `noise_model` to it, a warning will be raised and the ideal unitary will be computed.

## Troubleshooting

### CUDA Out of Memory

**Solutions:**

- Reduce circuit size
- Use CPU mode instead
- Close other GPU applications
- Upgrade to a GPU with more VRAM

### Slow Performance on CPU

**Solutions:**

- Enable GPU acceleration if available
- Reduce circuit size
- Use `misim` for supported gate types
- Consider circuit decomposition

## API Reference

### SparseStatevecSim.run()

**Signature:** `run(circuit, use_gpu=False, **options) -> Job`

**Parameters:**

- `circuit`: Quantum circuit to simulate
- `use_gpu`: Enable GPU acceleration (default: False)
- `**options`: Additional backend options:
  - `noise_model`: NoiseModel for noisy simulation (default: None)
  - `shots`: Number of measurement shots (default: 50 if noise, else 1)
  - `memory`: Save individual shot outcomes (default: False)
  - `full_state_memory`: Save full state vector for each shot (default: False)
  - `file_path`: Path to save results (default: None)
  - `file_name`: Name for saved results file (default: None)

**Returns:** Job object with simulation results

**Note:** Noise simulation requires `shots >= 50` and uses multiprocessing for parallel execution.

### SparseUnitarySim.run()

**Signature:** `run(circuit, use_gpu=False, **options) -> Job`

**Parameters:**

- `circuit`: Quantum circuit to construct unitary for
- `use_gpu`: Enable GPU acceleration (default: False)
- `**options`: Additional backend options

**Returns:** Job object with unitary matrix

## See Also

- [Tutorial](tutorial.md) - Complete MQT Qudits tutorial
- [API Reference](api/mqt/qudits/index.rst) - Full API documentation
- [GitHub Issues](https://github.com/cda-tum/mqt-qudits/issues) - Report bugs or request features
