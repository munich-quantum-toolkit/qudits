from __future__ import annotations

import numpy as np
import pytest

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.quantum_circuit.gate import Gate
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.simulation import MQTQuditProvider

class StateSetGate(Gate):
    """
    A strictly Unitary gate that acts as a generalized X gate.
    It permutes |0> to |target_level>.
    """
    def __init__(self, circuit: QuantumCircuit, target_qudit: int, dimension: int, target_level: int):
        super().__init__(
            circuit=circuit,
            name=f"set_state_{target_level}",
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudit,
            dimensions=dimension,
        )
        self._dimension = dimension
        self._target_level = target_level

    def __array__(self) -> np.ndarray:
        # Create a permutation matrix (Identity with swapped columns)
        # This guarantees the matrix is Unitary.
        matrix = np.eye(self._dimension, dtype=np.complex128)
        
        # We want |0> to go to |target_level>
        # So we swap column 0 with column target_level
        if self._target_level != 0:
            matrix[:, [0, self._target_level]] = matrix[:, [self._target_level, 0]]
            
        return matrix

@pytest.fixture
def provider():
    return MQTQuditProvider()

def test_mixed_dims_endianness_check(provider):
    """
    CRITICAL ARCHITECTURE TEST:
    Verifies that the simulator uses Big-Endian ordering (First Register = MSB).
    
    Setup:
    - Qudit 0 (Reg A): Dim 3, State set to |1>
    - Qudit 1 (Reg B): Dim 4, State set to |2>
    
    Mathematical Expectation:
    If Big Endian (Reg A is MSB):
        Index = (State_A * Dim_B) + State_B
        Index = (1 * 4) + 2 = 6
        
    If Little Endian (Reg A is LSB):
        Index = (State_B * Dim_A) + State_A
        Index = (2 * 3) + 1 = 7
    """
    backend = provider.get_backend("sparse_statevec")
    
    # Create Mixed Dimensions
    dim_A = 3
    dim_B = 4
    
    reg_a = QuantumRegister("A", 1, [dim_A])
    reg_b = QuantumRegister("B", 1, [dim_B])
    
    circuit = QuantumCircuit()
    circuit.append(reg_a) # Added first -> MSB
    circuit.append(reg_b) # Added last -> LSB
    
    # Apply gates to set specific states
    # Set Reg A (Qudit 0) to state |1>
    circuit.instructions.append(StateSetGate(circuit, 0, dim_A, target_level=1))
    
    # Set Reg B (Qudit 1) to state |2>
    # Note: In the circuit list, Qudit 1 is the second wire
    circuit.instructions.append(StateSetGate(circuit, 1, dim_B, target_level=2))
    
    # Run Simulation
    job = backend.run(circuit)
    state_vector = job.result().get_state_vector().flatten()
    
    # Find the index of the non-zero amplitude
    non_zero_indices = np.nonzero(state_vector)[0]
    
    assert len(non_zero_indices) == 1, f"Expected distinct state, found superposition at {non_zero_indices}"
    
    found_index = non_zero_indices[0]
    expected_index_big_endian = (1 * dim_B) + 2
    
    error_msg = (
        f"Endianness Mismatch!\n"
        f"Found index: {found_index}\n"
        f"Expected (Big Endian): {expected_index_big_endian}\n"
        f"Note: If found index is 7, your simulator is Little Endian."
    )
    
    assert found_index == expected_index_big_endian, error_msg

def test_backends_agree_on_layout(provider):
    """
    Ensures that Sparse and TNSim use the exact same memory layout.
    """
    sparse = provider.get_backend("sparse_statevec")
    tnsim = provider.get_backend("tnsim")
    
    # Use prime number dimensions to avoid accidental symmetries
    dim_0 = 2
    dim_1 = 3 
    dim_2 = 2
    
    qreg = QuantumRegister("test", 3, [dim_0, dim_1, dim_2])
    circuit = QuantumCircuit(qreg)
    
    # Set a specific state: |1, 2, 0>
    circuit.instructions.append(StateSetGate(circuit, 0, dim_0, 1))
    circuit.instructions.append(StateSetGate(circuit, 1, dim_1, 2))
    # Qudit 2 stays at 0
    
    sparse_result = sparse.run(circuit).result().get_state_vector().flatten()
    tnsim_result = tnsim.run(circuit).result().get_state_vector().flatten()
    
    assert np.allclose(sparse_result, tnsim_result, atol=1e-12), "Backends disagree on state vector layout"