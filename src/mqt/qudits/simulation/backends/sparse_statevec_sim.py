"""Sparse State Vector Simulator for quantum circuits.

This module provides a memory-efficient state vector simulator using sparse matrix representations.
Supports both CPU (NumPy/SciPy) and GPU (CuPy) backends.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix  # type: ignore[import-not-found]
from typing_extensions import Unpack

from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider

# Try to import CuPy for GPU support (optional)
try:
    import cupy as cp  # type: ignore[import-not-found]
    import cupyx.scipy.sparse as cp_sparse  # type: ignore[import-not-found]

    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    cp_sparse = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False


class SparseStatevecSim(Backend):
    """Sparse state vector simulator.

    Simulates quantum circuits using sparse matrix operations for memory efficiency.
    Supports both CPU (NumPy/SciPy) and GPU (CuPy) backends.
    Best suited for circuits with sparse gate matrices.

    Examples:
        >>> from mqt.qudits.simulation import MQTQuditProvider
        >>> provider = MQTQuditProvider()
        >>> # CPU backend
        >>> backend = provider.get_backend("sparse_statevec")
        >>> job = backend.run(circuit)
        >>> # GPU backend (requires CuPy)
        >>> job = backend.run(circuit, use_gpu=True)
        >>> result = job.result()
        >>> state_vector = result.get_state_vector()
    """

    def __init__(
        self,
        provider: MQTQuditProvider,
        name: str | None = None,
        description: str | None = None,
        **fields: Unpack[Backend.DefaultOptions],
    ) -> None:
        """Initialize the sparse state vector simulator.

        Args:
            provider: The provider instance
            name: Backend name (default: "sparse_statevec")
            description: Backend description
            **fields: Additional backend options
        """
        if name is None:
            name = "sparse_statevec"
        if description is None:
            description = "Sparse matrix state vector simulator (CPU/GPU) for memory-efficient simulation"
        super().__init__(provider, name=name, description=description, **fields)
        self.use_gpu = False
        self.xp: Any = np  # Will be np or cp
        self.sp: Any = None  # Will be scipy.sparse or cupyx.scipy.sparse

    def run(self, circuit: QuantumCircuit, use_gpu: bool = False, **options: Unpack[Backend.DefaultOptions]) -> Job:
        """Run the quantum circuit simulation.

        Args:
            circuit: The quantum circuit to simulate
            use_gpu: Use GPU acceleration with CuPy (default: False)
            **options: Additional simulation options (shots, memory, etc.)

        Returns:
            Job: A job object containing the simulation results

        Raises:
            ImportError: If use_gpu=True but CuPy is not installed
        """
        job = Job(self)

        self._options.update(options)
        self.shots = self._options.get("shots", 1)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)

        # Configure backend (CPU or GPU)
        self.use_gpu = use_gpu
        if use_gpu:
            if not CUPY_AVAILABLE:
                msg = "CuPy is not installed. Install with: pip install cupy-cuda12x"
                raise ImportError(msg)
            self.xp = cp
            self.sp = cp_sparse
        else:
            self.xp = np
            from scipy import sparse as scipy_sparse  # type: ignore[import-not-found]

            self.sp = scipy_sparse

        # Execute sparse simulation
        state_vector = self.execute(circuit)

        job.set_result(JobResult(state_vector=state_vector, counts=[]))

        return job

    def execute(self, circuit: QuantumCircuit) -> NDArray[np.complex128]:
        """Execute the circuit using sparse matrix operations.

        Args:
            circuit: The quantum circuit to simulate

        Returns:
            NDArray: The final state vector (on CPU as NumPy array)
        """
        # Calculate total Hilbert space dimension
        size = reduce(operator.mul, circuit.dimensions, 1)

        # Initialize state vector |0⟩ as sparse
        if self.use_gpu:
            # GPU path (CuPy)
            state = self.sp.lil_matrix((size, 1), dtype=self.xp.complex128)
            state[0, 0] = 1.0 + 0.0j
            state = state.tocsr()
        else:
            # CPU path (SciPy)
            state = lil_matrix((size, 1), dtype=np.complex128)
            state[0, 0] = 1.0 + 0.0j
            state = state.tocsr()

        # Apply each gate using sparse matrices
        for gate in circuit.instructions:
            # Get sparse gate matrix with identities=2 for full system
            gate_matrix = gate.to_matrix(identities=2, sparse=True)

            # Transfer to GPU if needed
            if self.use_gpu:
                # Convert SciPy sparse to CuPy sparse
                gate_matrix = self.sp.csr_matrix(gate_matrix)
            elif not isinstance(gate_matrix, csr_matrix):
                # Ensure CSR format on CPU
                gate_matrix = csr_matrix(gate_matrix)

            # Apply gate: |ψ'⟩ = U |ψ⟩
            state = gate_matrix @ state

        # Convert result to dense array
        state_array = self.xp.asnumpy(state.toarray()).flatten() if self.use_gpu else state.toarray().flatten()

        # Reshape dimensions for proper qudit ordering
        reversed_dimensions = list(reversed(circuit.dimensions))
        state_array = state_array.reshape(reversed_dimensions)

        # Reverse the order of the axes for the transpose operation
        axes_order = list(reversed(list(range(len(circuit.dimensions)))))

        # Transpose the state array
        state_array = np.transpose(state_array, axes_order)

        return state_array.reshape((1, size))
