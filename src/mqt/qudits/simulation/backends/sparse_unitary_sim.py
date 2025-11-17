"""Sparse Unitary Matrix Simulator for quantum circuits.

This module provides a memory-efficient unitary matrix simulator using sparse matrix representations.
Supports both CPU (NumPy/SciPy) and GPU (CuPy) backends.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix, identity  # type: ignore[import-not-found]
from typing_extensions import Unpack

from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...quantum_circuit import QuantumCircuit
    from ...quantum_circuit.gate import Gate
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


class SparseUnitarySim(Backend):
    """Sparse unitary matrix simulator.

    Builds the full circuit unitary matrix using sparse matrix operations.
    Supports both CPU (NumPy/SciPy) and GPU (CuPy) backends.
    Best suited for analyzing circuit unitaries with sparse gate matrices.

    Examples:
        >>> from mqt.qudits.simulation import MQTQuditProvider
        >>> provider = MQTQuditProvider()
        >>> # CPU backend
        >>> backend = provider.get_backend("sparse_unitary")
        >>> job = backend.run(circuit)
        >>> # GPU backend (requires CuPy)
        >>> job = backend.run(circuit, use_gpu=True)
        >>> result = job.result()
        >>> unitary_matrix = result.get_unitary()
    """

    def __init__(
        self,
        provider: MQTQuditProvider,
        name: str | None = None,
        description: str | None = None,
        **fields: Unpack[Backend.DefaultOptions],
    ) -> None:
        """Initialize the sparse unitary matrix simulator.

        Args:
            provider: The provider instance
            name: Backend name (default: "sparse_unitary")
            description: Backend description
            **fields: Additional backend options
        """
        if name is None:
            name = "sparse_unitary"
        if description is None:
            description = "Sparse matrix unitary simulator (CPU/GPU) for memory-efficient circuit unitary construction"
        super().__init__(provider, name=name, description=description, **fields)
        self.use_gpu = False
        self.xp: Any = np  # Will be np or cp
        self.sp: Any = None  # Will be scipy.sparse or cupyx.scipy.sparse

    def run(self, circuit: QuantumCircuit, use_gpu: bool = False, **options: Unpack[Backend.DefaultOptions]) -> Job:
        """Run the circuit unitary construction.

        Args:
            circuit: The quantum circuit to simulate
            use_gpu: Use GPU acceleration with CuPy (default: False)
            **options: Additional simulation options

        Returns:
            Job: A job object containing the unitary matrix

        Raises:
            ImportError: If use_gpu=True but CuPy is not installed
        """
        job = Job(self)

        self._options.update(options)

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

        # Build the full circuit unitary
        unitary = self.execute(circuit)

        # Store as "state_vector" for compatibility (though it's actually a unitary)
        job.set_result(JobResult(state_vector=unitary, counts=[]))

        return job

    def execute(self, circuit: QuantumCircuit, gate_list: list[Gate] | None = None) -> NDArray[np.complex128]:
        """Build the circuit unitary using sparse matrix operations.

        Args:
            circuit: The quantum circuit
            gate_list: Optional specific list of gates (default: all circuit instructions)

        Returns:
            NDArray: The circuit unitary matrix (on CPU as NumPy array)
        """
        # Calculate total Hilbert space dimension
        size = reduce(operator.mul, circuit.dimensions, 1)

        # Initialize with sparse identity matrix
        if self.use_gpu:
            # GPU path (CuPy)
            unitary = self.sp.identity(size, dtype=self.xp.complex128, format="csr")
        else:
            # CPU path (SciPy)
            unitary = identity(size, dtype=np.complex128, format="csr")

        # Use provided gate list or circuit instructions
        gates = gate_list if gate_list is not None else circuit.instructions

        # Apply each gate: U_total = U_n @ ... @ U_2 @ U_1
        for gate in gates:
            # Get sparse gate matrix with identities=2 for full system
            gate_matrix = gate.to_matrix(identities=2, sparse=True)

            # Transfer to GPU if needed
            if self.use_gpu:
                # Convert SciPy sparse to CuPy sparse
                gate_matrix = self.sp.csr_matrix(gate_matrix)
            elif not isinstance(gate_matrix, csr_matrix):
                # Ensure CSR format on CPU
                gate_matrix = csr_matrix(gate_matrix)

            # Multiply: U_total = U_i @ U_total
            unitary = gate_matrix @ unitary

        # Convert to dense array and transfer to CPU if needed
        if self.use_gpu:
            return self.xp.asnumpy(unitary.toarray())
        return unitary.toarray()

    def get_unitary_sparse(self, circuit: QuantumCircuit, gate_list: list[Gate] | None = None) -> csr_matrix | Any:  # noqa: ANN401
        """Build the circuit unitary and return as sparse matrix.

        Args:
            circuit: The quantum circuit
            gate_list: Optional specific list of gates

        Returns:
            csr_matrix: The circuit unitary as sparse matrix (for large circuits)
                       Returns scipy.sparse.csr_matrix on CPU or cupyx.scipy.sparse.csr_matrix on GPU
        """
        size = reduce(operator.mul, circuit.dimensions, 1)

        # Initialize identity based on current backend
        if self.use_gpu:
            unitary = self.sp.identity(size, dtype=self.xp.complex128, format="csr")
        else:
            unitary = identity(size, dtype=np.complex128, format="csr")

        gates = gate_list if gate_list is not None else circuit.instructions

        for gate in gates:
            gate_matrix = gate.to_matrix(identities=2, sparse=True)

            # Transfer to GPU if needed
            if self.use_gpu:
                gate_matrix = self.sp.csr_matrix(gate_matrix)
            elif not isinstance(gate_matrix, csr_matrix):
                gate_matrix = csr_matrix(gate_matrix)

            unitary = gate_matrix @ unitary

        return unitary
