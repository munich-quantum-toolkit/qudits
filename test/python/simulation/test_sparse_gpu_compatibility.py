"""Test GPU compatibility and fallback behavior for sparse simulators."""

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pytest

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestSparseGPUCompatibility(TestCase):
    """Test CPU/GPU compatibility for sparse simulators."""

    @staticmethod
    def test_cpu_mode_statevec():
        """Test CPU mode works correctly for sparse statevec."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        # CPU mode (default)
        job = backend.run(circuit, use_gpu=False)
        result = job.result()
        state = result.get_state_vector().flatten()

        # Verify Bell state
        assert np.isclose(np.abs(state[0]) ** 2, 0.5)
        assert np.isclose(np.abs(state[3]) ** 2, 0.5)
        assert np.isclose(np.linalg.norm(state), 1.0)

    @staticmethod
    def test_cpu_mode_unitary():
        """Test CPU mode works correctly for sparse unitary."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        # CPU mode (default)
        job = backend.run(circuit, use_gpu=False)
        result = job.result()
        unitary = result.get_state_vector()

        # Verify unitarity
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(4))

    @staticmethod
    def test_gpu_mode_without_cupy_statevec():
        """Test GPU mode raises proper error when CuPy is not installed."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        # Try to use GPU mode (should fail gracefully if CuPy not installed)
        try:
            import cupy  # type: ignore[import-not-found]  # noqa: F401

            # CuPy is available, test should pass
            job = backend.run(circuit, use_gpu=True)
            result = job.result()
            state = result.get_state_vector()
            assert state is not None
        except ImportError:
            # CuPy not available, should get ImportError from backend
            with pytest.raises(ImportError, match="CuPy is not installed"):
                backend.run(circuit, use_gpu=True)

    @staticmethod
    def test_gpu_mode_without_cupy_unitary():
        """Test GPU mode raises proper error when CuPy is not installed."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        # Try to use GPU mode (should fail gracefully if CuPy not installed)
        try:
            import cupy  # type: ignore[import-not-found]  # noqa: F401

            # CuPy is available, test should pass
            job = backend.run(circuit, use_gpu=True)
            result = job.result()
            unitary = result.get_state_vector()
            assert unitary is not None
        except ImportError:
            # CuPy not available, should get ImportError from backend
            with pytest.raises(ImportError, match="CuPy is not installed"):
                backend.run(circuit, use_gpu=True)

    @staticmethod
    def test_default_is_cpu():
        """Test that default behavior uses CPU."""
        provider = MQTQuditProvider()
        backend_statevec = provider.get_backend("sparse_statevec")
        backend_unitary = provider.get_backend("sparse_unitary")

        qreg = QuantumRegister("test", 1, [2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        # Default run should use CPU (no error even without CuPy)
        job1 = backend_statevec.run(circuit)
        result1 = job1.result()
        assert result1.get_state_vector() is not None

        job2 = backend_unitary.run(circuit)
        result2 = job2.result()
        assert result2.get_state_vector() is not None

    @staticmethod
    def test_sparse_vs_dense_equivalence():
        """Test that sparse matrix results match dense matrix results."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        qreg = QuantumRegister("test", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.rz(1, [0, 1, 0.5])
        circuit.csum([0, 1])

        # Get sparse result
        job_sparse = backend.run(circuit, use_gpu=False)
        state_sparse = job_sparse.result().get_state_vector().flatten()

        # Build expected result using dense matrices
        zero_state = np.zeros(9)
        zero_state[0] = 1.0

        h = circuit.instructions[0].to_matrix(identities=2, sparse=False)
        rz = circuit.instructions[1].to_matrix(identities=2, sparse=False)
        csum = circuit.instructions[2].to_matrix(identities=2, sparse=False)

        state_dense = csum @ rz @ h @ zero_state

        # Should match (within numerical precision)
        assert np.allclose(state_sparse, state_dense)


class TestSparseMatrixGeneration(TestCase):
    """Test that all gates can generate sparse matrices."""

    @staticmethod
    def test_all_single_qudit_gates_sparse():
        """Test all single-qudit gates produce sparse matrices."""
        from scipy.sparse import issparse  # type: ignore[import-not-found]

        qreg = QuantumRegister("test", 1, [3])
        circuit = QuantumCircuit(qreg)

        # Test each gate type
        gates = [
            circuit.h(0),
            circuit.x(0),
            circuit.z(0),
            circuit.s(0),
            circuit.rz(0, [0, 1, 0.5]),
            circuit.rh(0, [0, 1]),
            circuit.r(0, [0, 1, 0.5, 0.3]),
            circuit.virtrz(0, [0, 0.5]),
        ]

        for gate in gates:
            # Request sparse matrix
            matrix_sparse = gate.to_matrix(identities=0, sparse=True)
            assert issparse(matrix_sparse), f"{gate.gate_type} did not return sparse matrix"

            # Request dense matrix
            matrix_dense = gate.to_matrix(identities=0, sparse=False)
            assert not issparse(matrix_dense), f"{gate.gate_type} returned sparse when dense requested"

            # Verify equivalence
            assert np.allclose(matrix_sparse.toarray(), matrix_dense), f"{gate.gate_type} sparse != dense"

    @staticmethod
    def test_all_two_qudit_gates_sparse():
        """Test all two-qudit gates produce sparse matrices."""
        from scipy.sparse import issparse  # type: ignore[import-not-found]

        qreg = QuantumRegister("test", 2, [2, 3])
        circuit = QuantumCircuit(qreg)

        # Test each gate type
        gates = [
            circuit.csum([0, 1]),
            circuit.cx([0, 1], [0, 1, 0, 0.5]),
        ]

        for gate in gates:
            # Request sparse matrix
            matrix_sparse = gate.to_matrix(identities=0, sparse=True)
            assert issparse(matrix_sparse), f"{gate.gate_type} did not return sparse matrix"

            # Request dense matrix
            matrix_dense = gate.to_matrix(identities=0, sparse=False)
            assert not issparse(matrix_dense), f"{gate.gate_type} returned sparse when dense requested"

            # Verify equivalence
            assert np.allclose(matrix_sparse.toarray(), matrix_dense), f"{gate.gate_type} sparse != dense"
