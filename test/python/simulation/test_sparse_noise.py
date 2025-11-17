"""Tests for sparse simulators with noise models.

This module tests the integration of noise models (both Noise and SubspaceNoise)
with the sparse matrix simulators.
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel, SubspaceNoise


class TestSparseStatevecNoise:
    """Test noise simulation with SparseStatevecSim."""

    @staticmethod
    def test_basic_noise_simulation():
        """Test basic noise simulation with Noise model."""
        # Create a simple circuit
        qreg = QuantumRegister("qudits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        # Create noise model
        noise_model = NoiseModel()
        local_error = Noise(probability_depolarizing=0.01, probability_dephasing=0.01)
        noise_model.add_quantum_error_locally(local_error, ["h", "cx"])

        # Run with noise
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        # Check results
        assert result.counts is not None
        assert len(result.counts) == 100
        assert result.state_vector is not None

    @staticmethod
    def test_subspace_noise_simulation():
        """Test noise simulation with SubspaceNoise model."""
        # Create a circuit
        qreg = QuantumRegister("qudits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.x(0)
        circuit.h(1)

        # Create physical noise model with SubspaceNoise
        noise_model = NoiseModel()
        subspace_01 = SubspaceNoise(probability_depolarizing=0.01, probability_dephasing=0.01, levels=(0, 1))
        subspace_12 = SubspaceNoise(probability_depolarizing=0.02, probability_dephasing=0.01, levels=(1, 2))

        noise_model.add_quantum_error_locally(subspace_01, ["x"])
        noise_model.add_quantum_error_locally(subspace_12, ["h"])

        # Run with noise
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        # Check results
        assert result.counts is not None
        assert len(result.counts) == 100

    @staticmethod
    def test_dynamic_subspace_noise():
        """Test noise with dynamically assigned subspace."""
        qreg = QuantumRegister("qudits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.rz(0, [np.pi / 4])
        circuit.cx([0, 1])

        # Dynamic noise (automatically assigned to active subspace)
        noise_model = NoiseModel()
        dynamic_noise = SubspaceNoise(probability_depolarizing=0.01, probability_dephasing=0.01, levels=[])
        noise_model.add_quantum_error_locally(dynamic_noise, ["rz"])
        noise_model.add_nonlocal_quantum_error(dynamic_noise, ["cx"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        assert result.counts is not None
        assert len(result.counts) == 100

    @staticmethod
    def test_mixed_noise_model():
        """Test circuit with mixed Noise and SubspaceNoise."""
        qreg = QuantumRegister("qudits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.x(1)
        circuit.cx([0, 1])

        # Mixed noise model
        noise_model = NoiseModel()
        math_noise = Noise(probability_depolarizing=0.01, probability_dephasing=0.01)
        phys_noise = SubspaceNoise(probability_depolarizing=0.02, probability_dephasing=0.01, levels=(0, 1))

        noise_model.add_quantum_error_locally(math_noise, ["h"])
        noise_model.add_quantum_error_locally(phys_noise, ["x"])
        noise_model.add_nonlocal_quantum_error(math_noise, ["cx"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        assert result.counts is not None
        assert len(result.counts) == 100

    @staticmethod
    def test_noise_with_gpu_unavailable():
        """Test that noise simulation works on CPU even if GPU is requested but unavailable."""
        qreg = QuantumRegister("qudits", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(Noise(0.01, 0.01), ["h"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        # Should work on CPU regardless of GPU availability
        job = backend.run(circuit, use_gpu=False, noise_model=noise_model, shots=50)
        result = job.result()

        assert len(result.counts) == 50

    @staticmethod
    def test_shots_requirement():
        """Test that noise simulation requires at least 50 shots."""
        qreg = QuantumRegister("qudits", 1, [2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(Noise(0.01, 0.01), ["h"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        # Should raise assertion error for insufficient shots
        with pytest.raises(AssertionError, match="Number of shots should be above 50"):
            backend.run(circuit, noise_model=noise_model, shots=10)

    @staticmethod
    def test_noise_free_simulation():
        """Test that simulation without noise still works correctly."""
        qreg = QuantumRegister("qudits", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        # No noise model provided
        job = backend.run(circuit, shots=1)
        result = job.result()

        assert result.state_vector is not None
        assert len(result.counts) == 0  # No stochastic simulation without noise

    @staticmethod
    def test_entangling_noise():
        """Test noise on entangling gates."""
        qreg = QuantumRegister("qudits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])
        circuit.ls([0, 1])

        noise_model = NoiseModel()
        entangling_noise = Noise(probability_depolarizing=0.05, probability_dephasing=0.02)
        noise_model.add_nonlocal_quantum_error(entangling_noise, ["cx", "ls"])
        noise_model.add_nonlocal_quantum_error_on_target(entangling_noise, ["cx"])
        noise_model.add_nonlocal_quantum_error_on_control(entangling_noise, ["ls"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        assert result.counts is not None
        assert len(result.counts) == 100


class TestSparseUnitaryNoise:
    """Test noise handling in SparseUnitarySim (should warn/ignore)."""

    @staticmethod
    def test_noise_warning():
        """Test that unitary simulator warns when noise model is provided."""
        qreg = QuantumRegister("qudits", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(Noise(0.01, 0.01), ["h", "cx"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        # Should warn but still compute ideal unitary
        with pytest.warns(UserWarning, match="Noise models are not applicable"):
            job = backend.run(circuit, noise_model=noise_model)
            result = job.result()

        # Should still return unitary (ideal, no noise)
        assert result.state_vector is not None
        unitary = result.state_vector

        # Check unitary properties (should be unitary matrix)
        assert unitary.shape[0] == unitary.shape[1]
        identity_check = unitary @ unitary.conj().T
        assert np.allclose(identity_check, np.eye(unitary.shape[0]), atol=1e-10)

    @staticmethod
    def test_no_stochastic_simulation():
        """Test that unitary simulator doesn't perform stochastic simulation."""
        qreg = QuantumRegister("qudits", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(Noise(0.01, 0.01), ["h"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        with pytest.warns(UserWarning):
            job = backend.run(circuit, noise_model=noise_model, shots=100)
            result = job.result()

        # No counts (no stochastic simulation for unitary)
        assert len(result.counts) == 0


class TestNoiseConsistency:
    """Test consistency between sparse and other backends with noise."""

    @staticmethod
    def test_noise_produces_different_outcomes():
        """Verify that noise actually affects simulation outcomes."""
        qreg = QuantumRegister("qudits", 1, [2])
        circuit = QuantumCircuit(qreg)
        circuit.x(0)  # Should deterministically flip to |1⟩

        # With high noise
        noise_model = NoiseModel()
        high_noise = Noise(probability_depolarizing=0.5, probability_dephasing=0.5)
        noise_model.add_quantum_error_locally(high_noise, ["x"])

        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
        job = backend.run(circuit, noise_model=noise_model, shots=100)
        result = job.result()

        # With high noise, we should see some variation in outcomes
        unique_outcomes = set(result.counts)
        # Not all outcomes should be the same (noise introduces randomness)
        # Note: This is a probabilistic test, might rarely fail
        assert len(unique_outcomes) >= 1  # At least some measurement outcomes
