from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestSparseUnitarySim(TestCase):
    @staticmethod
    def test_unitary_construction():
        """Test that unitary matrices are constructed correctly."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        # Single gate H
        for d in range(2, 6):
            qreg = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg)
            h = circuit.h(0)

            job = backend.run(circuit)
            unitary = job.result().get_state_vector()

            # Compare with expected unitary
            expected = h.to_matrix(identities=0)
            assert np.allclose(unitary, expected)

        # Two gates
        for d in range(2, 5):
            qreg = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg)
            h = circuit.h(0)
            x = circuit.x(0)

            job = backend.run(circuit)
            unitary = job.result().get_state_vector()

            # U = X @ H
            expected = x.to_matrix(identities=0) @ h.to_matrix(identities=0)
            assert np.allclose(unitary, expected)

        # Entangling circuit
        for d1 in range(2, 5):
            for d2 in range(2, 5):
                qreg = QuantumRegister("reg", 2, [d1, d2])
                circuit = QuantumCircuit(qreg)
                h = circuit.h(0)
                csum = circuit.csum([0, 1])

                job = backend.run(circuit)
                unitary = job.result().get_state_vector()

                # U = CSUM @ (H ⊗ I)
                expected = csum.to_matrix(identities=0) @ (np.kron(h.to_matrix(), np.identity(d2)))
                assert np.allclose(unitary, expected)

    @staticmethod
    def test_unitary_properties():
        """Test mathematical properties of generated unitaries."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        # Test unitarity: U†U = I
        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Check U†U = I
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(unitary.shape[0]))

        # Check UU† = I
        identity_test2 = unitary.conj().T @ unitary
        assert np.allclose(identity_test2, np.eye(unitary.shape[0]))

        # Check |det(U)| = 1
        det = np.linalg.det(unitary)
        assert np.isclose(np.abs(det), 1.0)

    @staticmethod
    def test_multi_gate_circuit():
        """Test circuits with multiple gates."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")
        rng = np.random.default_rng()

        qreg = QuantumRegister("test", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        h = circuit.h(0)
        angle = rng.uniform(0, 2 * np.pi)
        rz = circuit.rz(1, [0, 1, angle])
        csum = circuit.csum([0, 1])

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Build expected unitary: U = CSUM @ RZ @ (H ⊗ I)
        h_ext = np.kron(h.to_matrix(), np.identity(3))
        rz_ext = np.kron(np.identity(3), rz.to_matrix())
        expected = csum.to_matrix(identities=0) @ rz_ext @ h_ext

        assert np.allclose(unitary, expected)

        # Verify unitarity
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(unitary.shape[0]))

    @staticmethod
    def test_controlled_gates():
        """Test circuits with controlled gates."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        qreg = QuantumRegister("test", 3, [2, 2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.x(1).control([0], [1])

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Verify unitarity
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(unitary.shape[0]))

        # Check determinant
        det = np.linalg.det(unitary)
        assert np.isclose(np.abs(det), 1.0)

    @staticmethod
    def test_qudit_dimensions():
        """Test with different qudit dimensions."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        # Qutrit circuit
        qreg = QuantumRegister("qutrits", 2, [3, 3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.csum([0, 1])

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Should be 9x9 for two qutrits
        assert unitary.shape == (9, 9)

        # Verify unitarity
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(9))

        # Mixed dimensions
        qreg = QuantumRegister("mixed", 3, [2, 3, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.h(1)

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Should be 12x12 for 2*3*2
        assert unitary.shape == (12, 12)

        # Verify unitarity
        identity_test = unitary @ unitary.conj().T
        assert np.allclose(identity_test, np.eye(12))

    @staticmethod
    def test_empty_circuit():
        """Test empty circuit returns identity."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_unitary")

        qreg = QuantumRegister("test", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        # No gates added

        job = backend.run(circuit)
        unitary = job.result().get_state_vector()

        # Should be identity matrix
        expected = np.eye(4)
        assert np.allclose(unitary, expected)
