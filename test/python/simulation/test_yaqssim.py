from unittest import TestCase
import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel


def _generalized_x(d: int) -> np.ndarray:
    X = np.zeros((d, d), dtype=complex)
    for j in range(d):
        X[(j + 1) % d, j] = 1.0
    return X


def _generalized_z(d: int) -> np.ndarray:
    omega = np.exp(2j * np.pi / d)
    return np.diag([omega**j for j in range(d)])


class TestYAQSSim(TestCase):
    @staticmethod
    def test_single_qudit_h():
        provider = MQTQuditProvider()
        backend = provider.get_backend("yaqssim")

        for d in range(2, 5):
            qreg = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg)
            h = circuit.h(0)

            zero = np.zeros(d)
            zero[0] = 1.0
            expected = h.to_matrix() @ zero

            job = backend.run(circuit)
            sv = job.result().get_state_vector()

            assert np.allclose(sv, expected)
        
    @staticmethod
    def test_two_qudit_csum():
        provider = MQTQuditProvider()
        backend = provider.get_backend("yaqssim")

        for d in range(2,5):
            qreg = QuantumRegister("reg", 2, [d, d])
            circuit = QuantumCircuit(qreg)
            h = circuit.h(0)
            csum = circuit.csum([0, 1])

            zero = np.zeros(d * d)
            zero[0] = 1.0
            expected = csum.to_matrix() @ np.kron(h.to_matrix(), np.eye(d)) @ zero

            job = backend.run(circuit)
            sv = job.result().get_state_vector()

            assert np.allclose(sv, expected)

    @staticmethod
    def test_long_range_gate():
        provider = MQTQuditProvider()
        backend = provider.get_backend("yaqssim")

        d = 2
        qreg = QuantumRegister("reg", 3, [d, d, d])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.csum([0, 2])
        
        job = backend.run(circuit)
        sv = job.result().get_state_vector()
        expected = np.zeros(d**3, dtype=complex)
        expected[0] = 1 / np.sqrt(2)
        expected[5] = 1 / np.sqrt(2)
        assert np.allclose(sv, expected)

    @staticmethod
    def test_generalized_operators():
        assert np.allclose(_generalized_x(2), np.array([[0, 1], [1, 0]]))
        assert np.allclose(_generalized_z(2), np.array([[1, 0], [0, -1]]))

        for d in range(2, 6):
            assert np.allclose(np.linalg.matrix_power(_generalized_x(d), d), np.eye(d))
            assert np.allclose(np.linalg.matrix_power(_generalized_z(d), d), np.eye(d))

    @staticmethod
    def test_noise():
        provider = MQTQuditProvider()
        backend = provider.get_backend("yaqssim")

        qreg = QuantumRegister("reg", 1, [3])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)

        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(Noise(0.1, 0.05), ["h"])

        job = backend.run(circuit, noise_model=noise_model)
        sv = job.result().get_state_vector()

        assert np.allclose(np.linalg.norm(sv), 1.0)