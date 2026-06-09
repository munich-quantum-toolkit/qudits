from unittest import TestCase
import numpy as np
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider

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