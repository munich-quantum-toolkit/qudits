from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_phy_unitary_sim, mini_unitary_sim
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


class TestPhyMultiSimplePass(TestCase):
    @staticmethod
    def test_two_transpile_rctrl():
        # Create the original circuit
        circuit = QuantumCircuit(4, [3, 3, 4, 4], 0)
        circuit.r(0, [1, 2, np.pi / 3, np.pi / 7]).control([2, 1], [1, 2])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_rzctrl():
        # Create the original circuit
        circuit = QuantumCircuit(4, [3, 3, 4, 4], 0)
        circuit.rz(1, [1, 2, np.pi / 3]).control([2, 3, 0], [1, 2, 2])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_rctrl_close():
        # Create the original circuit
        circuit = QuantumCircuit(4, [3, 3, 4, 4], 0)
        circuit.r(0, [0, 1, np.pi / 3, np.pi / 7]).control([2, 3, 1], [1, 2, 2])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_rrz_close():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 4, 4], 0)

        circuit.r(1, [0, 3, np.pi / 3, np.pi / 4]).control([0], [0])
        circuit.r(2, [0, 3, np.pi / 5, np.pi / 5]).control([0, 1], [1, 1])
        circuit.rz(2, [0, 3, np.pi / 5]).control([0, 1], [1, 1])
        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntSimplePass", "PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl, rtol=1e-8, atol=1e-8)
