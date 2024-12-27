from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_phy_unitary_sim, mini_unitary_sim
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


class TestPhyTwoSimplePass(TestCase):
    @staticmethod
    def test_two_transpile_rctrl():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 3, 4], 0)
        circuit.r(0, [1, 2, np.pi / 5, np.pi / 7]).control([2], [1])
        circuit.r(0, [1, 2, np.pi / 5, np.pi / 7]).control([2], [1])
        circuit.r(1, [1, 2, np.pi / 5, np.pi / 7]).control([2], [1])
        circuit.r(2, [1, 2, np.pi / 5, np.pi / 7]).control([0], [0])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyEntSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_rzctrl():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 3, 4], 0)
        circuit.rz(0, [0, 1, np.pi / 3]).control([2], [1])
        circuit.rz(1, [1, 2, np.pi / 3]).control([2], [1])
        circuit.rz(2, [0, 3, np.pi / 3]).control([0], [0])
        circuit.rz(2, [0, 3, -np.pi / 3]).control([0], [1])
        circuit.rz(1, [0, 2, -np.pi / 2]).control([0], [2])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyEntSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit).round(10)
        uni_cl = mini_phy_unitary_sim(new_circuit).round(10)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_rctrl_close():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 3, 4], 0)
        circuit.r(0, [0, 1, np.pi / 3, np.pi / 7]).control([1], [1])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyEntSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_cex():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 3, 4], 0)
        circuit.cx([0, 2], [1, 2, 2, np.pi / 7])
        circuit.cx([0, 1], [0, 2, 2, np.pi / 5])
        circuit.cx([1, 2], [1, 3, 1, -np.pi / 7])
        circuit.cx([0, 2], [0, 2, 0, -np.pi / 7])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyEntSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

    @staticmethod
    def test_two_transpile_cex_close():
        # Create the original circuit
        circuit = QuantumCircuit(2, [3, 3], 0)
        circuit.cx([0, 1], [1, 2, 0, -np.pi / 4])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyEntSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)
        pi_p = new_circuit.instructions[0]
        cex = new_circuit.instructions[1]
        pi_pb = new_circuit.instructions[2]

        assert pi_p.lev_a == 0
        assert pi_p.lev_b == 2

        assert cex.lev_a == 0
        assert cex.lev_b == 1

        assert pi_pb.lev_a == 0
        assert pi_pb.lev_b == 2

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)
