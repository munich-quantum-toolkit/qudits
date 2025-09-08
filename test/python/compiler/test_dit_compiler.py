from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.visualisation.plot_information import remap_result


class TestQuditCompiler(TestCase):
    @staticmethod
    def test_compile_00():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2six")
        qudit_compiler = QuditCompiler()
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        circuit_33.h(0).to_matrix(2).round(2)
        circuit_33.r(0, [0, 1, np.pi / 7, np.pi / 3]).to_matrix(0).round(2)
        circuit_33.r(0, [1, 2, np.pi / 5, -np.pi / 3]).to_matrix(0).round(2)
        circuit_33.x(0).dag().to_matrix(2).round(2)

        og_state = circuit_33.simulate().round(5)
        ogp = remap_result(og_state, circuit_33)

        circuit = qudit_compiler.compile_O0(backend_ion, circuit_33)
        state = circuit.simulate().round(5)
        new_s = remap_result(state, circuit)

        assert np.allclose(new_s, ogp)
