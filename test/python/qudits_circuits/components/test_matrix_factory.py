from __future__ import annotations

from unittest import TestCase

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister


class TestMatrixFactory(TestCase):
    def test_generate_matrix(self):
        # no control
        qreg_example = QuantumRegister("reg", 1, [3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(qreg_example[0])

        """# with 1 control, 2 3
        qreg_example = QuantumRegister("reg", 2, [2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0).control([1], [0])

        # inverted
        qreg_example = QuantumRegister("reg", 2, [2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0).control([1], [0])

        # with 1 control, 3 2
        qreg_example = QuantumRegister("reg", 2, [3, 2])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0).control([1], [0])

        # inverted
        qreg_example = QuantumRegister("reg", 2, [3, 2])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0).control([1], [0])

        # no control, wrapped in identities
        qreg_example = QuantumRegister("reg", 1, [2, 3, 4])
        circuit = QuantumCircuit(qreg_example)
        h = circuit.h(qreg_example[0])

        hmat = np.kron(np.kron(np.kron(h.to_matrix(), np.identity(2)), np.identity(d2)),
                       np.identity(2))  # h.to_matrix(identities=2)
        hh = h.to_matrix(2)
        mats = np.allclose(h.to_matrix(2), hmat)

        # with 2 controls, 2 3
        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(2).control([0, 1], [1, 1])
"""
