from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister


class TestQuantumCircuit(TestCase):
    def test_to_qasm(self):
        """Export circuit as QASM program"""
        qreg_field = QuantumRegister("field", 7, [7, 7, 7, 7, 7, 7, 7])
        qreg_matter = QuantumRegister("matter", 2, [2, 2])

        # Initialize the circuit
        circ = QuantumCircuit(qreg_field)
        circ.append(qreg_matter)

        # Apply operations
        circ.x(qreg_field[0])
        circ.h(qreg_matter[0])
        circ.cx([qreg_field[0], qreg_field[1]])
        circ.cx([qreg_field[1], qreg_field[2]])
        circ.r(qreg_matter[1], [0, 1, np.pi, np.pi / 2])
        circ.csum([qreg_field[2], qreg_matter[1]])
        circ.pm([qreg_matter[0], qreg_matter[1]], [1, 0, 2, 3])
        circ.rh(qreg_field[2], [0, 1])
        circ.ls([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.ms([qreg_field[2], qreg_matter[0]], [np.pi / 3])
        circ.rz(qreg_matter[1], [0, 1, np.pi / 5])
        circ.s(qreg_field[6])
        circ.virtrz(qreg_field[6], [1, np.pi / 5])
        circ.z(qreg_field[4])
        circ.randu([qreg_field[0], qreg_matter[0], qreg_field[1]])
        circ.cu_one(qreg_field[0], np.identity(7))
        circ.cu_two([qreg_field[0], qreg_matter[1]], np.identity(7 * 2))
        circ.cu_multi([qreg_field[0], qreg_matter[1], qreg_matter[0]], np.identity(7 * 2 * 2))

        file = circ.save_to_file(file_name="test")
        circ.to_qasm()
        circ_new = QuantumCircuit()
        circ_new.load_from_file(file)
