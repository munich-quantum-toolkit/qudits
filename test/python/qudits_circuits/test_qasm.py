from __future__ import annotations

from unittest import TestCase

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestQASM(TestCase):
    qasm = """
            DITQASM 2.0;

            qreg field [7][5,5,5,5,5,5,5];
            qreg matter [2];

            creg meas_matter[7];
            creg meas_fields[3];

            h matter[0] ctl field[0] field[1] [0,0];
            cx field[2], matter[0];
            cx field[2], matter[1];
            rxy (0, 1, pi, pi/2) field[3];

            measure q[0] -> meas[0];
            measure q[1] -> meas[1];
            measure q[2] -> meas[2];
            """

    circuit = QuantumCircuit()
    circuit.from_qasm(qasm)
