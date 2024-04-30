from __future__ import annotations

from unittest import TestCase

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestQASM(TestCase):
    def test_from_qasm(self):
        """Create circuit from QASM program"""
        qasm = """
                DITQASM 2.0;
                qreg field [7][5,5,5,5,5,5,5];
                qreg matter [2];
                creg meas [2];
                creg fieldc [7];
                x field[0];
                h matter[0] ctl field[0] field[1] [0,0];
                cx (0, 1, 1, pi/2) field[2], matter[0];
                cx (1, 2, 0, pi ) field[2], matter[1];
                rxy (0, 1, pi, pi/2) field[3];
                csum field[2], matter[1];
                pm ([1, 0, 2, 3]) matter[0], matter[1];
                rh (0, 1) field[3];
                ls (pi/3) field[2], matter[0];
                ms (pi/3) field[5], matter[1];
                rz (0, 1, pi) field[3];
                s field[6];
                virtrz (1, pi/5) field[6];
                z field[4];
                rdu matter[0], field[1], field[2];
                measure q[0] -> meas[0];
                measure q[1] -> meas[1];
                measure q[2] -> meas[2];
                """

        circuit = QuantumCircuit()
        circuit.from_qasm(qasm)
        assert circuit._num_cl == 9
        assert circuit._num_qudits == 9
        assert circuit._dimensions == [5, 5, 5, 5, 5, 5, 5, 2, 2]
        assert circuit.number_gates == 15
        assert len(circuit.quantum_registers) == 2
        assert len(circuit.classic_registers) == 2
        assert [s.qasm_tag for s in circuit.instructions] == [
            "x",
            "h",
            "cx",
            "cx",
            "rxy",
            "csum",
            "pm",
            "rh",
            "ls",
            "ms",
            "rz",
            "s",
            "virtrz",
            "z",
            "rdu",
        ]
