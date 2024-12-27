from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import R

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit.components.extensions.controls import ControlData


class TestR(TestCase):
    @staticmethod
    def test___array__():
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [1, 2, np.pi / 3, np.pi / 7])
        # R(np.pi / 3, np.pi / 7, 1, 2, 3)
        r1_test = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.86602 + 0.0j, -0.21694 - 0.45048j],
            [0.0 + 0.0j, 0.21694 - 0.45048j, 0.86602 + 0.0j],
        ])

        assert np.allclose(r.to_matrix(identities=0), r1_test)

        r1_test_dag = np.array([
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.86602 + 0.0j, 0.21694 + 0.45048j],
            [0.0 + 0.0j, -0.21694 + 0.45048j, 0.86602 + 0.0j],
        ])

        assert np.allclose(r.dag().to_matrix(identities=0), r1_test_dag)

        circuit_4 = QuantumCircuit(1, [4], 0)
        r_2 = circuit_4.r(0, [0, 2, np.pi / 3, np.pi / 7])

        # R(np.pi / 3, np.pi / 7, 0, 2, 4)
        r_2_test = np.array([
            [0.8660254 + 0.0j, 0.0 + 0.0j, -0.21694187 - 0.45048443j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.21694187 - 0.45048443j, 0.0 + 0.0j, 0.8660254 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(r_2.to_matrix(identities=0), r_2_test)

        r_2_test_dag = np.array([
            [0.8660254 + 0.0j, 0.0 + 0.0j, 0.21694187 + 0.45048443j, 0.0 + 0.0j],
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [-0.21694187 + 0.45048443j, 0.0 + 0.0j, 0.8660254 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
        ])

        assert np.allclose(r_2.dag().to_matrix(identities=0), r_2_test_dag)

    @staticmethod
    def test_regulate_theta():
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [1, 2, 0.01 * np.pi, np.pi / 7])
        # R(0.01 * np.pi, np.pi / 7, 0, 2, 3)
        assert round(r.theta, 4) == 12.5978

    @staticmethod
    def test_levels_setter():
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, 0.01 * np.pi, np.pi / 7])
        # R(0.01 * np.pi, np.pi / 7, 2, 0, 3)

        assert r.lev_a == 0
        assert r.lev_b == 2
        assert r.original_lev_a == 2
        assert r.original_lev_b == 0

    @staticmethod
    def test_cost():
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, 0.01 * np.pi, np.pi / 7])
        assert round(r.cost, 4) == 0.0016

    @staticmethod
    def test_validate_parameter():
        circuit_3 = QuantumCircuit(1, [3], 0)
        r = circuit_3.r(0, [2, 0, np.pi, np.pi / 7])

        assert r.validate_parameter([2, 0, np.pi, np.pi / 7])
        try:
            r.validate_parameter([3, 0, np.pi, np.pi / 7])
        except AssertionError:
            assert True
        try:
            r.validate_parameter([1, 3, np.pi, np.pi / 7])
        except AssertionError:
            assert True

    @staticmethod
    def test_control():
        circuit_3 = QuantumCircuit(2, [2, 3], 0)
        r = circuit_3.r(0, [1, 0, np.pi, np.pi / 7]).control([1], [1])
        ci = cast("ControlData", r.control_info["controls"])
        assert ci.indices == [1]
        assert ci.ctrl_states == [1]
        assert r.gate_type == GateTypes.TWO
        assert isinstance(r, R)

        circuit_3_2 = QuantumCircuit(3, [2, 3, 3], 0)
        r = circuit_3_2.r(0, [1, 0, np.pi, np.pi / 7]).control([1, 2], [1, 1])
        ci = cast("ControlData", r.control_info["controls"])
        assert r.gate_type == GateTypes.MULTI
        assert isinstance(r, R)
        assert ci.indices == [1, 2]
        assert ci.ctrl_states == [1, 1]
