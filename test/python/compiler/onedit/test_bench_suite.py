from __future__ import annotations

import tempfile
import typing
import unittest
from pathlib import Path

import numpy as np

from mqt.qudits.compiler.onedit.randomized_benchmarking.bench_suite import (
    generate_clifford_group,
    get_h_gate,
    get_package_data_path,
    get_s_gate,
    load_clifford_group_from_file,
    matrix_hash,
    save_clifford_group_to_file,
)
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

rng = np.random.default_rng()


def randint(x: int, y: int | None = None) -> int:
    return rng.integers(0, x) if y is None else rng.integers(x, y)


class TestCliffordGroupGeneration(unittest.TestCase):
    @staticmethod
    def test_get_h_gate():
        h_gate = get_h_gate(2)
        expected_h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(h_gate, expected_h)

    @staticmethod
    def test_get_s_gate():
        s_gate = get_s_gate(2)
        expected_s = np.array([[1, 0], [0, 1j]])
        np.testing.assert_array_almost_equal(s_gate, expected_s)

    @staticmethod
    def test_matrix_hash():
        matrix1 = np.array([[1, 0], [0, 1]])
        matrix2 = np.array([[1, 0], [0, 1]])
        matrix3 = np.array([[0, 1], [1, 0]])

        assert matrix_hash(matrix1) == matrix_hash(matrix2)
        assert matrix_hash(matrix1) != matrix_hash(matrix3)

    @staticmethod
    def test_generate_clifford_group():
        clifford_group = generate_clifford_group(2, max_length=2)
        assert len(clifford_group) == 6

    @staticmethod
    def test_get_package_data_path():
        filename = "test_file.pkl"
        path = get_package_data_path(filename)
        assert path.parent.name == "data"
        assert path.name == filename

    @staticmethod
    def test_save_and_load_clifford_group():
        clifford_group: dict[str, NDArray] = generate_clifford_group(2, max_length=2)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = Path(temp_file.name).name

        try:
            save_clifford_group_to_file(clifford_group, filename)
            loaded_group = load_clifford_group_from_file(filename)
            assert loaded_group is not None
            assert len(clifford_group) == len(loaded_group)
            for key in clifford_group:  # noqa: PLC0206
                np.testing.assert_array_almost_equal(clifford_group[key], loaded_group[key])
        finally:
            get_package_data_path(filename).unlink()

    @staticmethod
    def test_benching():
        dim_g = 3

        clifford_group = generate_clifford_group(dim_g, max_length=10)
        save_clifford_group_to_file(clifford_group, f"cliffords_{dim_g}.dat")
        clifford_group = typing.cast("dict[str, NDArray]", load_clifford_group_from_file(f"cliffords_{dim_g}.dat"))

        def create_rb_sequence(length: int = 2) -> QuantumCircuit:
            circuit = QuantumCircuit()
            dit_register = QuantumRegister("dits", 1, [3])
            circuit.append(dit_register)
            inversion = np.eye(dim_g)
            check = np.eye(dim_g)

            for _ in range(length):
                random_gate = list(clifford_group.values())[randint(len(clifford_group))]
                inversion = np.matmul(inversion, random_gate.conj().T)
                circuit.cu_one(0, random_gate)
                check = np.matmul(random_gate, check)
            circuit.cu_one(0, inversion)
            return circuit

        circuit = create_rb_sequence(length=16)
        circuit.compileO1("faketraps2trits", "adapt")
