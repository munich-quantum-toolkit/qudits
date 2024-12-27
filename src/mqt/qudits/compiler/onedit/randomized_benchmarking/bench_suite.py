from __future__ import annotations

import itertools
import pickle  # noqa: S403
import typing
from pathlib import Path

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


# Define H and S gates for a specific qudit dimension
def get_h_gate(dim: int) -> NDArray:
    circuit_d = QuantumCircuit(1, [dim], 0)
    h = circuit_d.h(0)
    return h.to_matrix()


def get_s_gate(dim: int) -> NDArray:
    circuit_d = QuantumCircuit(1, [dim], 0)
    s = circuit_d.s(0)
    return s.to_matrix()


def matrix_hash(matrix: NDArray) -> int:
    """Hash a numpy matrix using its byte representation."""
    return hash(matrix.tobytes())


def generate_clifford_group(d: int, max_length: int = 5) -> dict[str, NDArray]:
    # Initialize H and S gates
    h_gate = get_h_gate(d)
    s_gate = get_s_gate(d)

    gates = {"h": h_gate, "s": s_gate}
    clifford_group: dict[str, NDArray] = {}
    hash_table = set()  # To store matrix hashes for fast lookups

    # Iterate over different combination lengths
    for length in range(1, max_length + 1):
        for seq in itertools.product("hs", repeat=length):
            seq_str = "".join(seq)
            gate_product = np.eye(d)

            # Multiply gates in the sequence
            for gate in seq:
                gate_product = np.dot(gates[gate], gate_product)

            # Hash the matrix
            matrix_h = matrix_hash(gate_product)

            # Check if this matrix is already in the group using the hash table
            if matrix_h not in hash_table:
                clifford_group[seq_str] = gate_product
                hash_table.add(matrix_h)

    return clifford_group


def get_package_data_path(filename: str) -> Path:
    """Get the path to the data directory within the package."""
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir / filename


def save_clifford_group_to_file(clifford_group: dict[str, NDArray], filename: str) -> None:
    """Save the Clifford group to the 'data' directory in the current package."""
    filepath = get_package_data_path(filename)
    filepath.write_bytes(pickle.dumps(clifford_group))


def load_clifford_group_from_file(filename: str) -> dict[str, NDArray] | None:
    """Load the Clifford group from the 'data' directory in the current package."""
    filepath = get_package_data_path(filename)
    if filepath.exists():
        return typing.cast("dict[str, NDArray]", pickle.loads(filepath.read_bytes()))  # noqa: S301
    return None
