from __future__ import annotations

import typing

import numpy as np

from mqt.qudits.compiler.twodit.blocks.pswap import PSwapGen
from mqt.qudits.compiler.twodit.entanglement_qr import CRotGen

if typing.TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate


class CZRotGen:
    def __init__(self, circuit: QuantumCircuit, indices: list[int]) -> None:
        self.circuit: QuantumCircuit = circuit
        self.indices: list[int] = indices

    def z_from_crot_101_list(self, i: int, phase: float) -> list[Gate]:
        crotgen = CRotGen(self.circuit, self.indices)
        pi_there = crotgen.permute_crot_101_as_list(i, np.pi / 2, 0.0)
        rotate = crotgen.permute_crot_101_as_list(i, phase, np.pi / 2)
        pi_back = crotgen.permute_crot_101_as_list(i, -np.pi / 2, 0.0)

        return pi_back + rotate + pi_there

    def z_pswap_101_as_list(self, i: int, phase: float) -> list[Gate]:
        pswap_gen = PSwapGen(self.circuit, self.indices)
        pi_there = pswap_gen.permute_pswap_101_as_list(i, np.pi / 2, 0.0)
        rotate = pswap_gen.permute_pswap_101_as_list(i, phase, np.pi / 2)
        pi_back = pswap_gen.permute_pswap_101_as_list(i, -np.pi / 2, 0.0)
        return pi_back + rotate + pi_there
