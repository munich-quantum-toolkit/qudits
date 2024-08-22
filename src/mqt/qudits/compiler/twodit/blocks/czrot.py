from __future__ import annotations

import numpy as np

from mqt.qudits.compiler.twodit.blocks.pswap import PSwapGen
from mqt.qudits.compiler.twodit.entanglement_qr import CRotGen


class CZRotGen:
    def __init__(self, circuit, indices) -> None:
        self.circuit = circuit
        self.indices = indices

    def z_from_crot_101_list(self, i, phase):
        crotgen = CRotGen(self.circuit, self.indices)
        pi_there = crotgen.permute_crot_101_as_list(i, np.pi / 2, 0.0)
        rotate = crotgen.permute_crot_101_as_list(i, phase, np.pi / 2)
        pi_back = crotgen.permute_crot_101_as_list(i, -np.pi / 2, 0.0)

        return pi_back + rotate + pi_there

    def z_pswap_101_as_list(self, i, phase):
        pswap_gen = PSwapGen(self.circuit, self.indices)
        pi_there = pswap_gen.permute_pswap_101_as_list(i, np.pi / 2, 0.0)
        rotate = pswap_gen.permute_pswap_101_as_list(i, phase, np.pi / 2)
        pi_back = pswap_gen.permute_pswap_101_as_list(i, -np.pi / 2, 0.0)
        return pi_back + rotate + pi_there
