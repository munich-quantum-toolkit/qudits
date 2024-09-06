#!/usr/bin/env python3
from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.compiler.twodit.blocks.crot import CEX_SEQUENCE
from mqt.qudits.quantum_circuit import gates

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate


class PSwapGen:
    def __init__(self, circuit: QuantumCircuit, indices: list[int]) -> None:
        self.circuit: QuantumCircuit = circuit
        self.indices: list[int] = indices

    def pswap_101_as_list_phases(self, theta: float, phi: float) -> list[Gate]:
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        if dim_target == 2:
            theta = -theta

        h_0 = gates.Rh(self.circuit, "Rh", index_ctrl, [0, 1], dim_ctrl)
        h_1 = gates.Rh(self.circuit, "Rh", index_target, [0, 1], dim_target)
        # HditR(0, 1, d).matrix

        zpiov2_0 = gates.Rz(self.circuit, "Rz-zpiov2", index_ctrl, [0, 1, np.pi / 2], dim_ctrl)
        # ZditR(np.pi / 2, 0, 1, d).matrix

        zp_0 = gates.Rz(self.circuit, "Rz-zp", index_ctrl, [0, 1, np.pi], dim_ctrl)
        # ZditR(np.pi, 0, 1, d).matrix

        rphi_there_1 = gates.R(self.circuit, "R_there", index_target, [0, 1, np.pi / 2, -phi - np.pi / 2], dim_target)
        # R(np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

        rphi_back_1 = gates.R(self.circuit, "R_back", index_target, [0, 1, -np.pi / 2, -phi - np.pi / 2], dim_target)
        # R(-np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

        # Possible solution to improve of decomposition
        single_excitation = gates.VirtRz(self.circuit, "vR", index_target, [0, -np.pi], dim_target)
        single_excitation_back = gates.VirtRz(self.circuit, "vR", index_target, [0, np.pi], dim_target)

        tminus = gates.Rz(self.circuit, "Rz", index_target, [0, 1, -theta / 2], dim_target)
        # on1(ZditR(-theta / 2, 0, 1, d).matrix, d))

        tplus = gates.Rz(self.circuit, "Rz", index_target, [0, 1, +theta / 2], dim_target)
        # on1(ZditR(theta / 2, 0, 1, d).matrix, d)

        if CEX_SEQUENCE is None:
            cex = gates.CEx(
                self.circuit,
                "CEx" + str([self.circuit.dimensions[i] for i in self.indices]),
                self.indices,
                None,
                [self.circuit.dimensions[i] for i in self.indices],
                None,
            )
            # Cex().cex_101(d, 0)
        else:
            cex_s = CEX_SEQUENCE

        #############################################################################################
        # START THE DECOMPOSITION
        """
        ph1 = -1 * np.identity(self.circuit.dimensions[index_ctrl], dtype="complex")
        ph1[0][0] = 1
        ph1[1][1] = 1
        ph1 = gates.CustomOne(
                self.circuit,
                "PH1" + str(self.circuit.dimensions[index_ctrl]),
                index_ctrl,
                ph1,
                self.circuit.dimensions[index_ctrl],
                None,
        )
        """
        compose: list[Gate] = []
        # Used to be [on0(ph1, d), on0(h_, d)]
        #################################

        if dim_target != 2:
            r_flip_1 = gates.R(self.circuit, "R_flip", index_target, [1, dim_target - 1, np.pi, np.pi / 2], dim_target)
            compose.append(r_flip_1)  # (on1(R(np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        """compose.append(h_0)
        compose.append(h_1)  # (on1(h_, d))

        compose.append(zpiov2_0)  # (on0(zpiov2, d))"""
        compose.extend((h_0, h_1, zpiov2_0))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex_s

        compose.append(h_0)  # (on0(h_, d))
        compose.append(h_1)  # (on1(h_, d))

        compose.append(zp_0)  # (on0(zp, d))

        ###############################################################
        # START OF CONTROLLED ROTATION
        compose.append(rphi_there_1)  # (on1(rphi_there, d))  # ----------

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex_s
        compose.append(single_excitation)
        compose.append(tminus)

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex_s

        compose.append(tplus)
        compose.append(single_excitation_back)

        compose.append(rphi_back_1)  # (on1(rphi_back, d))

        ##################################

        compose.append(h_0)
        compose.append(h_1)  # (on1(h_, d))

        compose.append(zpiov2_0)  # (on0(zpiov2, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex_s

        compose.append(h_0)  # (on0(h_, d))
        compose.append(h_1)  # (on1(h_, d))

        if dim_target != 2:
            r_flip_back_1 = gates.R(
                self.circuit, "R_flip_back", index_target, [1, dim_target - 1, -np.pi, np.pi / 2], dim_target
            )
            compose.append(r_flip_back_1)  # (on1(R(-np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        return compose

    def pswap_101_as_list_no_phases(self, theta: float, phi: float) -> list[Gate]:
        rotation = self.pswap_101_as_list_phases(-theta / 4, phi)
        return rotation + rotation + rotation + rotation

    def permute_pswap_101_as_list(self, pos: int, theta: float, phase: float, with_phase: bool = False) -> list[Gate]:
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        control_block = floor(pos / dim_target)
        if with_phase:
            rotation = self.pswap_101_as_list_phases(theta, phase)
        else:
            rotation = self.pswap_101_as_list_no_phases(theta, phase)

        if control_block != 0:
            permute_there_00 = gates.R(
                self.circuit, "R_there_00", index_ctrl, [0, control_block, np.pi, -np.pi / 2], dim_ctrl
            )
            # on0(R(np.pi, -np.pi / 2, 0, j, d).matrix, d)
            permute_there_01 = gates.R(
                self.circuit, "R_there_01", index_ctrl, [1, control_block + 1, -np.pi, np.pi / 2], dim_ctrl
            )
            # on0(R(-np.pi, np.pi / 2, 1, j + 1, d).matrix, d))

            permute_there_00_dag = gates.R(
                self.circuit, "R_there_00", index_ctrl, [0, control_block, np.pi, -np.pi / 2], dim_ctrl
            ).dag()
            permute_there_01_dag = gates.R(
                self.circuit, "R_there_01", index_ctrl, [1, control_block + 1, -np.pi, np.pi / 2], dim_ctrl
            ).dag()

            perm = [permute_there_00, permute_there_01]
            permb = [permute_there_01_dag, permute_there_00_dag]

            return perm + rotation + permb
        return rotation
