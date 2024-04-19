from __future__ import annotations

from math import floor

import numpy as np

from ....quantum_circuit import gates
from .crotgen import CEX_SEQUENCE


class PSwapGen:
    def __init__(self, circuit, indices) -> None:
        self.circuit = circuit
        self.indices = indices

    def pswap_101_as_list(self, teta, phi):
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        h_0 = gates.Rh(self.circuit, "Rh", index_ctrl, [0, 1], dim_ctrl)
        h_1 = gates.Rh(self.circuit, "Rh", index_target, [0, 1], dim_target)
        # HditR(0, 1, d).matrix

        zpiov2_0 = gates.Rz(self.circuit, "Rz-zpiov2", index_ctrl, [0, 1, np.pi / 2], dim_ctrl)
        # ZditR(np.pi / 2, 0, 1, d).matrix

        zp_0 = gates.Rz(self.circuit, "Rz-zp", index_ctrl, [0, 1, np.pi], dim_ctrl)
        # ZditR(np.pi, 0, 1, d).matrix

        rphi_there_1 = gates.R(self.circuit, "R", index_target, [0, 1, np.pi / 2, -phi - np.pi / 2], dim_target)
        # R(np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

        rphi_back_1 = gates.R(self.circuit, "R", index_target, [0, 1, -np.pi / 2, -phi - np.pi / 2], dim_target)
        # R(-np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix

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
            cex = CEX_SEQUENCE

        ph1 = -1 * np.identity(d, dtype="complex")
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

        ##############################################################################

        compose = [ph1, h_0]  # [on0(ph1, d), on0(h_, d)]

        #################################

        if dim_target != 2:
            r_flip_1 = gates.R(self.circuit, "R", index_target, [1, dim_target - 1, np.pi, np.pi / 2], dim_target)
            compose.append(r_flip_1)  # (on1(R(np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        compose.append(h_1)  # (on1(h_, d))

        compose.append(zpiov2_0)  # (on0(zpiov2, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(h_0)  # (on0(h_, d))

        compose.append(h_1)  # (on1(h_, d))

        ###############################################################
        compose.append(zp_0)  # (on0(zp, d))

        compose.append(rphi_there_1)  # (on1(rphi_there, d))  # ----------

        ##################################

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        zth_1 = gates.Rz(self.circuit, "Rz-th_1", index_target, [0, 1, teta / 2], dim_target)
        compose.append(zth_1)  # (on1(ZditR(teta / 2, 0, 1, d).matrix, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        zth_back_1 = gates.Rz(self.circuit, "Rz-thback_1", index_target, [0, 1, -teta / 2], dim_target)
        compose.append(zth_back_1)
        # compose.append(on1(ZditR(-teta / 2, 0, 1, d).matrix, d))

        ##################################

        compose.append(rphi_back_1)  # (on1(rphi_back, d))

        ##################################

        compose.append(h_0)  # (on0(h_, d))

        compose.append(h_1)  # (on1(h_, d))

        compose.append(zpiov2_0)  # (on0(zpiov2, d))

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose = compose + cex

        compose.append(h_0)  # (on0(h_, d))

        compose.append(h_1)  # (on1(h_, d))

        ##########################
        if dim_target != 2:
            r_flip_back_1 = gates.R(
                self.circuit, "R_flip_back", index_target, [1, dim_target - 1, -np.pi, np.pi / 2], dim_target
            )
            compose.append(r_flip_back_1)  # (on1(R(-np.pi, np.pi / 2, 1, d - 1, d).matrix, d))

        return compose

    def permute_pswap_101_as_list(self, pos, theta, phase):
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        control_block = floor(pos / dim_target)
        rotation = self.pswap_101_as_list(theta, phase)

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

    def permute_quad_pswap_101_as_list(self, pos, theta, phase):
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        control_block = floor(pos / dim_target)
        rotation = self.pswap_101_as_list(theta / 4, phase)
        rotation.reverse()

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

            """
            if j != 0:
                permute_there_00 = on0(R(np.pi, -np.pi / 2, 0, j, d).matrix, d)
                permute_there_01 = on0(R(-np.pi, np.pi / 2, 1, j + 1, d).matrix, d)
                perm = matmul(permute_there_00, permute_there_01)
                permb = perm.conj().T
           """
            perm = [permute_there_00, permute_there_01]
            permb = [permute_there_01_dag, permute_there_00_dag]
            return permb + rotation + rotation + rotation + rotation + perm
        return rotation + rotation + rotation + rotation

    def z_pswap_101_as_list(self, i, phase, dimension_single):
        pi_there = self.permute_quad_pswap_101_as_list(i, np.pi / 2, 0.0)
        rotate = self.permute_quad_pswap_101_as_list(i, phase, np.pi / 2)
        pi_back = self.permute_quad_pswap_101_as_list(i, -np.pi / 2, 0.0)

        return pi_back + rotate + pi_there
