from __future__ import annotations

import numpy as np

from ....quantum_circuit import gates

CEX_SEQUENCE = None  # list of numpy arrays


class CRotGen:
    def __init__(self, circuit, indices) -> None:
        self.circuit = circuit
        self.indices = indices

    def crot_101_as_list(self, theta, phi):
        # Assuming that 0 was control and 1 was target
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        frame_back = gates.R(self.circuit, "R", index_target, [0, 1, -np.pi / 2, -phi - np.pi / 2], dim_target)
        # on1(R(-np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix, d)

        tminus = gates.Rz(self.circuit, "Rz", index_target, [0, 1, -theta / 2], dim_target)
        # on1(ZditR(-theta / 2, 0, 1, d).matrix, d))

        tplus = gates.Rz(self.circuit, "Rz", index_target, [0, 1, +theta / 2], dim_target)
        # on1(ZditR(theta / 2, 0, 1, d).matrix, d)

        frame_there = gates.R(self.circuit, "R", index_target, [0, 1, np.pi / 2, -phi - np.pi / 2], dim_target)
        # on1(R(np.pi / 2, -phi - np.pi / 2, 0, 1, d).matrix, d)

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

        #############

        compose = [frame_there]

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex

        compose.append(tminus)

        if CEX_SEQUENCE is None:
            compose.append(cex)
        else:
            compose += cex

        compose.append(tplus)

        compose.append(frame_back)

        return compose

    def permute_crot_101_as_list(self, i, theta, phase):
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        q0_i = np.floor(i / dim_target)  # finds the control block (level) of control line
        q1_i = i - (dim_target * q0_i)  # finds lev_a or rotated subspace on target

        rot_there = []
        rot_back = []

        rotation = self.crot_101_as_list(theta, phase)

        if q0_i == 1 and q1_i == 0:
            return rotation

        if q1_i != 0:
            permute_there_10 = gates.R(self.circuit, "R", index_target, [0, q1_i, np.pi, -np.pi / 2], dim_target)
            # on1(R(np.pi, -np.pi / 2, 0, q1_i, d).matrix, d)
            permute_there_11 = gates.R(self.circuit, "R", index_target, [1, q1_i + 1, -np.pi, np.pi / 2], dim_target)
            # on1(R(-np.pi, np.pi / 2, 1, q1_i + 1, d).matrix, d)

            permute_there_10_dag = gates.R(
                self.circuit, "R", index_target, [0, q1_i, np.pi, -np.pi / 2], dim_target
            ).dag()
            permute_there_11_dag = gates.R(
                self.circuit, "R", index_target, [1, q1_i + 1, -np.pi, np.pi / 2], dim_target
            ).dag()

            perm = [permute_there_10, permute_there_11]  # matmul(permute_there_10, permute_there_11)
            perm_back = [permute_there_11_dag, permute_there_10_dag]  # perm.conj().T

            rot_there += perm
            rot_back += perm_back

        if q0_i != 1:
            permute_there_00 = gates.R(self.circuit, "R", index_ctrl, [1, q0_i, np.pi, -np.pi / 2], dim_ctrl)
            # on0(R(np.pi, -np.pi / 2, 1, q0_i, d).matrix, d)
            permute_back_00 = gates.R(self.circuit, "R", index_ctrl, [1, q0_i, np.pi, np.pi / 2], dim_ctrl)
            # on0(R(np.pi, np.pi / 2, 1, q0_i, d).matrix, d)

            rot_there.append(permute_there_00)
            rot_back.insert(0, permute_back_00)

        return rot_there + rotation + rot_back

    def permute_doubled_crot_101_as_list(self, i, theta, phase):
        index_ctrl = self.indices[0]
        dim_ctrl = self.circuit.dimensions[index_ctrl]
        index_target = self.indices[1]
        dim_target = self.circuit.dimensions[index_target]

        q0_i = np.floor(i / dim_target)  # finds the control block (level) of control line
        q1_i = i - (dim_target * q0_i)  # finds lev_a or rotated subspace on target

        rot_there = []
        rot_back = []

        rotation = self.crot_101_as_list(-theta / 2, -phase)
        rotation.reverse()

        if q0_i == 1 and q1_i == 0:
            return rotation + rotation

        if q1_i != 0:
            permute_there_10 = gates.R(self.circuit, "R", index_target, [0, q1_i, np.pi, -np.pi / 2], dim_target)
            # on1(R(np.pi, -np.pi / 2, 0, q1_i, d).matrix, d)
            permute_there_11 = gates.R(self.circuit, "R", index_target, [1, q1_i + 1, -np.pi, np.pi / 2], dim_target)
            # on1(R(-np.pi, np.pi / 2, 1, q1_i + 1, d).matrix, d)

            permute_there_10_dag = gates.R(
                self.circuit, "R", index_target, [0, q1_i, np.pi, -np.pi / 2], dim_target
            ).dag()
            permute_there_11_dag = gates.R(
                self.circuit, "R", index_target, [1, q1_i + 1, -np.pi, np.pi / 2], dim_target
            ).dag()

            perm = [permute_there_10, permute_there_11]  # matmul(permute_there_10, permute_there_11)
            perm_back = [permute_there_11_dag, permute_there_10_dag]  # perm.conj().T

            rot_there += perm
            rot_back += perm_back

            """permute_there_10 =
            # on1(R(np.pi, -np.pi / 2, 0, q1_i, d).matrix, d)
            permute_there_11 =
            # on1(R(-np.pi, np.pi / 2, 1, q1_i + 1, d).matrix, d)

            perm = matmul(permute_there_10, permute_there_11)
            permb = perm.conj().T

            rot_there.append(perm)
            rot_back.append(permb)"""

        if q0_i != 1:
            permute_there_00 = gates.R(self.circuit, "R", index_ctrl, [1, q0_i, np.pi, -np.pi / 2], dim_ctrl)
            # on0(R(np.pi, -np.pi / 2, 1, q0_i, d).matrix, d)
            permute_back_00 = gates.R(self.circuit, "R", index_ctrl, [1, q0_i, np.pi, np.pi / 2], dim_ctrl)
            # on0(R(np.pi, np.pi / 2, 1, q0_i, d).matrix, d)

            rot_there.append(permute_there_00)
            rot_back.insert(0, permute_back_00)

            """permute_there_00 =
            # on0(R(np.pi, -np.pi / 2, 1, q0_i, d).matrix, d)
            permute_back_00 =
            # on0(R(np.pi, np.pi / 2, 1, q0_i, d).matrix, d)

            rot_there.append(permute_there_00)
            rot_back.insert(0, permute_back_00)"""

        rot_back.reverse()
        rot_there.reverse()

        return rot_back + rotation + rotation + rot_there

    def z_from_crot_101_list(self, i, phase):
        pi_there = self.permute_doubled_crot_101_as_list(i, np.pi / 2, 0.0)
        rotate = self.permute_doubled_crot_101_as_list(i, phase, np.pi / 2)
        pi_back = self.permute_doubled_crot_101_as_list(i, -np.pi / 2, 0.0)

        return pi_back + rotate + pi_there
