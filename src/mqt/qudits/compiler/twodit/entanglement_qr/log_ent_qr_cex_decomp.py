from __future__ import annotations

import gc

import numpy as np

from ....quantum_circuit.gate import GateTypes
from ...compilation_minitools import pi_mod
from ...compiler_pass import CompilerPass
from .crotgen import CRotGen
from .pswap import PSwapGen


class LogEntQRCEXPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.TWO:
                energy_graph_c = self.backend.energy_level_graphs[gate._target_qudits[0]]
                energy_graph_t = self.backend.energy_level_graphs[gate._target_qudits[1]]

                eqr = EntangledQRCEX(gate, energy_graph_c, energy_graph_t)
                decomp = eqr.execute()
                new_instructions += decomp
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class EntangledQRCEX:
    def __init__(self, gate, graph_orig_c, graph_orig_t) -> None:
        self.gate = gate
        self.circuit = gate.parent_circuit
        self.dimensions = gate._dimensions
        self.qudit_indices = gate._target_qudits
        self.u = gate.to_matrix(identities=0)
        self.graph_c = graph_orig_c
        self.graph_t = graph_orig_t
        self.decomposition = None
        self.decomp_indexes = []

    def execute(self):
        crot_counter = 0
        pswap_counter = 0

        pswap_gen = PSwapGen(self.circuit, self.qudit_indices)
        crot_gen = CRotGen(self.circuit, self.qudit_indices)

        decomp = []

        u_ = self.u
        dim_control = self.dimensions[0]
        dim_target = self.dimensions[1]
        matrix_dimension = dim_control * dim_target

        index_iterator = list(range(matrix_dimension))
        index_iterator.reverse()

        for c in range(matrix_dimension):
            diag_index = index_iterator.index(c)

            for r in index_iterator[:diag_index]:
                if abs(u_[r, c]) > 1.0e-8:
                    coef_r1 = u_[r - 1, c].round(15)
                    coef_r = u_[r, c].round(15)

                    theta = 2 * np.arctan2(abs(coef_r), abs(coef_r1))

                    phi = -(np.pi / 2 + np.angle(coef_r1) - np.angle(coef_r))

                    phi = pi_mod(phi)
                    #######################
                    if (r - 1) != 0 and np.mod(r, dim_target) == 0:
                        sequence_rotation_involved = pswap_gen.permute_quad_pswap_101_as_list(r - 1, theta, phi)
                        pswap_counter += 4
                    else:
                        sequence_rotation_involved = crot_gen.permute_doubled_crot_101_as_list(r - 1, theta, phi)
                        crot_counter += 2
                    ######################

                    for r___ in sequence_rotation_involved:
                        u_ = r___.to_matrix() @ u_

                    decomp = decomp + sequence_rotation_involved

        diag_u = np.diag(u_)
        args_of_diag = []

        for i in range(matrix_dimension):
            args_of_diag.append(round(np.angle(diag_u[i]), 6))

        phase_equations = np.zeros((matrix_dimension, matrix_dimension - 1))

        last_1 = -1
        for i in range(matrix_dimension):
            if last_1 + 1 < matrix_dimension - 1:
                phase_equations[i, last_1 + 1] = 1

            if last_1 > -1:
                phase_equations[i, last_1] = -1

            last_1 = i

        phases_t = phase_equations.conj().T
        pseudo_inv = np.matmul(phases_t, phase_equations)
        pseudo_diag = np.matmul(phases_t, np.array(args_of_diag))

        if np.linalg.det(pseudo_inv) == 0:
            raise Exception

        phases = np.linalg.solve(pseudo_inv, pseudo_diag)

        for i, phase in enumerate(phases):
            if abs(phase * 2) > 1.0e-4:
                if i != 0 and np.mod(i + 1, dim_target) == 0:
                    sequence_rotation_involved = pswap_gen.z_pswap_101_as_list(i, phase * 2)
                    pswap_counter += 12
                else:
                    sequence_rotation_involved = crot_gen.z_from_crot_101_list(i, phase * 2)
                    crot_counter += 6

                ######################
                for r___ in sequence_rotation_involved:
                    u_ = r___.to_matrix() @ u_
                #######################

                decomp = decomp + sequence_rotation_involved

        self.decomposition = decomp
        return decomp, crot_counter, pswap_counter
