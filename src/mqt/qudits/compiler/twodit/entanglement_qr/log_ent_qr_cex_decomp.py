#!/usr/bin/env python3
from __future__ import annotations

import gc
import typing
from operator import itemgetter

import numpy as np
from numpy import matmul as mml
from numpy.linalg import det, solve

from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes

from ...compilation_minitools import on0, on1, pi_mod
from ...compiler_pass import CompilerPass
from ..blocks.crot import CRotGen
from ..blocks.czrot import CZRotGen
from ..blocks.pswap import PSwapGen

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class LogEntQRCEXPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)

    @staticmethod
    def transpile_gate(gate: Gate) -> list[Gate]:
        eqr = EntangledQRCEX(gate)
        decomp, _countcr, _countpsw = eqr.execute()
        return decomp

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []
        for gate in instructions:
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                gate_trans.reverse()
                new_instructions.extend(gate_trans)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class EntangledQRCEX:
    def __init__(self, gate: Gate) -> None:
        self.gate: Gate = gate
        self.circuit: QuantumCircuit = gate.parent_circuit
        self.dimensions: list[int] = itemgetter(*gate.reference_lines)(self.circuit.dimensions)
        self.qudit_indices: list[int] = gate.reference_lines
        self.u: NDArray = gate.to_matrix(identities=0)
        self.decomposition: list[Gate] = []

    @staticmethod
    def get_gate_matrix(rotation: Gate, qudit_indices: list[int], dimensions: list[int]) -> NDArray:
        if rotation.gate_type != GateTypes.SINGLE:
            return rotation.to_matrix()

        if rotation.target_qudits == qudit_indices[0]:
            return on0(rotation.to_matrix(), dimensions[1])
        return on1(rotation.to_matrix(), dimensions[0])

    def execute(self) -> tuple[list[Gate], int, int]:
        crot_counter = 0
        pswap_counter = 0

        pswap_gen = PSwapGen(self.circuit, self.qudit_indices)
        crot_gen = CRotGen(self.circuit, self.qudit_indices)
        czrot_gen = CZRotGen(self.circuit, self.qudit_indices)

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
                        sequence_rotation_involved = pswap_gen.permute_pswap_101_as_list(r - 1, theta, phi)
                        pswap_counter += 4
                    else:
                        sequence_rotation_involved = crot_gen.permute_crot_101_as_list(r - 1, theta, phi)
                        crot_counter += 1
                    ######################

                    for rotation in sequence_rotation_involved:
                        gate_matrix = self.get_gate_matrix(rotation, self.qudit_indices, self.dimensions)
                        u_ = gate_matrix @ u_

                    decomp += sequence_rotation_involved

        diag_u = np.diag(u_)
        args_of_diag = [round(np.angle(diag_u[i]), 6) for i in range(matrix_dimension)]

        phase_equations = np.zeros((matrix_dimension, matrix_dimension - 1))

        last_1 = -1
        for i in range(matrix_dimension):
            if last_1 + 1 < matrix_dimension - 1:
                phase_equations[i, last_1 + 1] = 1

            if last_1 > -1:
                phase_equations[i, last_1] = -1

            last_1 = i

        phases_t = phase_equations.conj().T
        pseudo_inv = mml(phases_t, phase_equations)
        pseudo_diag = mml(phases_t, np.array(args_of_diag))

        if det(pseudo_inv) == 0:
            raise RuntimeError

        phases = solve(pseudo_inv, pseudo_diag)

        for i, phase in enumerate(phases):
            if abs(phase * 2) > 1.0e-4:
                if i != 0 and np.mod(i + 1, dim_target) == 0:
                    sequence_rotation_involved = czrot_gen.z_pswap_101_as_list(i, phase * 2)
                    pswap_counter += 12
                else:
                    sequence_rotation_involved = czrot_gen.z_from_crot_101_list(i, phase * 2)
                    crot_counter += 3
                ######################
                for r___ in sequence_rotation_involved:
                    if r___.gate_type == GateTypes.SINGLE:
                        if r___.target_qudits == self.qudit_indices[0]:
                            gate_matrix = on0(r___.to_matrix(), self.dimensions[1])
                        else:
                            gate_matrix = on1(r___.to_matrix(), self.dimensions[0])
                    else:
                        gate_matrix = r___.to_matrix()

                    u_ = gate_matrix @ u_
                u_.round(3)
                #######################

                decomp += sequence_rotation_involved

        self.decomposition = decomp
        return decomp, crot_counter, pswap_counter
