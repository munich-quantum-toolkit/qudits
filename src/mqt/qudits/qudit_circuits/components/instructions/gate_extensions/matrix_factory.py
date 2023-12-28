import itertools
import operator
from functools import reduce

import numpy as np

from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes


class MatrixFactory:
    def __init__(self, gate, identities_flag):
        self.gate = gate
        self.ids = identities_flag

    def generate_matrix(self):
        lines = self.gate.reference_lines
        circuit = self.gate.parent_circuit
        ref_slice = list(range(min(lines), max(lines) + 1))
        dimensions_slice = circuit.dimensions[min(lines) : max(lines) + 1]
        matrix = self.gate.__array__()
        if self.gate.dagger:
            matrix = matrix.conj().T

        control_info = self.gate.control_info

        if self.gate.gate_type != GateTypes.SINGLE:
            if control_info["controls"]:
                controls = control_info.indices
                ctrl_levs = control_info.ctrl_states
                # supported only CONTROLLED-One qudit gates to be made as multi-controlled, it is still a low level
                # control library
                matrix = MatrixFactory.apply_identites_and_controls(
                    matrix, self.gate._target_qudits, dimensions_slice, ref_slice, controls, ctrl_levs
                )
            elif self.ids > 0:
                matrix = MatrixFactory.apply_identites_and_controls(
                    matrix, self.gate._target_qudits, dimensions_slice, ref_slice
                )

        if self.ids >= 2:
            matrix = MatrixFactory.wrap_in_identities(matrix, lines, circuit.dimensions)

        return matrix

    # @classmethod
    # def check_invariant_lists(cls, list1, list2, excluded_indices):
    #     # Check if the lengths of the lists are equal
    #     if len(list1) != len(list2):
    #         return False
    #
    #     # Iterate over the indices of the lists
    #     for i in range(len(list1)):
    #         # Skip the excluded indices
    #         if i in excluded_indices:
    #             continue
    #
    #         # Compare the elements at the current index
    #         if list1[i] != list2[i]:
    #             return False
    #
    #     # All elements at non-excluded indices are equal
    #     return True

    # @classmethod
    # def apply_identites_and_controls(cls, parameters, qudits_applied, dimensions, controls = None, controls_levels = None):
    #
    #     control_dim = dimensions_slice[0]
    #     target_dim = dimensions_slice[-1]
    #
    #     reshaped_tensor = np.reshape(parameters, (control_dim, target_dim, control_dim, target_dim))
    #     reshaped_tensor = np.transpose(reshaped_tensor, (0, 2, 1, 3))
    #
    #     reshaped_dims = reshaped_tensor.shape
    #     parameters = np.reshape(reshaped_tensor, (reshaped_dims[0] * reshaped_dims[1], reshaped_dims[2] * reshaped_dims[3]))
    #     U, S, V = np.linalg.svd(parameters, full_matrices=False)
    #
    #     V = np.diag(S) @ V
    #
    #     U_tensor = np.reshape(U, (control_dim, control_dim, len(S)))
    #     U_tensor = np.expand_dims(U_tensor, axis=0)
    #     U_tensor = np.transpose(U_tensor, (0, 1, 3, 2))
    #     V_tensor = np.reshape(V, (len(S), target_dim, target_dim))
    #     V_tensor = np.expand_dims(V_tensor, axis=3)
    #     V_tensor = np.transpose(V_tensor, (0, 1, 3, 2))
    #
    #     print(U_tensor.shape)
    #     print(V_tensor.shape)
    #     for i, line_dim in enumerate(dimensions_slice[1:-1]):
    #         id_tensor = np.eye(len(S) * line_dim)
    #         id_tensor = np.reshape(id_tensor, (len(S), line_dim, len(S), line_dim))
    #         temp_res = np.einsum("abcd, cefg->abefgd", U_tensor, id_tensor)
    #         dims = temp_res.shape
    #         U_tensor = np.reshape(temp_res, (dims[0], dims[1] * dims[2], dims[3], dims[4] * dims[5]))
    #     result = np.einsum("abcd, cefg->abefgd", U_tensor, V_tensor)
    #     result = result.squeeze()
    #     result = np.transpose(result, (0, 1, 3, 2))
    #     dims = result.shape
    #     result = np.reshape(result, (dims[0] * dims[1], dims[2] * dims[3]))
    #     return result
    @classmethod
    def apply_identites_and_controls(
        cls, matrix, qudits_applied, dimensions, ref_lines, controls=None, controls_levels=None
    ):
        single_site_logics = []
        og_states_space = []
        og_state_to_index = {}
        for d in list(operator.itemgetter(*qudits_applied)(dimensions)):
            single_site_logics.append(list(range(d)))

        for element in itertools.product(*single_site_logics):
            og_states_space.append(list(element))

        for i in range(len(og_states_space)):
            og_state_to_index[tuple(og_states_space[i])] = i

        global_single_site_logics = []
        global_states_space = []
        global_index_to_state = {}
        for d in dimensions:
            global_single_site_logics.append(list(range(d)))

        for element in itertools.product(*global_single_site_logics):
            global_states_space.append(list(element))

        for i in range(len(global_states_space)):
            global_index_to_state[i] = global_states_space[i]

        result = np.zeros((reduce(operator.mul, dimensions, 1), reduce(operator.mul, dimensions, 1)), dtype="complex")

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if controls is not None:
                    if operator.itemgetter(*controls)(global_index_to_state[r]) == controls_levels:
                        rest_of_indices = set(ref_lines) - set(qudits_applied) - set(controls)
                        if operator.itemgetter(*rest_of_indices)(global_index_to_state[r]) == operator.itemgetter(
                            *rest_of_indices
                        )(global_index_to_state[c]):
                            og_row_key = operator.itemgetter(*qudits_applied)(global_index_to_state[r])
                            og_col_key = operator.itemgetter(*qudits_applied)(global_index_to_state[c])
                            matrix_row = og_state_to_index[tuple(og_row_key)]
                            matrix_col = og_state_to_index[tuple(og_col_key)]
                            value = matrix[matrix_row, matrix_col]
                            result[r, c] = value

                else:
                    rest_of_indices = set(ref_lines) - set(qudits_applied)
                    if operator.itemgetter(*rest_of_indices)(global_index_to_state[r]) == operator.itemgetter(
                        *rest_of_indices
                    )(global_index_to_state[c]):
                        og_row_key = operator.itemgetter(*qudits_applied)(global_index_to_state[r])
                        og_col_key = operator.itemgetter(*qudits_applied)(global_index_to_state[c])
                        matrix_row = og_state_to_index[tuple(og_row_key)]
                        matrix_col = og_state_to_index[tuple(og_col_key)]
                        value = matrix[matrix_row, matrix_col]
                        result[r, c] = value

        return result

    @classmethod
    def wrap_in_identities(cls, matrix, indices, sizes):
        sizes = sizes.copy()
        sizes.reverse()
        if any(index >= len(sizes) for index in indices):
            msg = "Index out of range"
            raise ValueError(msg)

        i = 0
        result = np.identity(sizes[i])
        while i < len(sizes):
            if i == indices[0]:
                result = matrix if i == 0 else np.kron(matrix, result)
            else:
                if i > indices[-1]:
                    result = np.kron(np.identity(sizes[i]), result)

            i += 1

        return result
