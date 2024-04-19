from unittest import TestCase
import numpy as np

from gates.entangling_gates import ms_gate
from src import global_vars, customize_vars
from src.decomposer.cex import Cex
from src.layered.compile_pkg.ansatz.instantiate import create_cu_instance, create_ls_instance, create_ms_instance
from src.layered.compile_pkg.opt.distance_measures import fidelity_on_unitares
from src.layered.compile_pkg.search import binary_search_compile

from src.utils.qudit_circ_utils import gate_expand_to_circuit
from src.utils.rotation_utils import matmul


class TestSearch(TestCase):
    def setUp(self) -> None:
        global_vars.OBJ_FIDELITY = 1e-4
        global_vars.SINGLE_DIM = 2
        global_vars.TARGET_GATE = Cex().cex_101(global_vars.SINGLE_DIM)
        global_vars.MAX_NUM_LAYERS = (2 * global_vars.SINGLE_DIM ** 2)

    def test_binary_search_compile_ms(self):

        best_layer, best_error, best_xi = binary_search_compile(global_vars.SINGLE_DIM, global_vars.MAX_NUM_LAYERS,
                                                                "MS")

        decomposed_target = create_ms_instance(best_xi, global_vars.SINGLE_DIM)

        unitary = gate_expand_to_circuit(np.identity(global_vars.SINGLE_DIM, dtype=complex), n=2, target=0,
                                         dim=global_vars.SINGLE_DIM)

        for rot in decomposed_target:
            unitary = matmul(unitary, rot)

        print((1 - fidelity_on_unitares(unitary, global_vars.TARGET_GATE)) < 1e-4)  # outcome should be true

    def test_binary_search_compile_ls(self):

        best_layer, best_error, best_xi = binary_search_compile(global_vars.SINGLE_DIM, global_vars.MAX_NUM_LAYERS,
                                                                "LS")

        decomposed_target = create_ls_instance(best_xi, global_vars.SINGLE_DIM)

        unitary = gate_expand_to_circuit(np.identity(global_vars.SINGLE_DIM, dtype=complex), n=2, target=0,
                                         dim=global_vars.SINGLE_DIM)

        for rot in decomposed_target:
            unitary = matmul(unitary, rot)

        print((1 - fidelity_on_unitares(unitary, global_vars.TARGET_GATE)) < 1e-4)  # outcome should be true

    def test_binary_search_compile_cu(self):

        customize_vars.CUSTOM_PRIMITIVE = ms_gate(np.pi / 2, global_vars.SINGLE_DIM)

        best_layer, best_error, best_xi = binary_search_compile(global_vars.SINGLE_DIM, global_vars.MAX_NUM_LAYERS,
                                                                "CU")

        decomposed_target = create_cu_instance(best_xi, global_vars.SINGLE_DIM)

        unitary = gate_expand_to_circuit(np.identity(global_vars.SINGLE_DIM, dtype=complex), n=2, target=0,
                                         dim=global_vars.SINGLE_DIM)

        for rot in decomposed_target:
            unitary = matmul(unitary, rot)

        print((1 - fidelity_on_unitares(unitary, global_vars.TARGET_GATE)) < 1e-4)  # outcome should be true
