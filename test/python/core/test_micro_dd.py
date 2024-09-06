from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.core.micro_dd import (
    create_decision_tree,
    cut_branches,
    dd_reduction,
    get_node_contributions,
    normalize,
    normalize_all,
    one,
    zero,
)


class TestMicroDD(TestCase):
    @staticmethod
    def test_terminals():
        assert zero.value == "zero"
        assert zero.terminal is True
        assert zero.dd_hash == hash(0)
        assert one.value == "one"
        assert one.terminal is True
        assert one.dd_hash == hash(1)

    @staticmethod
    def test_normalize():
        # test_all_non_zero_weights
        in_weight = 2 + 3j
        out_weights = [1 + 2j, 3 + 4j]
        expected_in_weight_result = (2 + 3j) * np.sqrt((1**2 + 2**2) + (3**2 + 4**2))
        expected_out_weights_result = [
            (1 + 2j) / np.sqrt((1**2 + 2**2) + (3**2 + 4**2)),
            (3 + 4j) / np.sqrt((1**2 + 2**2) + (3**2 + 4**2)),
        ]
        result_in_weight, result_out_weights = normalize(in_weight, out_weights)
        assert np.allclose(result_in_weight, expected_in_weight_result)
        for result, expected in zip(result_out_weights, expected_out_weights_result):
            assert np.allclose(result, expected)

        # test_zero_out_weights
        in_weight = 2 + 3j
        out_weights = [0 + 0j, 0 + 0j]
        expected_in_weight_result = 0 + 0j
        expected_out_weights_result = [0 + 0j, 0 + 0j]
        result_in_weight, result_out_weights = normalize(in_weight, out_weights)
        assert result_in_weight == expected_in_weight_result
        assert result_out_weights == expected_out_weights_result

        # test_mixed_zero_and_non_zero_weights
        in_weight = 1 + 1j
        out_weights = [0 + 0j, 3 + 4j]
        expected_in_weight_result = (1 + 1j) * np.sqrt(3**2 + 4**2)
        expected_out_weights_result = [(0 + 0j) / np.sqrt(3**2 + 4**2), (3 + 4j) / np.sqrt(3**2 + 4**2)]
        result_in_weight, result_out_weights = normalize(in_weight, out_weights)
        assert np.allclose(result_in_weight, expected_in_weight_result)
        for result, expected in zip(result_out_weights, expected_out_weights_result):
            assert np.allclose(result, expected)

        # test_single_non_zero_weight
        in_weight = 3 + 3j
        out_weights = [5 + 12j]
        expected_in_weight_result = (3 + 3j) * np.sqrt(5**2 + 12**2)
        expected_out_weights_result = [(5 + 12j) / np.sqrt(5**2 + 12**2)]
        result_in_weight, result_out_weights = normalize(in_weight, out_weights)
        assert np.allclose(result_in_weight, expected_in_weight_result)
        assert np.allclose(result_out_weights[0], expected_out_weights_result[0])

    @staticmethod
    def test_create_decision_tree_basic():
        labels = [0, 1]
        cardinalities = [2, 2]
        data = [0.3 + 0j, 0.1 + 0j, 0.01 + 0j, 0.5 + 0j]

        root = create_decision_tree(labels, cardinalities, data)[0]

        assert root.value == "r"
        assert round(root.weight.real, 4) == 0.5917
        assert len(root.children) == 2

        assert root.children[0].value == 0
        assert round(root.children[0].weight.real, 4) == 0.5344
        assert len(root.children[0].children) == 2
        assert root.children[1].value == 0
        assert round(root.children[1].weight.real, 4) == 0.8452
        assert len(root.children[1].children) == 2

        assert root.children[0].children[0].value == 1
        assert round(root.children[0].children[0].weight.real, 4) == 0.9487
        assert len(root.children[0].children[0].children) == 1
        assert root.children[0].children[1].value == 1
        assert round(root.children[0].children[1].weight.real, 4) == 0.3162
        assert len(root.children[0].children[1].children) == 1

        assert root.children[1].children[0].value == 1
        assert round(root.children[1].children[0].weight.real, 3) == 0.02
        assert len(root.children[1].children[0].children) == 1
        assert root.children[1].children[1].value == 1
        assert round(root.children[1].children[1].weight.real, 3) == 1.0
        assert len(root.children[1].children[1].children) == 1

    @staticmethod
    def test_dd_approximation():
        labels = [0, 1]
        cardinalities = [2, 2]
        data = [np.sqrt(1 - 0.02 - 0.01), np.sqrt(0.02), np.sqrt(0.005), np.sqrt(0.005)]

        root, _number_of_nodes = create_decision_tree(labels, cardinalities, data)
        contributions = get_node_contributions(root, labels)
        cut_branches(contributions, 0.02)
        normalize_all(root, cardinalities)

        assert root.value == "r"
        assert root.weight == 0.9949874371066199 + 0j
        assert root.available
        assert len(root.children) == 2

        assert root.children[0].value == 0
        assert root.children[0].available
        assert root.children[0].weight == 1 + 0j
        assert len(root.children[0].children) == 2

        assert root.children[1].value == 0
        assert root.children[1].available
        assert root.children[1].weight == 0 + 0j
        assert len(root.children[1].children) == 1

        assert root.children[0].children[0].value == 1
        assert root.children[0].children[0].available
        assert round(root.children[0].children[0].weight.real, 4) == 0.9898
        assert len(root.children[0].children[0].children) == 1

        assert root.children[0].children[1].value == 1
        assert root.children[0].children[1].available
        assert round(root.children[0].children[1].weight.real, 4) == 0.1421
        assert len(root.children[0].children[1].children) == 1

    @staticmethod
    def test_dd_reduction():
        labels = [0, 1]
        cardinalities = [2, 2]
        data = [np.sqrt(1 - 0.02 - 0.01), np.sqrt(0.02), np.sqrt(0.005), np.sqrt(0.005)]

        root, _number_of_nodes = create_decision_tree(labels, cardinalities, data)
        root = dd_reduction(root, cardinalities)

        assert root.value == "r"
        assert len(root.children) == 2
        assert root.children_index == [0, 1]

        assert root.children[1].value == 0
        assert root.children[1].children_index == [0, 0]
