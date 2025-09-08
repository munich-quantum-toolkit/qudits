from __future__ import annotations

import math
import operator
from typing import TYPE_CHECKING, cast


class MicroDDNode:
    def __init__(self, label: int | str) -> None:
        self.id: int | None = None
        self.value: int | str = label
        self.children: list[MicroDDNode] = []
        self.reduced: bool | None = None
        self.children_index: list[int] = []
        self.weight: complex = 1
        self.p: complex | None = None
        self.terminal: bool = False
        self.dd_hash: int | None = None
        self.available: bool = True
        self.data: NDArray[complex] | list[complex] = []

    def __lt__(self, other: MicroDDNode) -> bool:
        # Compare based on the 'value' field
        if isinstance(self.value, int) and isinstance(other.value, int):
            return self.value < int(other.value)
        return False

    def __gt__(self, other: MicroDDNode) -> bool:
        # based on the inverse of lt
        return other.__lt__(self)

    def __le__(self, other: MicroDDNode) -> bool:
        # based on the other two for efficiency
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other: MicroDDNode) -> bool:
        # based on the other two for efficiency
        return self.__gt__(other) or self.__eq__(other)


zero: MicroDDNode = MicroDDNode("zero")
zero.terminal = True
zero.dd_hash = hash(0)
one: MicroDDNode = MicroDDNode("one")
one.terminal = True
one.dd_hash = hash(1)

if TYPE_CHECKING:
    NodeContribution = list[list[tuple[MicroDDNode, float]]]
    from numpy.typing import NDArray


def get_node_contributions(root: MicroDDNode, labels: list[int]) -> NodeContribution:
    q = []
    probs = {root: abs(root.weight) ** 2}

    q.append(root)

    while q:
        node = q.pop(0)
        parent_prob = probs[node]

        for c in node.children:
            if c.weight != 0 + 0j:
                if c not in probs:
                    probs[c] = 0
                probs[c] += parent_prob * abs(c.weight) ** 2

                if not c.terminal:
                    q.append(c)

    qq: NodeContribution = [[] for _ in range(len(labels))]

    for node, probability in probs.items():
        if node.value not in {"r", "zero", "one"}:
            qq[cast("int", node.value)].append((node, probability))

    for i in range(len(qq)):
        qq[i] = sorted(qq[i], key=operator.itemgetter(1))

    return qq


def unique_weights(root: MicroDDNode) -> set[complex]:
    set_unique_weights = set()  # To store the unique weights
    stack = [(root, root.weight)]
    while stack:
        current_node, parent_weight = stack.pop()

        # Add the edge weight to the set of unique weights
        if parent_weight is not None:
            set_unique_weights.add(parent_weight)

        # Add children to the stack for further traversal
        stack.extend((child, child.weight) for child in current_node.children)

    return set_unique_weights


def normalize(in_weight: complex, out_weights: list[complex]) -> tuple[complex, list[complex]]:
    mags_squared = [x.real**2 + x.imag**2 for x in out_weights]
    norm_squared = sum(mags_squared)
    norm = math.sqrt(norm_squared)
    if norm == 0:
        return 0 + 0j, out_weights

    common_factor = norm

    normalized_numbers = [num / norm for num in out_weights]

    return in_weight * common_factor, normalized_numbers


def create_decision_tree(
    labels: list[int], cardinalities: list[int], data: NDArray[complex] | list[complex]
) -> tuple[MicroDDNode, list[int]]:
    root = MicroDDNode("r")
    root.data = data
    number_of_nodes = [1]
    build_decision_tree(labels, root, cardinalities, data, number_of_nodes)

    return root, number_of_nodes


def build_decision_tree(
    labels: list[int],
    node: MicroDDNode,
    cardinalities: list[int],
    data: NDArray[complex] | list[complex],
    number_of_nodes: list[int],
    depth: int = 0,
) -> None:
    if depth == len(cardinalities):
        node.weight = data[0]
        if data[0] == 0 + 0j:
            node.terminal = True
            node.children.append(zero)
        else:
            node.terminal = True
            node.children.append(one)
        return

    # Calculate the length of each subarray
    split_index = len(data) // cardinalities[depth]

    node.weight = 1 + 0j

    for i in range(cardinalities[depth]):
        # Split the array into two subarrays
        branch_data = data[i * split_index : (i + 1) * split_index]

        child = MicroDDNode(labels[depth])
        number_of_nodes[0] += 1
        child.data = branch_data

        # Recursively split each subarray
        build_decision_tree(labels, child, cardinalities, branch_data, number_of_nodes, depth + 1)

        node.children.append(child)

    cweights = [c.weight for c in node.children]

    # Managing Probability
    for c in node.children:
        c.p = c.weight**2
    ####################

    node.weight, new_weights = normalize(node.weight, cweights)

    if node.weight == 0 + 0j:
        node.terminal = True
        node.children = [zero]

    for i in range(len(node.children)):
        node.children[i].weight = new_weights[i]


def dd_approximation(node: MicroDDNode, cardinalities: list[int], tolerance: float, depth: int = 0) -> None:
    if depth == len(cardinalities) or node.terminal:
        return

    for i in range(cardinalities[depth]):
        if abs(node.children[i].weight) ** 2 < tolerance and abs(node.children[i].weight) ** 2 != 0.0:
            node.children[i].weight = 0 + 0j
        dd_approximation(node.children[i], cardinalities, tolerance, depth + 1)

    cweights = [c.weight for c in node.children]
    node.weight, new_weights = normalize(node.weight, cweights)

    if node.weight == 0 + 0j:
        node.terminal = True
        node.children = [zero]

    if len(node.children) != 1 and node.children[0] != zero:
        for i in range(len(node.children)):
            node.children[i].weight = new_weights[i]


def remove_children(node: MicroDDNode) -> None:
    for child in node.children:
        child.available = False
        remove_children(child)


def cut_branches(contributions: NodeContribution, tolerance: float) -> None:
    current = 0.0
    for level in reversed(contributions):
        for node, prob in level:
            if current + prob < tolerance and node.available:
                node.weight = 0 + 0j
                remove_children(node)
                current += prob
            else:
                break


def normalize_all(node: MicroDDNode, cardinalities: list[int], depth: int = 0) -> None:
    if depth == len(cardinalities) or node.terminal:
        return

    for i in range(cardinalities[depth]):
        normalize_all(node.children[i], cardinalities, depth + 1)

    cweights = [c.weight for c in node.children]
    node.weight, new_weights = normalize(node.weight, cweights)

    if node.weight == 0 + 0j:
        node.terminal = True
        node.children = [zero]

    if len(node.children) != 1 and node.children[0] != zero:
        for i in range(len(node.children)):
            node.children[i].weight = new_weights[i]


def dd_reduction_hashing(node: MicroDDNode, cardinalities: list[int], depth: int = 0) -> int:
    collect_hash_data: list[tuple[int | None, complex]] = []
    if depth == len(cardinalities) or node.terminal:
        collect_hash_data.append((node.children[0].dd_hash, 1))
    else:
        collect_hash_data = [
            (dd_reduction_hashing(node.children[i], cardinalities, depth + 1), node.children[i].weight)
            for i in range(cardinalities[depth])
        ]

    collect_hash_data_t = tuple(collect_hash_data)
    node.dd_hash = hash(collect_hash_data_t)

    return node.dd_hash


def dd_reduction_aggregation(node: MicroDDNode, cardinalities: list[int], depth: int = 0) -> None:
    if depth == len(cardinalities):
        return
    previous_objects: dict[int | None, MicroDDNode] = {}
    previous_objects_index: dict[int | None, int] = {}

    for i, obj in enumerate(node.children):
        if obj.dd_hash in previous_objects:
            node.children[i].id = id(previous_objects[obj.dd_hash])
            node.children_index.append(previous_objects_index[obj.dd_hash])
        else:
            obj.id = id(obj)
            previous_objects[obj.dd_hash] = obj
            previous_objects_index[obj.dd_hash] = i
            node.children_index.append(i)

    done = []
    for child in node.children:
        if child.id not in done:
            dd_reduction_aggregation(child, cardinalities, depth + 1)
            done.append(child.id)

    if node.terminal:
        node.reduced = True
    else:
        reduced_children = [
            node.children_index[c_i] for c_i in range(cardinalities[depth]) if abs(node.children[c_i].weight) != 0.0
        ]

        if len(reduced_children) == 1:
            node.reduced = False
        else:
            node.reduced = True
            first_element = reduced_children[0]
            for element in reduced_children[1:]:
                if element != first_element:
                    node.reduced = False


def dd_reduction(root: MicroDDNode, cardinalities: list[int]) -> MicroDDNode:
    dd_reduction_hashing(root, cardinalities)
    dd_reduction_aggregation(root, cardinalities)
    return root


def count_nodes_after(node: MicroDDNode, counter: list[int], cardinalities: list[int], depth: int = 0) -> None:
    counter[0] += 1
    if node.terminal:
        return
    if not node.reduced:
        for i in range(cardinalities[depth]):
            count_nodes_after(node.children[node.children_index[i]], counter, cardinalities, depth + 1)
    else:
        count_nodes_after(node.children[node.children_index[0]], counter, cardinalities, depth + 1)


def print_decision_weights(node: MicroDDNode, indent: str = "") -> None:
    print(indent + "Q " + str(node.value), node.weight)  # noqa: T201
    for child in node.children:
        print_decision_weights(child, indent + "  ")


def print_decision_obj_id(node: MicroDDNode, indent: str = "") -> None:
    print(indent + "Q " + str(node.value), id(node))  # noqa: T201
    for child in node.children:
        print_decision_obj_id(child, indent + "  ")


def print_decision_hash(node: MicroDDNode, indent: str = "") -> None:
    print(indent + "Q " + str(node.value), node.dd_hash)  # noqa: T201
    for child in node.children:
        print_decision_hash(child, indent + "  ")
