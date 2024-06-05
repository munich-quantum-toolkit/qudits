from __future__ import annotations

import math
import operator


class TreeNode:
    def __init__(self, label) -> None:
        self.id = None
        self.value = label
        self.children = []
        self.reduced = None
        self.children_index = []
        self.weight = None
        self.p = None
        self.terminal = False
        self.dd_hash = None
        self.available = True

    def __lt__(self, other):
        # Compare based on the 'value' field
        return self.value < int(other.value)

    def __gt__(self, other):
        # based on the inverse of lt
        return other.__lt__(self)

    def __le__(self, other):
        # based on the other two for efficiency
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        # based on the other two for efficiency
        return self.__gt__(other) or self.__eq__(other)


zero = TreeNode("zero")
zero.terminal = True
zero.dd_hash = hash(0)
one = TreeNode("one")
one.terminal = True
one.dd_hash = hash(1)


def getNodeContributions(root, labels):
    q = []
    probs = {root: abs(root.weight) ** 2}

    q.append(root)

    while q:
        node = q.pop(0)
        parentProb = probs[node]

        for c in node.children:
            if c.weight != 0 + 0j:
                if c not in probs:
                    probs[c] = 0
                probs[c] += parentProb * abs(c.weight) ** 2

                if not c.terminal:
                    q.append(c)

    qq = [[] for _ in range(len(labels))]

    for node, probability in probs.items():
        if node.value not in {"r", "zero", "one"}:
            qq[node.value].append((node, probability))

    for i in range(len(qq)):
        qq[i] = sorted(qq[i], key=operator.itemgetter(1))

    return qq


def unique_weights(root):
    set_unique_weights = set()  # To store the unique weights
    stack = [(root, root.weight)]
    while stack:
        current_node, parent_weight = stack.pop()

        # Add the edge weight to the set of unique weights
        if parent_weight is not None:
            set_unique_weights.add(parent_weight)

        # Add children to the stack for further traversal
        for child in current_node.children:
            stack.append((child, child.weight))

    return set_unique_weights


def normalize(in_weight: complex, out_weights: list[complex]):
    mags_squared = [x.real**2 + x.imag**2 for x in out_weights]
    norm_squared = sum(mags_squared)
    norm = math.sqrt(norm_squared)
    if norm == 0:
        return 0 + 0j, out_weights

    common_factor = norm

    normalized_numbers = [num / norm for num in out_weights]

    return in_weight * common_factor, normalized_numbers


def create_decision_tree(labels, cardinalities, data):
    root = TreeNode("r")
    root.data = data
    number_of_nodes = [1]
    build_decision_tree(labels, root, cardinalities, data, number_of_nodes)

    return root, number_of_nodes


def build_decision_tree(labels, node, cardinalities, data, number_of_nodes, depth=0) -> None:
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

        child = TreeNode(labels[depth])
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


def dd_approximation(node, cardinalities, tolerance, depth=0) -> None:
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


def remove_children(node) -> None:
    for child in node.children:
        child.available = False
        remove_children(child)


def cut_branches(contributions, tolerance) -> None:
    current = 0
    for level in reversed(contributions):
        for node, prob in level:
            if current + prob < tolerance and node.available:
                node.weight = 0 + 0j
                remove_children(node)
                current += prob
            else:
                break


def normalize_all(node, cardinalities, depth=0) -> None:
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


def dd_reduction_hashing(node, cardinalities, depth=0):
    collect_hash_data = []
    if depth == len(cardinalities) or node.terminal:
        collect_hash_data.extend((node.children[0].dd_hash, 1))
    else:
        for i in range(cardinalities[depth]):
            collect_hash_data.extend((
                dd_reduction_hashing(node.children[i], cardinalities, depth + 1),
                node.children[i].weight,
            ))

    collect_hash_data = tuple(collect_hash_data)
    node.dd_hash = hash(collect_hash_data)

    return node.dd_hash


def dd_reduction_aggregation(node, cardinalities, depth=0) -> None:
    if depth == len(cardinalities):
        return
    previous_objects = {}
    previous_objects_index = {}

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
        reduced_children = []
        for c_i in range(cardinalities[depth]):
            if abs(node.children[c_i].weight) != 0.0:
                reduced_children.append(node.children_index[c_i])

        if len(reduced_children) == 1:
            node.reduced = False
        else:
            node.reduced = True
            first_element = reduced_children[0]
            for element in reduced_children[1:]:
                if element != first_element:
                    node.reduced = False


def dd_reduction(root, cardinalities):
    dd_reduction_hashing(root, cardinalities)
    dd_reduction_aggregation(root, cardinalities)
    return root


def count_nodes_after(node, counter, cardinalities, depth=0) -> None:
    counter[0] += 1
    if node.terminal:
        return
    if not node.reduced:
        for i in range(cardinalities[depth]):
            count_nodes_after(node.children[node.children_index[i]], counter, cardinalities, depth + 1)
    else:
        count_nodes_after(node.children[node.children_index[0]], counter, cardinalities, depth + 1)


def print_decision_weights(node, indent="") -> None:
    print(indent + "Q " + str(node.value), node.weight)
    for child in node.children:
        print_decision_weights(child, indent + "  ")


def print_decision_obj_id(node, indent="") -> None:
    print(indent + "Q " + str(node.value), id(node))
    for child in node.children:
        print_decision_obj_id(child, indent + "  ")


def print_decision_hash(node, indent="") -> None:
    print(indent + "Q " + str(node.value), node.dd_hash)
    for child in node.children:
        print_decision_hash(child, indent + "  ")
