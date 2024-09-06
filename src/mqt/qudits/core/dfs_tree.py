from __future__ import annotations

from typing import TYPE_CHECKING

from ..exceptions import NodeNotFoundError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..quantum_circuit import gates
    from ..quantum_circuit.gates import CustomOne
    from . import LevelGraph


class Node:
    def __init__(
        self,
        key: int,
        rotation: gates.R | CustomOne,
        u_of_level: NDArray,
        graph_current: LevelGraph,
        current_cost: float,
        current_decomp_cost: float,
        max_cost: tuple[float, float],
        pi_pulses: list[gates.R],
        parent_key: int | None = None,
        children: list[Node] | None = None,
    ) -> None:
        if children is None:
            children = []
        self.key = key
        self.children: list[Node] = children
        self.rotation = rotation
        self.u_of_level = u_of_level
        self.finished: bool = False
        self.current_cost = current_cost
        self.current_decomp_cost = current_decomp_cost
        self.max_cost = max_cost
        self.size = 0
        self.parent_key = parent_key
        self.graph = graph_current
        self.PI_PULSES = pi_pulses

    def add(
        self,
        new_key: int,
        rotation: gates.R | CustomOne,
        u_of_level: NDArray,
        graph_current: LevelGraph,
        current_cost: float,
        current_decomp_cost: float,
        max_cost: tuple[float, float],
        pi_pulses: list[gates.R],
    ) -> None:
        # TODO refactor so that size is kept track also in the tree upper structure

        new_node = Node(
            new_key,
            rotation,
            u_of_level,
            graph_current,
            current_cost,
            current_decomp_cost,
            max_cost,
            pi_pulses,
            self.key,
        )
        if self.size == 0:
            self.children = []

        self.children.append(new_node)

        self.size += 1

    def __str__(self) -> str:
        return str(self.key)

    def __bool__(self) -> bool:
        return True


class NAryTree:
    # todo put method to refresh size when algorithm has finished

    def __init__(self) -> None:
        self.size: int = 0
        self.global_id_counter: int = 0

    def add(
        self,
        new_key: int,
        rotation: gates.R | CustomOne,
        u_of_level: NDArray,
        graph_current: LevelGraph,
        current_cost: float,
        current_decomp_cost: float,
        max_cost: tuple[float, float],
        pi_pulses: list[gates.R],
        parent_key: int | None = None,
    ) -> None:
        if parent_key is None:
            self.root = Node(
                new_key,
                rotation,
                u_of_level,
                graph_current,
                current_cost,
                current_decomp_cost,
                max_cost,
                pi_pulses,
                parent_key,
            )
            self.size = 1
        else:
            parent_node = self.find_node(self.root, parent_key)
            if not parent_node:
                msg = "No element was found with the informed parent key."
                raise NodeNotFoundError(msg)
            parent_node.add(
                new_key, rotation, u_of_level, graph_current, current_cost, current_decomp_cost, max_cost, pi_pulses
            )
            self.size += 1

    def find_node(self, node: Node, key: int) -> Node | None:
        if node.key == key or node is None:
            return node

        if node.children is not None:
            for child in node.children:
                return_node = self.find_node(child, key)
                if return_node:
                    return return_node
        return None

    def depth(self, key: int) -> int:
        # GIVES DEPTH FROM THE KEY NODE to LEAVES
        node = self.find_node(self.root, key)
        if not node:
            msg = "No element was found with the informed parent key."
            raise NodeNotFoundError(msg)
        return self.max_depth(node)

    def max_depth(self, node: Node) -> int:
        if not node.children:
            return 0
        children_max_depth = [self.max_depth(child) for child in node.children]
        return 1 + max(children_max_depth)

    def size_refresh(self, node: Node) -> int:
        if len(node.children) == 0:
            return 0
        children_size = len(node.children)
        for child in node.children:
            children_size += self.size_refresh(child)

        return children_size

    def found_checker(self, node: Node) -> bool:
        if not node.children:
            return node.finished

        children_checking = [self.found_checker(child) for child in node.children]
        if True in children_checking:
            node.finished = True

        return node.finished

    def min_cost_decomp(self, node: Node) -> tuple[list[Node], tuple[float, float], LevelGraph]:
        if not node.children:
            return [node], (node.current_cost, node.current_decomp_cost), node.graph

        children_cost = [self.min_cost_decomp(child) for child in node.children if child.finished]

        minimum_child, best_cost, final_graph = min(children_cost, key=lambda t: t[1][0])
        minimum_child.insert(0, node)
        return minimum_child, best_cost, final_graph

    def retrieve_decomposition(self, node: Node) -> tuple[list[Node], tuple[float, float], LevelGraph]:
        self.found_checker(node)

        if not node.finished:
            decomp_nodes: list[Node] = []
            from numpy import inf

            best_cost = (inf, inf)
            final_graph = node.graph
        else:
            decomp_nodes, best_cost, final_graph = self.min_cost_decomp(node)

        return decomp_nodes, best_cost, final_graph

    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def total_size(self) -> int:
        self.size = self.size_refresh(self.root)
        return self.size + 1

    def print_tree(self, node: Node, str_aux: str) -> str:
        f = ""
        if node.finished:
            f = "-Finished-"
        str_aux += "N" + str(node) + f + "("
        if node.size > 0:
            str_aux += "\n\t"
            for i in range(len(node.children)):
                child = node.children[i]
                end = "," if i < len(node.children) - 1 else ""
                str_aux = self.print_tree(child, str_aux) + end

        str_aux += ")"
        return str_aux

    def __str__(self) -> str:
        return self.print_tree(self.root, "")
