from mqt.qudits.exceptions.compilerexception import NodeNotFoundException


class Node:
    def __init__(
        self,
        key,
        rotation,
        U_of_level,
        graph_current,
        current_cost,
        current_decomp_cost,
        max_cost,
        pi_pulses,
        parent_key,
        childs=None,
    ):
        self.key = key
        self.children = childs
        self.rotation = rotation
        self.U_of_level = U_of_level
        self.finished = False
        self.current_cost = current_cost
        self.current_decomp_cost = current_decomp_cost
        self.max_cost = max_cost
        self.size = 0
        self.parent_key = parent_key
        self.graph = graph_current
        self.PI_PULSES = pi_pulses

    def add(self, new_key, rotation, U_of_level, graph_current, current_cost, current_decomp_cost, max_cost, pi_pulses):
        # TODO refactor so that size is kept track also in the tree upper structure

        new_node = Node(
            new_key,
            rotation,
            U_of_level,
            graph_current,
            current_cost,
            current_decomp_cost,
            max_cost,
            pi_pulses,
            self.key,
        )
        if self.children is None:
            self.children = []

        self.children.append(new_node)

        self.size += 1

    def __str__(self):
        return str(self.key)


class NAryTree:
    # todo put method to refresh size when algortihm has finished

    def __init__(self):
        self.root = None
        self.size = 0
        self.global_id_counter = 0

    def add(
        self,
        new_key,
        rotation,
        U_of_level,
        graph_current,
        current_cost,
        current_decomp_cost,
        max_cost,
        pi_pulses,
        parent_key=None,
    ):
        if parent_key is None:
            self.root = Node(
                new_key,
                rotation,
                U_of_level,
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
                raise NodeNotFoundException(msg)
            parent_node.add(
                new_key, rotation, U_of_level, graph_current, current_cost, current_decomp_cost, max_cost, pi_pulses
            )
            self.size += 1

    def find_node(self, node, key):
        if node is None or node.key is key:
            return node

        if node.children is not None:
            for child in node.children:
                return_node = self.find_node(child, key)
                if return_node:
                    return return_node
        return None

    def depth(self, key):
        # GIVES DEPTH FROM THE KEY NODE to LEAVES
        node = self.find_node(self.root, key)
        if not (node):
            msg = "No element was found with the informed parent key."
            raise NodeNotFoundException(msg)
        return self.max_depth(node)

    def max_depth(self, node):
        if not node.children:
            return 0
        children_max_depth = []
        for child in node.children:
            children_max_depth.append(self.max_depth(child))
        return 1 + max(children_max_depth)

    def size_refresh(self, node):
        if node.children is None or len(node.children) == 0:
            return 0
        else:
            children_size = len(node.children)
            for child in node.children:
                children_size = children_size + self.size_refresh(child)

            return children_size

    def found_checker(self, node):
        if not node.children:
            return node.finished

        children_checking = []
        for child in node.children:
            children_checking.append(self.found_checker(child))
        if True in children_checking:
            node.finished = True

        return node.finished

    def min_cost_decomp(self, node):
        if not node.children:
            return [node], (node.current_cost, node.current_decomp_cost), node.graph
        else:
            children_cost = []

            for child in node.children:
                if child.finished:
                    children_cost.append(self.min_cost_decomp(child))

            minimum_child, best_cost, final_graph = min(children_cost, key=lambda t: t[1][0])
            minimum_child.insert(0, node)
            return minimum_child, best_cost, final_graph

    def retrieve_decomposition(self, node):
        self.found_checker(node)

        if not node.finished:
            decomp_nodes = []
            from numpy import inf

            best_cost = inf
            final_graph = node.graph
        else:
            decomp_nodes, best_cost, final_graph = self.min_cost_decomp(node)

        return decomp_nodes, best_cost, final_graph

    def is_empty(self):
        return self.size == 0

    @property
    def total_size(self):
        self.size = self.size_refresh(self.root)
        return self.size + 1

    def print_tree(self, node, str_aux):
        if node is None:
            return "Empty tree"
        f = ""
        if node.finished:
            f = "-Finished-"
        str_aux += "N" + str(node) + f + "("
        if node.children is not None:
            str_aux += "\n\t"
            for i in range(len(node.children)):
                child = node.children[i]
                end = "," if i < len(node.children) - 1 else ""
                str_aux = self.print_tree(child, str_aux) + end

        str_aux += ")"
        return str_aux

    def __str__(self):
        return self.print_tree(self.root, "")
