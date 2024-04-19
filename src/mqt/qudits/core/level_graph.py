from __future__ import annotations

import copy

import networkx as nx
import numpy as np

from ..quantum_circuit.gates.virt_rz import VirtRz


class LevelGraph(nx.Graph):
    def __init__(self, edges, nodes, nodes_physical_mapping=None, initialization_nodes=None) -> None:
        super().__init__()
        self.logic_nodes = nodes
        self.add_nodes_from(self.logic_nodes)

        if nodes_physical_mapping:
            self.logic_physical_map(nodes_physical_mapping)

        self.add_edges_from(edges)

        if initialization_nodes:
            inreach_nodes = [x for x in nodes if x not in initialization_nodes]
            self.define__states(initialization_nodes, inreach_nodes)

    def phase_storing_setup(self) -> None:
        for node in self.nodes:
            node_dict = self.nodes[node]
            if "phase_storage" not in node_dict:
                node_dict["phase_storage"] = 0

    def distance_nodes(self, source, target):
        path = nx.shortest_path(self, source, target)
        return len(path) - 1

    def distance_nodes_pi_pulses_fixed_ancilla(self, source, target):
        path = nx.shortest_path(self, source, target)
        negs = 0
        pos = 0
        for n in path:
            if n >= 0:
                pos += 1
            else:
                negs += 1
        return (2 * negs) - 1 + (pos) - 1

    def logic_physical_map(self, physical_nodes) -> None:
        logic_phy_map = dict(zip(self.logic_nodes, physical_nodes))
        nx.set_node_attributes(self, logic_phy_map, "lpmap")

    def define__states(self, initialization_nodes, inreach_nodes) -> None:
        inreach_dictionary = dict.fromkeys(inreach_nodes, "r")
        initialization_dictionary = dict.fromkeys(initialization_nodes, "i")

        for _n in inreach_dictionary:
            nx.set_node_attributes(self, inreach_dictionary, name="level")

        for _n in initialization_dictionary:
            nx.set_node_attributes(self, initialization_dictionary, name="level")

    def update_list(self, lst_, num_a, num_b):
        new_lst = []

        mod_index = []
        for i, t in enumerate(lst_):
            tupla = [0, 0]
            if t[0] == num_a:
                tupla[0] = 1
            elif t[0] == num_b:
                tupla[0] = 2

            if t[1] == num_a:
                tupla[1] = 1
            elif t[1] == num_b:
                tupla[1] = 2

            mod_index.append(tupla)

        for i, t in enumerate(lst_):
            substituter = list(t)

            if mod_index[i][0] == 1:
                substituter[0] = num_b
            elif mod_index[i][0] == 2:
                substituter[0] = num_a

            if mod_index[i][1] == 1:
                substituter[1] = num_b
            elif mod_index[i][1] == 2:
                substituter[1] = num_a

            new_lst.append(tuple(substituter))

        return new_lst

    def deep_copy_func(self, l_n):
        cpy_list = []
        for li in l_n:
            d2 = copy.deepcopy(li)
            cpy_list.append(d2)

        return cpy_list

    def index(self, lev_graph, node):
        for i in range(len(lev_graph)):
            if lev_graph[i][0] == node:
                return i
        return None

    def swap_node_attributes(self, node_a, node_b):
        nodelistcopy = self.deep_copy_func(list(self.nodes(data=True)))
        node_a = self.index(nodelistcopy, node_a)
        node_b = self.index(nodelistcopy, node_b)

        dict_attr_inode = nodelistcopy[0][1]
        for attr in list(dict_attr_inode.keys()):
            attr_a = nodelistcopy[node_a][1][attr]
            attr_b = nodelistcopy[node_b][1][attr]
            nodelistcopy[node_a][1][attr] = attr_b
            nodelistcopy[node_b][1][attr] = attr_a

        return nodelistcopy

    def swap_node_attr_simple(self, node_a, node_b) -> None:
        res_list = [x[0] for x in self.nodes(data=True)]
        node_a = res_list.index(node_a)
        node_b = res_list.index(node_b)

        inode = self._1stInode
        if "phase_storage" in self.nodes[inode]:
            phi_a = self.nodes[node_a]["phase_storage"]
            phi_b = self.nodes[node_b]["phase_storage"]
            self.nodes[node_a]["phase_storage"] = phi_b
            self.nodes[node_b]["phase_storage"] = phi_a

    def swap_nodes(self, node_a, node_b):
        nodes = self.swap_node_attributes(node_a, node_b)
        # ------------------------------------------------
        new_Graph = LevelGraph([], nodes)

        edges = self.deep_copy_func(list(self.edges))

        attribute_list = []
        for e in edges:
            attribute_list.append(self.get_edge_data(*e).copy())

        swapped_nodes_edges = self.update_list(edges, node_a, node_b)

        new_edge_list = []
        for i, e in enumerate(swapped_nodes_edges):
            new_edge_list.append((*e, attribute_list[i]))

        new_Graph.add_edges_from(new_edge_list)

        return new_Graph

    def get_VRz_gates(self):
        matrices = []
        for node in self.nodes:
            node_dict = self.nodes[node]
            if "phase_storage" in node_dict:
                if node_dict["phase_storage"] > 1e-3 or np.mod(node_dict["phase_storage"], 2 * np.pi) > 1e-3:
                    phy_n_i = self.nodes[node]["lpmap"]

                    phase_gate = VirtRz(node_dict["phase_storage"], phy_n_i, len(list(self.nodes)))
                    matrices.append(phase_gate)

        return matrices

    def get_node_sensitivity_cost(self, node):
        neighbs = list(self.neighbors(node))

        total_sensibility = 0
        for i in range(len(neighbs)):
            total_sensibility += self[node][neighbs[i]]["sensitivity"]

        return total_sensibility

    def get_edge_sensitivity(self, node_a, node_b):
        return self[node_a][node_b]["sensitivity"]

    @property
    def _1stRnode(self):
        r_node = [x for x, y in self.nodes(data=True) if y["level"] == "r"]
        return r_node[0]

    @property
    def _1stInode(self):
        Inode = [x for x, y in self.nodes(data=True) if y["level"] == "i"]
        return Inode[0]

    def is_irnode(self, node):
        irnodes = [x for x, y in self.nodes(data=True) if y["level"] == "r"]
        return node in irnodes

    def is_Inode(self, node):
        Inodes = [x for x, y in self.nodes(data=True) if y["level"] == "i"]
        return node in Inodes

    @property
    def log_phy_map(self):
        nodes = self.nodes
        map_as_list = []

        for key in nodes:
            for N in self.nodes(data=True):
                if N[0] == key:
                    map_as_list.append(N[1]["lpmap"])
        return map_as_list

    def __str__(self) -> str:
        return str(self.nodes(data=True)) + "\n" + str(self.edges(data=True))
