from __future__ import annotations

import math

import networkx as nx
import numpy as np

from ....quantum_circuit.gates.r import R
from ...compilation_minitools import (
    new_mod,
    pi_mod,
    rotation_cost_calc,
    swap_elements,
)


def find_logic_from_phys(lev_a, lev_b, graph):
    # find node by physical level associated
    logic_nodes = [None, None]
    for node, node_data in graph.nodes(data=True):
        if node_data["lpmap"] == lev_a:
            logic_nodes[0] = node
        if node_data["lpmap"] == lev_b:
            logic_nodes[1] = node

    return logic_nodes


def graph_rule_update(gate, graph) -> None:
    if abs(abs(gate.theta) - math.pi) < 1e-2:
        inode = graph._1stInode
        if "phase_storage" not in graph.nodes[inode]:
            return

        g_lev_a = gate.lev_a
        g_lev_b = gate.lev_b

        logic_nodes = find_logic_from_phys(g_lev_a, g_lev_b, graph)

        # only pi pulses can update online the graph
        if logic_nodes[0] is not None and logic_nodes[1] is not None:
            # SWAPPING PHASES
            graph.swap_node_attr_simple(logic_nodes[0], logic_nodes[1])

            phase = pi_mod(gate.phi)
            if (gate.theta * phase) > 0:
                graph.nodes[logic_nodes[1]]["phase_storage"] = graph.nodes[logic_nodes[1]]["phase_storage"] + np.pi
                graph.nodes[logic_nodes[1]]["phase_storage"] = new_mod(graph.nodes[logic_nodes[1]]["phase_storage"])

            elif (gate.theta * phase) < 0:
                graph.nodes[logic_nodes[0]]["phase_storage"] = graph.nodes[logic_nodes[0]]["phase_storage"] + np.pi
                graph.nodes[logic_nodes[0]]["phase_storage"] = new_mod(graph.nodes[logic_nodes[0]]["phase_storage"])

    return


def graph_rule_ongate(gate, graph):
    inode = graph._1stInode
    if "phase_storage" not in graph.nodes[inode]:
        return gate

    g_lev_a = gate.lev_a
    g_lev_b = gate.lev_b
    new_g_phi = gate.phi  # old phase still inside the gate_matrix

    logic_nodes = find_logic_from_phys(g_lev_a, g_lev_b, graph)

    # MINUS source PLUS target according to pi pulse back
    if logic_nodes[0] is not None:
        new_g_phi = new_g_phi - graph.nodes[logic_nodes[0]]["phase_storage"]
    if logic_nodes[1] is not None:
        new_g_phi = new_g_phi + graph.nodes[logic_nodes[1]]["phase_storage"]

    return R(
        gate.parent_circuit, "R", gate._target_qudits, [g_lev_a, g_lev_b, gate.theta, new_g_phi], gate._dimensions
    )  # R(gate_matrix.theta, new_g_phi, g_lev_a, g_lev_b, gate_matrix.dimension)


def gate_chain_condition(previous_gates, current):
    if not previous_gates:
        return current

    new_source = current.lev_a
    new_target = current.lev_b
    theta = current.theta
    phi = current.phi

    last_gate = previous_gates[-1]
    last_source = last_gate.lev_a
    last_target = last_gate.lev_b

    # all phi flips are removed because already applied
    if new_source == last_source:
        if new_target > last_target or new_target < last_target:  # changed lower one with lower one
            pass

    elif new_target == last_target:
        if new_source < last_source or new_source > last_source:
            theta = theta * -1

    elif new_source == last_target:
        theta = theta * -1

    elif new_target == last_source:
        pass

    return R(
        current.parent_circuit,
        "R",
        current._target_qudits,
        [current.lev_a, current.lev_b, theta, phi],
        current._dimensions,
    )  # R(theta, phi, current.lev_a, current.lev_b, current.dimension)


def route_states2rotate_basic(gate, orig_placement):
    placement = orig_placement
    dimension = gate._dimensions

    cost_of_pi_pulses = 0
    pi_pulses_routing = []

    source = gate.original_lev_a  # Original code requires to know the direction of rotations
    target = gate.original_lev_b

    path = nx.shortest_path(placement, source, target)

    i = len(path) - 2

    while i > 0:
        phy_n_i = placement.nodes[path[i]]["lpmap"]
        phy_n_ip1 = placement.nodes[path[i + 1]]["lpmap"]

        pi_gate_phy = R(
            gate.parent_circuit, "R", gate._target_qudits, [phy_n_i, phy_n_ip1, np.pi, -np.pi / 2], dimension
        )  # R(np.pi, -np.pi / 2, phy_n_i, phy_n_ip1, dimension)

        pi_gate_phy = gate_chain_condition(pi_pulses_routing, pi_gate_phy)
        pi_gate_phy = graph_rule_ongate(pi_gate_phy, placement)

        # -- COSTING based only on the position of the pi pulse and angle phase is neglected ----------------
        pi_gate_logic = R(
            gate.parent_circuit,
            "R",
            gate._target_qudits,
            [path[i], path[i + 1], pi_gate_phy.theta, pi_gate_phy.phi / 2],
            dimension,
        )  # R(pi_gate_phy.theta, pi_gate_phy.phi, path[i], path[i + 1], dimension)
        cost_of_pi_pulses += rotation_cost_calc(pi_gate_logic, placement)
        # -----------------------------------------------------------------------------------------------------
        placement = placement.swap_nodes(path[i + 1], path[i])
        path = swap_elements(path, i + 1, i)

        pi_pulses_routing.append(pi_gate_phy)

        i -= 1

    return cost_of_pi_pulses, pi_pulses_routing, placement


def cost_calculator(gate, placement, non_zeros):
    cost_of_pi_pulses, pi_pulses_routing, new_placement = route_states2rotate_basic(gate, placement)
    gate_cost = rotation_cost_calc(gate, new_placement)
    total_costing = (gate_cost + cost_of_pi_pulses) * non_zeros

    return total_costing, pi_pulses_routing, new_placement, cost_of_pi_pulses, gate_cost
