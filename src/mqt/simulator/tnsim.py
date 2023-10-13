import numpy as np
import tensornetwork as tn

system_sizes = [2, 2]


def apply_gate(qudit_edges, gate, operating_qudits):
    op = tn.Node(gate)
    for i, bit in enumerate(operating_qudits):
        tn.connect(qudit_edges[bit], op[i])
        qudit_edges[bit] = op[i + len(operating_qudits)]


all_nodes = []

with tn.NodeCollection(all_nodes):
    state_nodes = []
    for s in system_sizes:
        z = [0] * s
        z[0] = 1
        state_nodes.append(tn.Node(np.array(z, dtype="complex")))

    qudits_legs = [node[0] for node in state_nodes]
    # apply_gate(qudits_legs, H(system_sizes[0]).matrix, [0])
    """apply_gate(
            qudits_legs,
            C_SUM_mixed(system_sizes[0], system_sizes[1]).reshape(
                    (system_sizes[0], system_sizes[1], system_sizes[0], system_sizes[1])
            ),
            [0, 1],
    )"""

print("\n")
result = tn.contractors.optimal(all_nodes, output_edge_order=qudits_legs)
print(result.tensor)
