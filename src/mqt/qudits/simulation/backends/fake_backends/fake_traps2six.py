from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

from mqt.qudits.simulation.noise_tools.noise import Noise

from ....core import LevelGraph
from ...noise_tools import NoiseModel
from ..tnsim import TNSim

if TYPE_CHECKING:
    from ... import MQTQuditProvider
    from ..backendv2 import Backend


class FakeIonTraps2Six(TNSim):
    @property
    def version(self) -> int:
        return 0

    def __init__(self, provider: MQTQuditProvider, **fields: Unpack[Backend.DefaultOptions]) -> None:
        super().__init__(
            provider=provider, name="FakeTrap2Six", description="A Fake backend of an ion trap qudit machine", **fields
        )
        self.options["noise_model"] = self.__noise_model()
        self.author = "<Kevin Mato>"
        self._energy_level_graphs: list[LevelGraph] = []

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        if len(self._energy_level_graphs) == 0:
            e_graphs: list[LevelGraph] = []
            # declare the edges on the energy level graph between logic states .
            edges = [
                (2, 0, {"delta_m": 0, "sensitivity": 3}),
                (3, 0, {"delta_m": 0, "sensitivity": 3}),
                (4, 0, {"delta_m": 0, "sensitivity": 4}),
                (5, 0, {"delta_m": 0, "sensitivity": 4}),
                (1, 2, {"delta_m": 0, "sensitivity": 4}),
                (1, 3, {"delta_m": 0, "sensitivity": 3}),
                (1, 4, {"delta_m": 0, "sensitivity": 3}),
                (1, 5, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes = [0, 1, 2, 3, 4, 5]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap = [0, 1, 2, 3, 4, 5]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_0 = LevelGraph(edges, nodes, nmap, [1])
            # declare the edges on the energy level graph between logic states .
            edges_1 = [
                (2, 0, {"delta_m": 0, "sensitivity": 3}),
                (3, 0, {"delta_m": 0, "sensitivity": 3}),
                (4, 0, {"delta_m": 0, "sensitivity": 4}),
                (5, 0, {"delta_m": 0, "sensitivity": 4}),
                (1, 2, {"delta_m": 0, "sensitivity": 4}),
                (1, 3, {"delta_m": 0, "sensitivity": 3}),
                (1, 4, {"delta_m": 0, "sensitivity": 3}),
                (1, 5, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes_1 = [0, 1, 2, 3, 4, 5]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap_1 = [0, 1, 2, 3, 4, 5]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_1 = LevelGraph(edges_1, nodes_1, nmap_1, [1])
            e_graphs.extend((graph_0, graph_1))

            self._energy_level_graphs = e_graphs
        return self._energy_level_graphs

    def __noise_model(self) -> NoiseModel | None:
        # Depolarizing quantum errors
        local_error = Noise(probability_depolarizing=0.001, probability_dephasing=0.001)
        local_error_rz = Noise(probability_depolarizing=0.03, probability_dephasing=0.03)
        entangling_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.001)
        entangling_error_extra = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        entangling_error_on_target = Noise(probability_depolarizing=0.1, probability_dephasing=0.0)
        entangling_error_on_control = Noise(probability_depolarizing=0.01, probability_dephasing=0.0)

        # Add errors to noise_tools model

        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate_matrix
        noise_model.add_all_qudit_quantum_error(local_error, ["csum"])
        # Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_target(entangling_error_on_target, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_control(entangling_error_on_control, ["csum", "cx", "ls", "ms"])
        # Super noisy Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error_extra, ["csum"])
        # Local Gates
        noise_model.add_quantum_error_locally(local_error, ["h", "rxy", "s", "x", "z"])
        noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])
        self.noise_model = noise_model
        return noise_model
