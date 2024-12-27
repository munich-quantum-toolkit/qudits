from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

from ....core import LevelGraph
from ...noise_tools import Noise, NoiseModel, SubspaceNoise
from ..tnsim import TNSim

if TYPE_CHECKING:
    from ... import MQTQuditProvider
    from ..backendv2 import Backend


class FakeIonTraps8Seven(TNSim):
    @property
    def version(self) -> int:
        return 0

    def __init__(
        self,
        provider: MQTQuditProvider,
        **fields: Unpack[Backend.DefaultOptions],
    ) -> None:
        super().__init__(
            provider=provider,
            name="FakeTrap8Seven",
            description="A Fake backend of an ion trap qudit machine with 8 ions and 7 levels each",
            **fields,
        )
        self.options["noise_model"] = self.__noise_model()
        self.author = "<Kevin Mato>"
        self._energy_level_graphs: list[LevelGraph] = []

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        if len(self._energy_level_graphs) == 0:
            e_graphs: list[LevelGraph] = []

            # Create graphs for all 8 ions
            for i in range(8):
                # declare the edges on the energy level graph between logic states
                # we have to fake a direct connection between 0 and 1 to make transpilations
                # physically compatible also for small systems
                edges = [
                    (0, 1, {"delta_m": 0, "sensitivity": 4}),
                    (2, 0, {"delta_m": 0, "sensitivity": 3}),
                    # (3, 0, {"delta_m": 0, "sensitivity": 3}),
                    (4, 0, {"delta_m": 0, "sensitivity": 3}),
                    (5, 0, {"delta_m": 0, "sensitivity": 4}),
                    (6, 0, {"delta_m": 0, "sensitivity": 4}),
                    (1, 2, {"delta_m": 0, "sensitivity": 4}),
                    (1, 3, {"delta_m": 0, "sensitivity": 3}),
                    (1, 4, {"delta_m": 0, "sensitivity": 3}),
                    (1, 5, {"delta_m": 0, "sensitivity": 4}),
                    (1, 6, {"delta_m": 0, "sensitivity": 4}),
                ]

                # name explicitly the logic states (0 through 6 for seven levels)
                nodes = list(range(7))

                # Different mappings for different ions
                if i % 3 == 0:
                    nmap = [1, 0, 2, 3, 4, 5, 6]  # Similar to graph_0 in original
                elif i % 3 == 1:
                    nmap = [2, 1, 0, 3, 4, 5, 6]  # Similar to graph_1 in original
                else:
                    nmap = [0, 2, 1, 3, 4, 5, 6]  # Similar to graph_2 in original

                # Construct the qudit energy level graph
                graph = LevelGraph(edges, nodes, nmap, [1])
                e_graphs.append(graph)

            self._energy_level_graphs = e_graphs
        return self._energy_level_graphs

    def __noise_model(self) -> NoiseModel:
        # Depolarizing and Dephasing quantum errors
        basic_error = Noise(probability_depolarizing=0.005, probability_dephasing=0.005)
        basic_subspace_dynamic_error = SubspaceNoise(
            probability_depolarizing=2e-4, probability_dephasing=2e-4, levels=[]
        )

        basic_subspace_dynamic_error_rz = SubspaceNoise(
            probability_depolarizing=6e-4, probability_dephasing=4e-4, levels=[]
        )
        subspace_error_01 = SubspaceNoise(probability_depolarizing=0.005, probability_dephasing=0.005, levels=(0, 1))
        subspace_error_01_cex = SubspaceNoise(
            probability_depolarizing=0.010, probability_dephasing=0.010, levels=(0, 1)
        )

        # Add errors to noise_tools model
        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate_matrix
        noise_model.add_nonlocal_quantum_error(basic_error, ["csum", "ls"])
        noise_model.add_quantum_error_locally(
            basic_error, ["cuone", "cutwo", "cumulti", "h", "perm", "rdu", "s", "x", "z"]
        )

        # Physical gates
        noise_model.add_nonlocal_quantum_error(subspace_error_01, ["ms"])
        noise_model.add_nonlocal_quantum_error(subspace_error_01_cex, ["cx"])
        noise_model.add_quantum_error_locally(basic_subspace_dynamic_error, ["rxy", "rh"])
        noise_model.add_quantum_error_locally(basic_subspace_dynamic_error_rz, ["rz"])
        self.noise_model = noise_model
        return noise_model
