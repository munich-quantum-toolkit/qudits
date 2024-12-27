from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

from mqt.qudits.simulation.noise_tools.noise import Noise, SubspaceNoise

from ....core import LevelGraph
from ...noise_tools import NoiseModel
from ..tnsim import TNSim

if TYPE_CHECKING:
    from ... import MQTQuditProvider
    from ..backendv2 import Backend


class FakeIonTraps2Trits(TNSim):
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
            name="FakeTrap2",
            description="A Fake backend of an ion trap qudit machine",
            **fields,
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
                (1, 0, {"delta_m": 0, "sensitivity": 3}),
                (0, 2, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes = [0, 1, 2]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap = [1, 2, 0]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_0 = LevelGraph(edges, nodes, nmap, [1])
            # declare the edges on the energy level graph between logic states .
            edges = [
                (1, 0, {"delta_m": 0, "sensitivity": 3}),
                (0, 2, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes = [0, 1, 2]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap = [0, 1, 2]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_1 = LevelGraph(edges, nodes, nmap, [1])
            e_graphs.extend((graph_0, graph_1))

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
