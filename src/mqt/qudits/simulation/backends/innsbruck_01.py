from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

# from pyseq.mqt_qudits_runner.sequence_runner import quantum_circuit_runner
from ...core import LevelGraph
from ..jobs import Job
from ..jobs.client_api import APIClient
from ..noise_tools import Noise, NoiseModel, SubspaceNoise
from .backendv2 import Backend

if TYPE_CHECKING:
    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider


class Innsbruck01(Backend):
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
            name="Innsbruck01",
            description="Interface to the Innsbruck machine 01. Interface prototype with MVP.",
            **fields,
        )
        self.outcome: list[int] = []
        self.options["noise_model"] = self.__noise_model()
        self.author = "<Kevin Mato>"
        self._energy_level_graphs: list[LevelGraph] = []
        self._api_client = APIClient()

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

    def edge_to_carrier(self, leva: int, levb: int, graph_index: int) -> int:
        e_graph = self.energy_level_graphs[graph_index]
        edge_data: dict[str, int] = e_graph.get_edge_data(leva, levb)
        return edge_data["carrier"]

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
        noise_model = NoiseModel()
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

    def run(self, circuit: QuantumCircuit, **options: Unpack[Backend.DefaultOptions]) -> Job:
        Job(self)

        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 50)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)
        self.file_path = self._options.get("file_path", None)
        self.file_name = self._options.get("file_name", None)

        job_id = self._api_client.submit_job(circuit, self.shots, self.energy_level_graphs)
        return Job(self, job_id, self._api_client)

    def close(self) -> None:
        self._api_client.close()

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> None:
        pass
