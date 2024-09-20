from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Unpack

# from pyseq.mqt_qudits_runner.sequence_runner import quantum_circuit_runner
from ...core import LevelGraph
from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider
    from ..noise_tools import NoiseModel


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

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        if len(self._energy_level_graphs) == 0:
            e_graphs: list[LevelGraph] = []
            # declare the edges on the energy level graph between logic states .
            edges = [
                (0, 1, {"delta_m": 0, "sensitivity": 3, "carrier": 0}),
                (1, 2, {"delta_m": 0, "sensitivity": 4, "carrier": 1}),
            ]
            # name explicitly the logic states .
            nodes = [0, 1, 2]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap = [0, 1, 2]
            graph_0 = LevelGraph(edges, nodes, nmap, [1])

            edges_1 = [
                (0, 1, {"delta_m": 0, "sensitivity": 3, "carrier": 0}),
                (1, 2, {"delta_m": 0, "sensitivity": 4, "carrier": 1}),
            ]
            # name explicitly the logic states .
            nodes_1 = [0, 1, 2]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap_1 = [0, 1, 2]

            graph_1 = LevelGraph(edges_1, nodes_1, nmap_1, [1])

            e_graphs.extend((graph_0, graph_1))

            self._energy_level_graphs = e_graphs
        return self._energy_level_graphs

    def edge_to_carrier(self, leva: int, levb: int, graph_index: int) -> int:
        e_graph = self.energy_level_graphs[graph_index]
        edge_data: dict[str, int] = e_graph.get_edge_data(leva, levb)
        return edge_data["carrier"]

    def __noise_model(self) -> NoiseModel | None:
        return self.noise_model

    def run(self, circuit: QuantumCircuit, **options: Unpack[Backend.DefaultOptions]) -> Job:
        job = Job(self)

        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 50)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)
        self.file_path = self._options.get("file_path", None)
        self.file_name = self._options.get("file_name", None)

        assert self.shots >= 50, "Number of shots should be above 50"
        self.execute(circuit)
        job.set_result(JobResult(state_vector=np.array([]), counts=self.outcome))

        return job

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> None:
        _ = noise_model  # Silences the unused argument warning
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions

        # quantum_circuit_runner(self.circ_operations)

        self.outcome = []  # quantum_circuit_runner(metadata, self.system_sizes)
