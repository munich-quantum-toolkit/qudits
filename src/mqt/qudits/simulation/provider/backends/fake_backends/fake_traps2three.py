from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

from mqt.qudits.core.structures.energy_level_graph.level_graph import LevelGraph
from mqt.qudits.qudit_circuits.components.instructions.instruction import Instruction
from mqt.qudits.simulation.provider.backend_properties.quditproperties import QuditProperties
from mqt.qudits.simulation.provider.backends.engines.tnsim import TNSim
from mqt.qudits.simulation.provider.noise_tools.noise import Noise, NoiseModel
from mqt.qudits.simulation.provider.provider import Provider


class FakeIonTraps2Trits(TNSim):
    @property
    def version(self):
        return 0

    def __init__(
        self,
        provider: Optional[Provider] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        online_date: Optional[datetime] = None,
        backend_version: Optional[str] = None,
        **fields,
    ):
        self._options = self._default_options()
        self._provider = provider

        if fields:
            # for field in fields:
            #    if field not in self._options.data:
            #        msg = f"Options field '{field}' is not valid for this backend"
            #        raise AttributeError(msg)
            self._options.update(fields)

        self.name = name

        self.name = "FakeTrap2"
        self.description = "A Fake backend of an ion trap qudit machine"
        self.author = "<Kevin Mato>"
        self.online_date = online_date
        self.backend_version = backend_version
        self._coupling_map = None
        self._energy_level_graphs = None

    @property
    def instructions(self) -> List[Tuple[Instruction, Tuple[int]]]:
        return self.target.instructions

    @property
    def operations(self) -> List[Instruction]:
        return list(self.target.operations)

    @property
    def operation_names(self) -> List[str]:
        return list(self.target.operation_names)

    @property
    def target(self):
        raise NotImplementedError

    @property
    def num_qudits(self) -> int:
        return self.target.num_qudits

    @property
    def coupling_map(self):
        raise NotImplementedError
        """
        if self._coupling_map is None:
            self._coupling_map = self.target.build_coupling_map()
        return self._coupling_map
        """

    @property
    def energy_level_graphs(self):
        e_graphs = []

        # declare the edges on the energy level graph between logic states .
        edges = [
            (1, 0, {"delta_m": 0, "sensitivity": 3}),
            (0, 2, {"delta_m": 0, "sensitivity": 3}),
        ]
        # name explicitly the logic states .
        nodes = [0, 1, 2]
        # declare physical levels in order of maping of the logic states just declared .
        # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
        nmap = [0, 1, 2]
        # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
        # calibrations of the operations. note: only the first is one counts in our current cost fucntion.
        graph_0 = LevelGraph(edges, nodes, nmap, [1])
        # declare the edges on the energy level graph between logic states .
        edges = [
            (1, 0, {"delta_m": 0, "sensitivity": 3}),
            (0, 2, {"delta_m": 0, "sensitivity": 3}),
        ]
        # name explicitly the logic states .
        nodes = [0, 1, 2]
        # declare physical levels in order of maping of the logic states just declared .
        # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
        nmap = [0, 1, 2]
        # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
        # calibrations of the operations. note: only the first is one counts in our current cost fucntion.
        graph_1 = LevelGraph(edges, nodes, nmap, [1])
        e_graphs.append(graph_0)
        e_graphs.append(graph_1)

        return e_graphs

    def __noise_model(self):
        """
        Noise model coded in plain sight, just for prototyping reasons
        :return: NoideModel
        """
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
        noise_model.add_recurrent_quantum_error_locally(local_error, ["csum"], [0])
        # Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_target(entangling_error_on_target, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_control(entangling_error_on_control, ["csum", "cx", "ls", "ms"])
        # Super noisy Entangling gates
        noise_model.add_nonlocal_quantum_error(entangling_error_extra, ["csum"])
        # Local Gates
        noise_model.add_quantum_error_locally(local_error, ["h", "rxy", "s", "x", "z"])
        noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])

        return noise_model

    @property
    def instruction_durations(self):
        raise NotImplementedError

    @property
    def max_circuits(self):
        raise NotImplementedError

    def _default_options(self):
        return {"shots": 1000, "memory": False, "noise_model": self.__noise_model()}

    @property
    def dt(self) -> Union[float, None]:
        raise NotImplementedError

    @property
    def dtm(self) -> float:
        raise NotImplementedError

    @property
    def meas_map(self) -> List[List[int]]:
        raise NotImplementedError

    @property
    def instruction_schedule_map(self):
        raise NotImplementedError

    def qudit_properties(self, qudit: Union[int, List[int]]) -> Union[QuditProperties, List[QuditProperties]]:
        raise NotImplementedError

    def drive_channel(self, qudit: int):
        raise NotImplementedError

    def measure_channel(self, qudit: int):
        raise NotImplementedError

    def acquire_channel(self, qudit: int):
        raise NotImplementedError

    def control_channel(self, qudits: Iterable[int]):
        raise NotImplementedError

    def set_options(self, **fields):
        for field in fields:
            if not hasattr(self._options, field):
                msg = f"Options field '{field}' is not valid for this backend"
                raise AttributeError(msg)
        self._options.update_options(**fields)

    @property
    def options(self):
        return self._options

    @property
    def provider(self):
        return self._provider
