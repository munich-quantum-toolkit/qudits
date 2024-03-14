from typing import Iterable, List, Union

import numpy as np
from mqt.misim import pymisim as misim
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
from mqt.qudits.simulation.provider.backend_properties.quditproperties import QuditProperties
from mqt.qudits.simulation.provider.backends.backendv2 import Backend
from mqt.qudits.simulation.provider.backends.stocastic_components.stocastic_sim import stocastic_simulation_misim
from mqt.qudits.simulation.provider.jobs.job import Job
from mqt.qudits.simulation.provider.jobs.job_result.job_result import JobResult

from mqt.qudits.simulation.provider.noise_tools.noise import NoiseModel


class MISim(Backend):
    @property
    def target(self):
        raise NotImplementedError

    @property
    def max_circuits(self):
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

    def run(self, circuit: QuantumCircuit, **options):
        job = Job(self)

        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 1 if self.noise_model is None else 1000)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)
        self.file_path = self._options.get("file_path", None)
        self.file_name = self._options.get("file_name", None)

        if self.noise_model is not None:
            assert self.shots >= 1000, "Number of shots should be above 1000"
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=stocastic_simulation_misim(self, circuit)))
        else:
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=None))

        return job

    def execute(self, circuit: QuantumCircuit, noise_model=None):
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions
        if noise_model is None:
            noise_model = NoiseModel()
        result = misim.state_vector_simulation(circuit, noise_model)
        state = np.array(result)
        state = state.reshape(1, -1)
        return state

    def __init__(self, **fields):
        self.system_sizes = None
        self.circ_operations = None
        super().__init__(**fields)

