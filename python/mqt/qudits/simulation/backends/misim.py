from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Unpack

from ..._qudits.misim import state_vector_simulation
from ..jobs import Job, JobResult
from ..noise_tools import NoiseModel
from .backendv2 import Backend
from .stochastic_sim import stochastic_simulation

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider


class MISim(Backend):
    def __init__(
        self,
        provider: MQTQuditProvider,
        name: str | None = None,
        description: str | None = None,
        **fields: Unpack[Backend.DefaultOptions],
    ) -> None:
        super().__init__(provider, name=name, description=description, **fields)

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

        if self.noise_model is not None:
            assert self.shots >= 50, "Number of shots should be above 50"
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=stochastic_simulation(self, circuit)))
        else:
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=[]))

        return job

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> NDArray[np.complex128]:
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions
        if noise_model is None:
            noise_model = NoiseModel()
        result = state_vector_simulation(circuit, noise_model)
        state = np.array(result)
        state_size = reduce(operator.mul, self.system_sizes, 1)
        # Reverse the dimensions of the circuit and reshape the state array
        reversed_dimensions = list(reversed(circuit.dimensions))
        state = state.reshape(reversed_dimensions)

        # Reverse the order of the axes for the transpose operation
        axes_order = list(reversed(list(range(len(circuit.dimensions)))))

        # Transpose the state array
        state = np.transpose(state, axes_order)
        return state.reshape((1, state_size))
