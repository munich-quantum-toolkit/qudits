from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict, cast

from typing_extensions import Unpack

if TYPE_CHECKING:
    from ...core import LevelGraph
    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider
    from ..jobs import Job
    from ..noise_tools import NoiseModel


class Backend(ABC):
    class DefaultOptions(TypedDict, total=False):
        shots: int
        memory: bool
        noise_model: NoiseModel | None
        file_path: str | None
        file_name: str | None
        full_state_memory: bool

    def __init__(
        self,
        provider: MQTQuditProvider,
        name: str | None = None,
        description: str | None = None,
        **fields: Unpack[DefaultOptions],
    ) -> None:
        self._provider: MQTQuditProvider = provider
        self.name = name
        self.description: str | None = description
        self._energy_level_graphs: list[LevelGraph] = []
        self.noise_model: NoiseModel | None = None
        self.shots: int = 50
        self.memory: bool = False
        self.full_state_memory: bool = False
        self.file_path: str | None = None
        self.file_name: str | None = None

        self._options = self._default_options()
        if fields:
            self._options.update(fields)

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        raise NotImplementedError

    @staticmethod
    def _default_options() -> DefaultOptions:
        return {"shots": 50, "memory": False, "noise_model": None}

    def set_options(self, **fields: Unpack[Backend.DefaultOptions]) -> None:
        for field in fields:
            if not hasattr(self._options, field):
                msg = f"Options field '{field}' is not valid for this backend"
                raise AttributeError(msg)
        self._options.update(fields)

    @property
    def options(self) -> dict[str, Any]:
        return cast(dict[str, Any], self._options)

    @property
    def provider(self) -> MQTQuditProvider:
        return self._provider

    @abstractmethod
    def run(self, circuit: QuantumCircuit, **options: Unpack[DefaultOptions]) -> Job:
        pass

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> None:
        raise NotImplementedError
