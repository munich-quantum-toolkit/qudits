from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime

    from ...core import LevelGraph
    from ...quantum_circuit.gate import Gate
    from ..jobs import Job
    from ..qudit_provider import QuditProvider as Provider


class Backend(ABC):
    @property
    def version(self) -> int:
        return 2

    def __init__(
        self,
        provider: Provider | None = None,
        name: str | None = None,
        description: str | None = None,
        online_date: datetime | None = None,
        backend_version: str | None = None,
        **fields: Any,
    ) -> None:
        self._options = self._default_options()
        self._provider = provider

        if fields:
            # for field in fields:
            #    if field not in self._options.data:
            #        msg = f"Options field '{field}' is not valid for this backend"
            #        raise AttributeError(msg)
            self._options.update(fields)

        self.name = name
        self.description = description
        self.online_date = online_date
        self.backend_version = backend_version
        self._coupling_map = None
        self._energy_level_graphs = None

    @property
    def instructions(self) -> list[tuple[Gate, tuple[int]]]:
        return self.target.instructions

    @property
    def operations(self) -> list[Gate]:
        return list(self.target.operations)

    @property
    def operation_names(self) -> list[str]:
        return list(self.target.operation_names)

    # todo: this has to be defined properly
    target = Any

    @property
    def num_qudits(self) -> int:
        return self.target.num_qudits

    @property
    def energy_level_graphs(self) -> list[LevelGraph, LevelGraph]:
        raise NotImplementedError

    def _default_options(self):
        return {"shots": 50, "memory": False}

    def set_options(self, **fields) -> None:
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

    @abstractmethod
    def run(self, run_input, **options) -> Job:
        pass
