from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, List, Optional, Tuple, Union

from mqt.circuit.components.instructions.instruction import Instruction
from mqt.simulation.provider.backend_properties.quditproperties import QuditProperties
from mqt.simulation.provider.provider import Provider


class Backend(ABC):
    @property
    def version(self):
        return 2

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
            for field in fields:
                if field not in self._options.data:
                    msg = f"Options field '{field}' is not valid for this backend"
                    raise AttributeError(msg)
            self._options.update_config(**fields)

        self.name = name
        self.description = description
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
    @abstractmethod
    def target(self):
        pass

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
        raise NotImplementedError
        """
        if self._energy_level_graphs is None:
            self._energy_level_graphs = self.target.build_energy_level_graphs()
        return self._energy_level_graphs
        """

    @property
    def instruction_durations(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def max_circuits(self):
        pass

    @classmethod
    @abstractmethod
    def _default_options(cls):
        pass

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

    @abstractmethod
    def qudit_properties(self, qudit: Union[int, List[int]]) -> Union[QuditProperties, List[QuditProperties]]:
        raise NotImplementedError

    @abstractmethod
    def drive_channel(self, qudit: int):
        raise NotImplementedError

    @abstractmethod
    def measure_channel(self, qudit: int):
        raise NotImplementedError

    @abstractmethod
    def acquire_channel(self, qudit: int):
        raise NotImplementedError

    @abstractmethod
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

    @abstractmethod
    def run(self, run_input, **options):
        pass
