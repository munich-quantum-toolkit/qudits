import re
from typing import ClassVar, List, Optional

from mqt.simulation.provider.backends.engines.misim import MISim
from mqt.simulation.provider.backends.engines.tnsim import TNSim
from mqt.simulation.provider.provider import Provider


class MQTProvider(Provider):
    __backends: ClassVar[dict] = {"tnsim": TNSim, "misim": MISim}

    def get_backend(self, name: Optional[str] = None, **kwargs) -> "Backend":
        keys_with_pattern = None
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern = key
        return self.__backends[keys_with_pattern]()

    def backends(self, name: Optional[str] = None, **kwargs) -> List["Backend"]:
        keys_with_pattern = []
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern.append(key)
        return keys_with_pattern
