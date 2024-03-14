import re
from typing import ClassVar, List, Optional

from mqt.qudits.simulation.provider.backends.engines.misim import MISim
from mqt.qudits.simulation.provider.backends.engines.tnsim import TNSim
from mqt.qudits.simulation.provider.backends.fake_backends.fake_traps2three import FakeIonTraps2Trits
from mqt.qudits.simulation.provider.backends.fake_backends.fake_traps2six import FakeIonTraps2Six
from mqt.qudits.simulation.provider.provider import Provider


class MQTQuditProvider(Provider):
    __backends: ClassVar[dict] = {"tnsim":           TNSim,
                                  "misim":           MISim,
                                  "faketraps2trits": FakeIonTraps2Trits,
                                  "faketraps2six": FakeIonTraps2Six }

    def get_backend(self, name: Optional[str] = None, **kwargs) -> "Backend":
        keys_with_pattern = None
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern = key
        return self.__backends[keys_with_pattern](**kwargs)

    def backends(self, name: Optional[str] = None, **kwargs) -> List["Backend"]:
        keys_with_pattern = []
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern.append(key)
        return keys_with_pattern
