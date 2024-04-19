from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

from .backends import MISim, TNSim
from .backends.fake_backends import FakeIonTraps2Six, FakeIonTraps2Trits

if TYPE_CHECKING:
    from .backends.backendv2 import Backend


class MQTQuditProvider:
    @property
    def version(self) -> int:
        return 0

    __backends: ClassVar[dict] = {
        "tnsim": TNSim,
        "misim": MISim,
        "faketraps2trits": FakeIonTraps2Trits,
        "faketraps2six": FakeIonTraps2Six,
    }

    def get_backend(self, name: str | None = None, **kwargs) -> Backend:
        """Return a single backend matching the specified filtering."""
        keys_with_pattern = None
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern = key
        return self.__backends[keys_with_pattern](**kwargs)

    def backends(self, name: str | None = None, **kwargs) -> list[Backend]:
        """Return a list of backends matching the specified filtering."""
        keys_with_pattern = []
        regex = re.compile(name)
        for key in self.__backends:
            if regex.search(key):
                keys_with_pattern.append(key)
        return keys_with_pattern

    def __eq__(self, other: object) -> bool:
        return type(self).__name__ == type(other).__name__

    def __hash__(self) -> int:
        return hash(type(self).__name__)
