from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, ClassVar

from .backends import MISim, TNSim
from .backends.fake_backends import FakeIonTraps2Six, FakeIonTraps2Trits, FakeIonTraps3Six

if TYPE_CHECKING:
    from .backends.backendv2 import Backend


class MQTQuditProvider:
    @property
    def version(self) -> int:
        return 0

    __backends: ClassVar[dict[str, type[Backend]]] = {
        "tnsim": TNSim,
        "misim": MISim,
        "faketraps2trits": FakeIonTraps2Trits,
        "faketraps2six": FakeIonTraps2Six,
        "faketraps3six": FakeIonTraps3Six,
    }

    def get_backend(self, name: str | None = None, **kwargs: dict[str, Any]) -> Backend:
        """Return a single backend matching the specified filtering."""
        if name is None:
            msg = "Backend name must be provided"
            raise ValueError(msg)

        regex = re.compile(name)
        matching_backends = [key for key in self.__backends if regex.search(key)]

        if not matching_backends:
            msg = f"No backend found matching '{name}'"
            raise ValueError(msg)
        if len(matching_backends) > 1:
            msg = f"Multiple backends found matching '{name}': {matching_backends}"
            raise ValueError(msg)

        return self.__backends[matching_backends[0]](provider=self, **kwargs)  # type: ignore[arg-type]

    def backends(self, name: str | None = None) -> list[str]:
        """Return a list of backend names matching the specified filtering."""
        if name is None:
            return list(self.__backends.keys())

        regex = re.compile(name)
        return [key for key in self.__backends if regex.search(key)]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MQTQuditProvider):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(MQTQuditProvider)
