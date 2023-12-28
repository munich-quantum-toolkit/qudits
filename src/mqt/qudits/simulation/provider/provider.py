from abc import ABC, abstractmethod
from typing import List, Optional

from mqt.qudits.exceptions.backenderror import BackendNotFoundError


class Provider(ABC):
    """Base class for a Backend Provider."""

    @property
    def version(self):
        return 0

    @abstractmethod
    def get_backend(self, name: Optional[str] = None, **kwargs) -> "Backend":
        """Return a single backend matching the specified filtering."""
        backends = self.backends(name, **kwargs)
        if len(backends) > 1:
            msg = "More than one backend matches the criteria"
            raise BackendNotFoundError(msg)
        if not backends:
            msg = "No backend matches the criteria"
            raise BackendNotFoundError(msg)
        return backends[0]

    @abstractmethod
    def backends(self, name: Optional[str] = None, **kwargs) -> List["Backend"]:
        """Return a list of backends matching the specified filtering."""

    def __eq__(self, other):
        return type(self).__name__ == type(other).__name__
