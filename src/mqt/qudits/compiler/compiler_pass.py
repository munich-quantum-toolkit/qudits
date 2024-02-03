from abc import ABC, abstractmethod


class CompilerPass(ABC):
    def __init__(self, backend, **kwargs):
        self.backend = backend

    @abstractmethod
    def transpile(self, circuit):
        pass
