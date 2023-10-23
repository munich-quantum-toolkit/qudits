from abc import ABC, abstractmethod


class Instruction(ABC):
    @abstractmethod
    def __init__(self, name):
        pass
