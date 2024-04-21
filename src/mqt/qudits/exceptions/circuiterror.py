from __future__ import annotations


class CircuitError(Exception):
    def __init__(self, message: str) -> None:
        print(message)
