from __future__ import annotations


class BackendNotFoundError(Exception):
    def __init__(self, message: str) -> None:
        print(message)
