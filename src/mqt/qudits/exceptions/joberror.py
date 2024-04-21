from __future__ import annotations


class JobError(Exception):
    def __init__(self, message: str) -> None:
        print(message)


class JobTimeoutError:
    def __init__(self, message: str) -> None:
        print(message)
