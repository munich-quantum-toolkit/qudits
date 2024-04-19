from __future__ import annotations


class JobError(Exception):
    def __init_(self, message) -> None:
        print(message)


class JobTimeoutError:
    def __init_(self, message) -> None:
        print(message)
