from __future__ import annotations


class JobError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobTimeoutError(TimeoutError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
