from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class JobResult:
    def __init__(self, state_vector: list[np.complex128], counts: list[int]) -> None:
        self.state_vector = state_vector
        self.counts = counts

    def get_counts(self) -> list[int]:
        return self.counts

    def get_state_vector(self) -> list[np.complex128]:
        return self.state_vector
