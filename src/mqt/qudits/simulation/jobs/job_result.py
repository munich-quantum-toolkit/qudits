from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class JobResult:
    def __init__(self, state_vector: NDArray[np.complex128], counts: Sequence[int]) -> None:
        self.state_vector = state_vector
        self.counts = counts

    def get_counts(self) -> Sequence[int]:
        return self.counts

    def get_state_vector(self) -> NDArray[np.complex128]:
        return self.state_vector
