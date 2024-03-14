from typing import List

import numpy as np


class JobResult:
    def __init__(self, state_vector: List[np.complex128], counts: List[int]):
        self.state_vector = state_vector
        self.counts = counts

    def get_counts(self) -> List[int]:
        return self.counts

    def get_state_vector(self) -> List[np.complex128]:
        return self.state_vector
