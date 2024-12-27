from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


class JobResultJSON(TypedDict):
    job_id: str  # required
    state_vector: NotRequired[list[complex]]  # optional
    counts: NotRequired[list[int]]  # optional


class JobResult:
    def __init__(
        self,
        job_id: str | JobResultJSON,
        state_vector: NDArray[np.complex128] | None = None,
        counts: Sequence[int] | None = None,
    ) -> None:
        # If first argument is a dict, treat it as JSON data
        if isinstance(job_id, dict):
            json_data = job_id
            self.job_id = json_data.get("job_id", "")
            # Convert list to numpy array for state vector
            state_vector_data = json_data.get("state_vector", [])
            self.state_vector = np.array(state_vector_data, dtype=np.complex128)
            self.counts = json_data.get("counts", [])
        elif counts is not None and state_vector is not None:
            # Traditional initialization with direct parameters
            self.job_id = job_id
            self.state_vector = state_vector
            self.counts = cast("list[int]", counts)

    def get_counts(self) -> Sequence[int]:
        return self.counts

    def get_state_vector(self) -> NDArray[np.complex128]:
        return self.state_vector

    def get_job_id(self) -> str:
        """Return the job ID associated with this result."""
        return self.job_id
