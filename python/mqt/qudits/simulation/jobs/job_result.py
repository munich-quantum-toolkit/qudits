# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray


class JobResult:
    def __init__(
        self,
        state_vector: NDArray[np.complex128] | None,
        counts: Sequence[int],
        density_matrix: NDArray[np.complex128] | None = None,
    ) -> None:
        self.state_vector = state_vector
        self.counts = counts
        self.density_matrix = density_matrix

    def get_counts(self) -> Sequence[int]:
        return self.counts

    def get_state_vector(self) -> NDArray[np.complex128]:
        if self.state_vector is None:
            msg = "No state vector available — use get_density_matrix() for multi-shot noisy results."
            raise ValueError(msg)
        return self.state_vector

    def get_density_matrix(self) -> NDArray[np.complex128] | None:
        return self.density_matrix
