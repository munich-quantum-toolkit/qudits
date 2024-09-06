from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit.gate import Gate


class Primitive:
    CUSTOM_PRIMITIVE: Gate | None = None

    @classmethod
    def set_class_variables(cls, primitive: Gate | None = None) -> None:
        cls.CUSTOM_PRIMITIVE = primitive


def reindex(ir: int, jc: int, num_col: int) -> int:
    return ir * num_col + jc


bound_1 = (0, np.pi)
bound_2 = (0, np.pi / 2)
bound_3 = (0, 2 * np.pi)
