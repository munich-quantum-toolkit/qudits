from __future__ import annotations

import numpy as np


class Primitive:
    CUSTOM_PRIMITIVE = None

    @classmethod
    def set_class_variables(cls, primitive) -> None:
        cls.CUSTOM_PRIMITIVE = primitive


def reindex(ir, jc, num_col):
    return ir * num_col + jc


bound_1 = [0, np.pi]
bound_2 = [0, np.pi / 2]
bound_3 = [0, 2 * np.pi]
