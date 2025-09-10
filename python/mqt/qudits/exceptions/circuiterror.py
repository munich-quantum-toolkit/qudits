# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations


class CircuitError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidQuditDimensionError(ValueError):
    """Raised when the qudit dimension is invalid for the S gate."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ShapeMismatchError(ValueError):
    """Raised when input arrays have mismatched shapes."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
