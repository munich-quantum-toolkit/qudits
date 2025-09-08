# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations


class JobError(RuntimeError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class JobTimeoutError(TimeoutError):
    def __init__(self, message: str) -> None:
        super().__init__(message)
