# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations


class NodeNotFoundError(ValueError):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self) -> str:
        return repr(self.message)


class SequenceFoundError(RuntimeError):
    def __init__(self, node_key: int = -1) -> None:
        self.last_node_id = node_key

    def __str__(self) -> str:
        return repr(self.last_node_id)


class RoutingError(RuntimeError):
    def __init__(self) -> None:
        self.message = "ROUTING PROBLEM STUCK!"

    def __str__(self) -> str:
        return repr(self.message)


class FidelityReachError(ValueError):
    def __init__(self, message: str = "") -> None:
        self.message = message
