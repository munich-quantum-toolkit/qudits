from __future__ import annotations


class NodeNotFoundException(Exception):
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)


class SequenceFoundException(Exception):
    def __init__(self, node_key=-1) -> None:
        self.last_node_id = node_key

    def __str__(self) -> str:
        return repr(self.last_node_id)


class RoutingException(Exception):
    def __init__(self) -> None:
        self.message = "ROUTING PROBLEM STUCK!"

    def __str__(self) -> str:
        return repr(self.message)


class FidelityReachException(Exception):
    def __init__(self, message="") -> None:
        self.message = message
