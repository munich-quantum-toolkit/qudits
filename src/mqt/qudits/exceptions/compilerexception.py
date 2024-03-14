class NodeNotFoundException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class SequenceFoundException(Exception):
    def __init__(self, node_key=-1):
        self.last_node_id = node_key

    def __str__(self):
        return repr(self.last_node_id)


class RoutingException(Exception):
    def __init__(self):
        self.message = "ROUTING PROBLEM STUCK!"

    def __str__(self):
        return repr(self.message)
