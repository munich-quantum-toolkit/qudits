class QuantumRegister:
    @classmethod
    def _from_map(cls, sitemap: dict):
        # TODO
        pass

    def __init__(self, name, size, dims=None):
        self.label = name
        self.size = size
        self.dimensions = size * [2] if dims is None else dims
        self.local_sitemap = {}

    @property
    def __qasm__(self):
        string_dims = str(self.dimensions).replace(" ", "")
        return "qreg " + self.label + " [" + str(self.size) + "]" + string_dims + ";"
