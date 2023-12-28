from typing import List


class QuantumRegister:
    @classmethod
    def from_map(cls, sitemap: dict) -> List["QuantumRegister"]:
        registers_map = {}

        for qreg_with_index, line_info in sitemap.items():
            reg_name, inreg_line_index = qreg_with_index
            if reg_name not in registers_map:
                registers_map[reg_name] = [{inreg_line_index: line_info[0]}, [line_info[1]]]
            else:
                registers_map[reg_name][0][inreg_line_index] = line_info[0]
                registers_map[reg_name][1].append(line_info[1])
        print(registers_map)
        registers_from_qasm = []
        for label, data in registers_map.items():
            temp = QuantumRegister(label, len(data[0]), data[1])
            temp.local_sitemap = data[0]
            registers_from_qasm.append(temp)
        print(registers_from_qasm)
        return registers_from_qasm

    def __init__(self, name, size, dims=None):
        self.label = name
        self.size = size
        self.dimensions = size * [2] if dims is None else dims
        self.local_sitemap = {}

    @property
    def __qasm__(self):
        string_dims = str(self.dimensions).replace(" ", "")
        return "qreg " + self.label + " [" + str(self.size) + "]" + string_dims + ";"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            return [self.local_sitemap[i] for i in range(start, stop)]

        return self.local_sitemap[key]
