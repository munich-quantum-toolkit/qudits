from __future__ import annotations


class ClassicRegister:
    @classmethod
    def from_map(cls, sitemap: dict) -> list[ClassicRegister]:
        registers_map = {}

        for creg_with_index, line_info in sitemap.items():
            reg_name, inreg_line_index = creg_with_index
            if reg_name not in registers_map:
                registers_map[reg_name] = [{inreg_line_index: creg_with_index[0]}, line_info]
            else:
                registers_map[reg_name][0][inreg_line_index] = line_info

        registers_from_qasm = []
        for label, data in registers_map.items():
            temp = ClassicRegister(label, len(data[0]))
            temp.local_sitemap = data[0]
            registers_from_qasm.append(temp)

        return registers_from_qasm

    def __init__(self, name, size) -> None:
        """
        Class for classical register memories
        """
        self.label = name
        self.size = size
        self.local_sitemap = {}

    def __qasm__(self):
        return "creg " + self.label + " [" + str(self.size) + "]" + ";"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            return [self.local_sitemap[i] for i in range(start, stop)]

        return self.local_sitemap[key]
