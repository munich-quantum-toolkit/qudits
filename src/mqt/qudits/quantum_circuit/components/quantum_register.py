from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

if TYPE_CHECKING:
    RegisterMap = dict[str, list[Union[dict[int, int], list[int]]]]
    SiteMap = dict[tuple[str, int], tuple[int, int]]


class QuantumRegister:
    @classmethod
    def from_map(cls, sitemap: SiteMap) -> list[QuantumRegister]:
        registers_map: RegisterMap = {}

        for qreg_with_index, line_info in sitemap.items():
            reg_name, inreg_line_index = qreg_with_index
            extracted_local_line_indexing: int = line_info[0]
            dimensionality_extracted: int = line_info[1]
            if reg_name not in registers_map:
                registers_map[reg_name] = [
                    {inreg_line_index: extracted_local_line_indexing},
                    [dimensionality_extracted],
                ]
            else:
                mapping: dict[int, int] = cast("dict[int, int]", registers_map[reg_name][0])
                mapping[inreg_line_index] = line_info[0]
                dimensions_mapped: list[int] = cast("list[int]", registers_map[reg_name][1])
                dimensions_mapped.append(line_info[1])
        # print(registers_map)
        registers_from_qasm = []
        for label, data in registers_map.items():
            global_indexing_dict: dict[int, int] = cast("dict[int, int]", data[0])
            global_dimensions_list: list[int] = cast("list[int]", data[1])
            temp = QuantumRegister(label, len(global_indexing_dict), global_dimensions_list)
            temp.local_sitemap = global_indexing_dict
            registers_from_qasm.append(temp)
        # print(registers_from_qasm)
        return registers_from_qasm

    def __init__(self, name: str, size: int, dims: list[int] | None = None) -> None:
        self.label = name
        self.size: int = size
        self.dimensions: list[int] = size * [2] if dims is None else dims
        self.local_sitemap: dict[int, int] = {}

    def __qasm__(self) -> str:  # noqa: PLW3201
        string_dims = str(self.dimensions).replace(" ", "")
        return "qreg " + self.label + " [" + str(self.size) + "]" + string_dims + ";"

    def __getitem__(self, key: int | slice) -> int | list[int]:
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            return [self.local_sitemap[i] for i in range(start, stop)]

        return self.local_sitemap[key]
