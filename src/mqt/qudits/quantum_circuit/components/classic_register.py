from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    ClSitemap = dict[tuple[str, int], tuple[int]]


class ClassicRegister:
    @classmethod
    def from_map(cls, sitemap: dict[tuple[str, int], Any]) -> list[ClassicRegister]:
        registers_map = {}

        for creg_with_index, extracted_local_line_indexing in sitemap.items():
            reg_name, inreg_line_index = creg_with_index
            if reg_name not in registers_map:
                registers_map[reg_name] = [{inreg_line_index: creg_with_index[0]}, extracted_local_line_indexing]
            else:
                registers_map[reg_name][0][inreg_line_index] = extracted_local_line_indexing

        registers_from_qasm = []
        for label, data in registers_map.items():
            global_indexing_dict: dict[int, int] = cast(dict[int, int], data[0])
            temp = ClassicRegister(label, len(global_indexing_dict))
            temp.local_sitemap = global_indexing_dict
            registers_from_qasm.append(temp)

        return registers_from_qasm

    def __init__(self, name: str, size: int) -> None:
        self.label: str = name
        self.size: int = size
        self.local_sitemap: dict[int, int] = {}

    def __qasm__(self) -> str:  # noqa: PLW3201
        return "creg " + self.label + " [" + str(self.size) + "]" + ";"

    def __getitem__(self, key: int | slice) -> int | list[int]:
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            return [self.local_sitemap[i] for i in range(start, stop)]

        return self.local_sitemap[key]


__all__ = ["ClSitemap", "ClassicRegister"]
