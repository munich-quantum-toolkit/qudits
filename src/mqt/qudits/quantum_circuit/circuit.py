from __future__ import annotations

import copy
import locale
import warnings
from typing import TYPE_CHECKING

from .gates import (
    LS,
    MS,
    CEx,
    CSum,
    CustomMulti,
    CustomOne,
    CustomTwo,
    GellMann,
    H,
    Perm,
    R,
    RandU,
    Rh,
    Rz,
    S,
    VirtRz,
    X,
    Z,
)
from .qasm import QASM

if TYPE_CHECKING:
    import numpy as np

    from .gate import ControlData, Gate


class QuantumRegister:
    @classmethod
    def from_map(cls, sitemap: dict) -> list[QuantumRegister]:
        registers_map = {}

        for qreg_with_index, line_info in sitemap.items():
            reg_name, inreg_line_index = qreg_with_index
            if reg_name not in registers_map:
                registers_map[reg_name] = [{inreg_line_index: line_info[0]}, [line_info[1]]]
            else:
                registers_map[reg_name][0][inreg_line_index] = line_info[0]
                registers_map[reg_name][1].append(line_info[1])
        # print(registers_map)
        registers_from_qasm = []
        for label, data in registers_map.items():
            temp = QuantumRegister(label, len(data[0]), data[1])
            temp.local_sitemap = data[0]
            registers_from_qasm.append(temp)
        # print(registers_from_qasm)
        return registers_from_qasm

    def __init__(self, name, size, dims=None) -> None:
        self.label = name
        self.size = size
        self.dimensions = size * [2] if dims is None else dims
        self.local_sitemap = {}

    def __qasm__(self):
        string_dims = str(self.dimensions).replace(" ", "")
        return "qreg " + self.label + " [" + str(self.size) + "]" + string_dims + ";"

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            return [self.local_sitemap[i] for i in range(start, stop)]

        return self.local_sitemap[key]


def add_gate_decorator(func):
    def gate_constructor(circ, *args):
        gate = func(circ, *args)
        circ.number_gates += 1
        circ.instructions.append(gate)
        return gate

    return gate_constructor


class QuantumCircuit:
    qasm_to_gate_set_dict = {
        "csum": "csum",
        "cuone": "cu_one",
        "cutwo": "cu_two",
        "cumulti": "cu_multi",
        "cx": "cx",
        "gell": "gellmann",
        "h": "h",
        "ls": "ls",
        "ms": "ms",
        "pm": "pm",
        "rxy": "r",
        "rh": "rh",
        "rdu": "randu",
        "rz": "rz",
        "virtrz": "virtrz",
        "s": "s",
        "x": "x",
        "z": "z",
    }

    def __init__(self, *args) -> None:
        self.inverse_sitemap = {}
        self.number_gates = 0
        self.instructions = []
        self.quantum_registers = []
        self._sitemap = {}
        self._num_cl = 0
        self._num_qudits = 0
        self._dimensions = []

        if len(args) == 0:
            return
        if len(args) > 1:
            # case 1
            # num_qudits: int, dimensions_slice: List[int]|None, numcl: int
            num_qudits = args[0]
            dims = num_qudits * [2] if args[1] is None else args[1]
            self.append(QuantumRegister("q", num_qudits, dims))
            # self._num_cl = args[2]
        elif isinstance(args[0], QuantumRegister):
            # case 2
            # quantum register based construction
            register = args[0]
            self.append(register)

    @classmethod
    def get_qasm_set(cls):
        return cls.qasm_to_gate_set_dict

    @property
    def num_qudits(self):
        return self._num_qudits

    @property
    def dimensions(self):
        return self._dimensions

    def reset(self) -> None:
        self.number_gates = 0
        self.instructions = []
        self.quantum_registers = []
        self.inverse_sitemap = {}
        self._sitemap = {}
        self._num_cl = 0
        self._num_qudits = 0
        self._dimensions = []

    def copy(self):
        return copy.deepcopy(self)

    def append(self, qreg: QuantumRegister) -> None:
        self.quantum_registers.append(qreg)
        self._num_qudits += qreg.size
        self._dimensions += qreg.dimensions

        num_lines_stored = len(self._sitemap)
        for i in range(qreg.size):
            qreg.local_sitemap[i] = num_lines_stored + i
            self._sitemap[(str(qreg.label), i)] = (num_lines_stored + i, qreg.dimensions[i])
            self.inverse_sitemap[num_lines_stored + i] = (str(qreg.label), i)

    @add_gate_decorator
    def csum(self, qudits: list[int]):
        return CSum(
            self, "CSum" + str([self.dimensions[i] for i in qudits]), qudits, [self.dimensions[i] for i in qudits], None
        )

    @add_gate_decorator
    def cu_one(self, qudits: int, parameters: np.ndarray, controls: ControlData | None = None):
        return CustomOne(
            self, "CUo" + str(self.dimensions[qudits]), qudits, parameters, self.dimensions[qudits], controls
        )

    @add_gate_decorator
    def cu_two(self, qudits: int, parameters: np.ndarray, controls: ControlData | None = None):
        return CustomTwo(
            self,
            "CUt" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            controls,
        )

    @add_gate_decorator
    def cu_multi(self, qudits: int, parameters: np.ndarray, controls: ControlData | None = None):
        return CustomMulti(
            self,
            "CUm" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            controls,
        )

    @add_gate_decorator
    def cx(self, qudits: list[int], parameters: list | None = None):
        return CEx(
            self,
            "CEx" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def gellmann(self, qudit: int, parameters: list | None = None, controls: ControlData | None = None):
        warnings.warn("Using this matrix in a circuit will not allow simulation.", UserWarning)
        return GellMann(self, "Gell" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def h(self, qudit: int, controls: ControlData | None = None):
        return H(self, "H" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def rh(self, qudit: int, controls: ControlData | None = None):
        return Rh(self, "Rh" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def ls(self, qudits: list[int], parameters: list | None = None):
        return LS(
            self,
            "LS" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def ms(self, qudits: list[int], parameters: list | None = None):
        return MS(
            self,
            "MS" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def pm(self, qudits: list[int], parameters: list):
        return Perm(
            self,
            "Pm" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def r(self, qudit: int, parameters: list, controls: ControlData | None = None):
        return R(self, "R" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def randu(self, qudits: list[int]):
        return RandU(
            self, "RandU" + str([self.dimensions[i] for i in qudits]), qudits, [self.dimensions[i] for i in qudits]
        )

    @add_gate_decorator
    def rz(self, qudit: int, parameters: list, controls: ControlData | None = None):
        return Rz(self, "Rz" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    def virtrz(self, qudit: int, parameters: list, controls: ControlData | None = None):
        return VirtRz(self, "VirtRz" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def s(self, qudit: int, controls: ControlData | None = None):
        return S(self, "S" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def x(self, qudit: int, controls: ControlData | None = None):
        return X(self, "X" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def z(self, qudit: int, controls: ControlData | None = None):
        return Z(self, "Z" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    def replace_gate(self, gate_index: int, sequence: list[Gate]) -> None:
        self.instructions[gate_index : gate_index + 1] = sequence
        self.number_gates = (self.number_gates - 1) + len(sequence)

    def set_instructions(self, sequence: list[Gate]):
        self.instructions = []
        self.instructions += sequence
        self.number_gates = len(sequence)
        return self

    def from_qasm(self, qasm_prog) -> None:
        self.reset()
        qasm_parser = QASM().parse_ditqasm2_str(qasm_prog)
        instructions = qasm_parser["instructions"]
        temp_sitemap = qasm_parser["sitemap"]

        for qreg in QuantumRegister.from_map(temp_sitemap):
            self.append(qreg)

        qasm_set = self.get_qasm_set()

        for op in instructions:
            if op["name"] in qasm_set:
                gate_constructor_name = qasm_set[op["name"]]
                if hasattr(self, gate_constructor_name):
                    function = getattr(self, gate_constructor_name)

                    tuples_qudits = op["qudits"]
                    if not tuples_qudits:
                        msg = "Qudit parameter not applied"
                        raise IndexError(msg)
                    # Check if the list contains only one tuple
                    if len(tuples_qudits) == 1:
                        qudits_call = tuples_qudits[0][0]
                    # Extract the first element from each tuple and return as a list
                    else:
                        qudits_call = [t[0] for t in list(tuples_qudits)]
                    if op["params"]:
                        if op["controls"]:
                            function(qudits_call, op["params"], op["controls"])
                        else:
                            function(qudits_call, op["params"])
                    elif op["controls"]:
                        function(qudits_call, op["controls"])
                    else:
                        function(qudits_call)
                else:
                    msg = "the required gate_matrix is not available anymore."
                    raise NotImplementedError(msg)

    def to_qasm(self):
        text = ""
        text += "DITQASM 2.0;\n"
        for qreg in self.quantum_registers:
            text += qreg.__qasm__()
            text += "\n"
        text += f"creg meas[{len(self.dimensions)}];\n"

        for op in self.instructions:
            text += op.__qasm__()

        cregs_indeces = iter(list(range(len(self.dimensions))))
        for qreg in self.quantum_registers:
            for i in range(qreg.size):
                text += f"measure {qreg.label}[{i}] -> meas[{next(cregs_indeces)}];\n"

        return text

    def save_to_file(self, file_name, file_path="/"):
        """
        Save qasm into a file with the specified name and path.

        Args:
            text (str): The text to be saved into the file.
            file_name (str): The name of the file.
            file_path (str, optional): The path where the file will be saved. Defaults to "." (current directory).

        Returns:
            str: The full path of the saved file.
        """
        # Combine the file path and name to get the full file path
        full_file_path = f"{file_path}/{file_name}.qasm"

        # Write the text to the file
        with open(full_file_path, "w+", encoding=locale.getpreferredencoding(False)) as file:
            file.write(self.to_qasm())

        return full_file_path

    def load_from_file(self, file_path):
        """
        Load text from a file.

        Args:
            file_path (str): The path of the file to load.

        Returns:
            str: The text loaded from the file.
        """
        try:
            with open(file_path, encoding=locale.getpreferredencoding(False)) as file:
                text = file.read()
            return self.from_qasm(text)
        except FileNotFoundError:
            return None

    def draw(self) -> None:
        # TODO
        pass

    @property
    def gate_set(self) -> None:
        for _item in self.qasm_to_gate_set_dict.values():
            pass
