from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from mqt.circuit.components.instructions.gate_set.h import H
from mqt.circuit.components.registers.quantum_register import QuantumRegister
from mqt.circuit.qasm_interface.qasm import QASM

if TYPE_CHECKING:
    from mqt.circuit.components.instructions.gate_extensions.controls import ControlData


def add_gate_decorator(func):
    def gate_constructor(circ, *args):
        gate = func(circ, *args)
        circ.number_gates += 1
        circ.instructions.append(gate)
        return gate

    return gate_constructor


class QuantumCircuit:
    __qasm_to_gate_set_dict: ClassVar[dict] = {
        "csum": "csum",
        "custom": "custom_unitary",
        "cx": "cx",
        "gell": "gellman",
        "h": "h",
        "ls": "ls",
        "ms": "ms",
        "pm": "pm",
        "rxy": "r",
        "rdu": "randu",
        "rz": "rz",
        "s": "s",
        "x": "x",
        "z": "z",
    }

    def __init__(self, *args):
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
            # num_qudits: int, dimensions: List[int]|None, numcl: int
            self._num_qudits = args[0]
            self._dimensions = self._num_qudits * [2] if args[1] is None else args[1]
            self._num_cl = args[2]
        elif isinstance(args[0], QuantumRegister):
            # case 2
            # quantum register based construction
            register = args[0]
            self.append(register)

    @classmethod
    def get_qasm_set(cls):
        return cls.__qasm_to_gate_set_dict

    @property
    def num_qudits(self):
        return self._num_qudits

    @property
    def dimensions(self):
        return self._dimensions

    def reset(self):
        self.number_gates = 0
        self.instructions = []
        self.quantum_registers = []
        self._sitemap = {}
        self._num_cl = 0
        self._num_qudits = 0
        self._dimensions = []

    def append(self, qreg: QuantumRegister):
        self.quantum_registers.append(qreg)
        self._num_qudits += qreg.size
        self._dimensions += qreg.dimensions

        num_lines_stored = len(self._sitemap)
        for i in range(qreg.size):
            qreg.local_sitemap[i] = num_lines_stored + i
            self._sitemap[(str(qreg.label), i)] = (num_lines_stored + i, qreg.dimensions[i])

    @add_gate_decorator
    def csum(self, qudits: list[int]):
        pass

    def custom_unitary(self, qudits: list[int] | int):
        pass

    def cx(self, qudits: list[int], controls: ControlData | None = None):
        pass

    @add_gate_decorator
    def h(self, qudit: int, controls: ControlData | None = None):
        return H(self, "H" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    def ls(self, qudits: list[int], controls: ControlData | None = None):
        pass

    def ms(self, qudits: list[int], controls: ControlData | None = None):
        pass

    def pm(self, params, qudits: list[int] | int, controls: ControlData | None = None):
        pass

    def r(self, params, qudit: int, controls: ControlData | None = None):
        pass

    def rz(self, params, qudit: int, controls: ControlData | None = None):
        pass

    def s(self, qudit: int, controls: ControlData | None = None):
        pass

    def z(self, qudit: int, controls: ControlData | None = None):
        pass

    def from_qasm(self, qasm_prog):
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
                        function(op["params"], qudits_call)
                    else:
                        function(qudits_call)
                else:
                    msg = "the required gate is not available anymore."
                    raise NotImplementedError(msg)

    def to_qasm(self):
        text = ""
        text += "DITQASM 2.0;\n"
        for qreg in self.quantum_registers:
            text += qreg.__qasm__
        for op in self.instructions:
            text += op.__qasm__

    def draw(self):
        # TODO
        pass
