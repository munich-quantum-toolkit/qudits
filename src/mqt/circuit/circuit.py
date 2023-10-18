from typing import ClassVar

from mqt.circuit.quantum_register import QuantumRegister
from mqt.interface.qasm import QASM


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

    def append(self, qreg: QuantumRegister):
        self.quantum_registers.append(qreg)
        self._num_qudits += qreg.size
        self._dimensions += qreg.dimensions

        num_lines_stored = len(self._sitemap)
        for i in range(qreg.size):
            qreg.local_sitemap[i] = num_lines_stored + i
            self._sitemap[(str(qreg.label), i)] = (num_lines_stored + i, qreg.dimensions[i])

    def csum(self, control: int, target: int):
        pass
        # self.instructions.append(CSum())

    def custom_unitary(self):
        pass

    def cx(self):
        pass

    def h(self):
        pass

    def ls(self):
        pass

    def ms(self):
        pass

    def pm(self):
        pass

    def r(self):
        pass

    def rz(self):
        pass

    def s(self):
        pass

    def z(self):
        pass

    def from_qasm_file(self, fname):
        self.reset()
        qasm_parser = QASM().parse_ditqasm2_file(fname)

        self._num_qudits = qasm_parser["n"]
        instructions = qasm_parser["instructions"]
        self._sitemap = qasm_parser["sitemap"]
        # self.number_gates = self._sitemap["n_gates"]

        qasm_set = self.get_qasm_set()
        for op in instructions:
            if op in qasm_set:
                gate_constructor_name = qasm_set[op["name"]]
                if hasattr(self, gate_constructor_name):
                    function = getattr(self, gate_constructor_name)
                    function(op["params"], op["qudits"])
                else:
                    msg = "the require gate is not available anymore."
                    raise NotImplementedError(msg)

    def to_qasm(self):
        text = ""
        text += "DITQASM 2.0;\n"
        for qreg in self.quantum_registers:
            text += qreg.__qasm__
        for op in self.instructions:
            text += op.__qasm__

    """
    def draw(self):
        custom_counter = 0

        for line in self.qreg:
            print("|0>---", end="")
            for gate in line:
                if isinstance(gate, Rz):
                    print("--[Rz" + str(gate.lev) + "(" + str(round(gate.theta, 2)) + ")]--", end="")

                elif isinstance(gate, R):
                    print("--[R" + str(gate.lev_a) + str(gate.lev_b) + "(" + str(round(gate.theta, 2)) + "," + str(
                        round(gate.phi, 2)) + ")]--", end="")

                elif isinstance(gate, Custom_Unitary):
                    print("--[C" + str(custom_counter) + "]--", end="")
                    custom_counter = custom_counter + 1

            print("---=||")
    """

    def reset(self):
        self.number_gates = 0
        self.instructions = []
        self.quantum_registers = []
        self._sitemap = {}
        self._num_cl = 0
        self._num_qudits = 0
        self._dimensions = []
