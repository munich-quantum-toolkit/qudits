from mqt.interface.qasm import QASM


class QuantumCircuit:
    def __init__(self, num_qudits, sizes=None, numcl=0):
        self._sitemap = None
        self._num_qudits = num_qudits
        self._sizes = num_qudits * [2] if sizes is None else sizes
        self._num_cl = numcl
        self.instructions = []

    def csum(self, control, target):
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
        qasm_parser = QASM().parse_ditqasm2_file(fname)
        self._num_qudits = qasm_parser["n"]
        self.instructions = qasm_parser["instructions"]
        self._sitemap = qasm_parser["sitemap"]

    def to_qasm(self):
        pass

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
