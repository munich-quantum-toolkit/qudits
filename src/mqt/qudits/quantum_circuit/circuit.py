from __future__ import annotations

import copy
import locale
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import numpy as np

from .components import ClassicRegister, QuantumRegister
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
    NoiseX,
    NoiseY,
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
    from collections.abc import Callable, Sequence

    from numpy.typing import ArrayLike, NDArray

    from .components.extensions.controls import ControlData
    from .components.quantum_register import SiteMap
    from .gate import Gate, Parameter

    InverseSitemap = dict[int, tuple[str, int]]
    from .components.classic_register import ClSitemap


def is_not_none_or_empty(variable: Parameter) -> bool:
    if variable is None:
        return False
    if isinstance(variable, np.ndarray):
        return bool(variable.size > 0)
    return bool(len(variable) > 0)


G = TypeVar("G", bound="Gate")


def add_gate_decorator(func: Callable[..., G]) -> Callable[..., G]:
    def gate_constructor(circ: QuantumCircuit, *args: Any) -> G:  # noqa: ANN401
        gate = func(circ, *args)
        circ.number_gates += 1
        circ.instructions.append(gate)
        return gate

    return gate_constructor


class QuantumCircuit:
    qasm_to_gate_set_dict: ClassVar = {
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
        "noisex": "noisex",
        "noisey": "noisey",
        "noisez": "noisez",
    }

    def __init__(self, *args: int | QuantumRegister | list[int] | None) -> None:
        self.cl_inverse_sitemap: InverseSitemap = {}
        self.inverse_sitemap: InverseSitemap = {}
        self.number_gates: int = 0
        self.instructions: list[Gate] = []
        self.quantum_registers: list[QuantumRegister] = []
        self.classic_registers: list[ClassicRegister] = []
        self.sitemap: SiteMap = {}
        self.classic_site_map: ClSitemap = {}
        self.num_cl: int = 0
        self._num_qudits: int = 0
        self._dimensions: list[int] = []
        self.final_mappings: list[list[int]] | None = None
        self.initial_mappings: list[list[int]] | None = None
        self.path_save: str | None = None

        if len(args) == 0:
            return
        if len(args) > 1:
            # case 1
            # num_qudits: int, dimensions_slice: List[int]|None, numcl: int
            num_qudits: int = cast("int", args[0])
            dims: list[int] = cast("list[int]", num_qudits * [2] if args[1] is None else args[1])
            self.append(QuantumRegister("q", num_qudits, dims))
            # self.num_cl = args[2]
        elif isinstance(args[0], QuantumRegister):
            # case 2
            # quantum register based construction
            register = args[0]
            self.append(register)

    @classmethod
    def get_qasm_set(cls) -> dict[str, str]:
        return cast("dict[str, str]", cls.qasm_to_gate_set_dict)

    @property
    def dimensions(self) -> list[int]:
        return self._dimensions

    @property
    def num_qudits(self) -> int:
        return self._num_qudits

    @num_qudits.setter
    def num_qudits(self, value: int) -> None:
        self._num_qudits = value

    def reset(self) -> None:
        self.cl_inverse_sitemap = {}
        self.inverse_sitemap = {}
        self.number_gates = 0
        self.instructions = []
        self.quantum_registers = []
        self.classic_registers = []
        self.sitemap = {}
        self.classic_site_map = {}
        self.num_cl = 0
        self.num_qudits = 0
        self._dimensions = []
        self.path_save = None

    def copy(self) -> QuantumCircuit:
        return copy.deepcopy(self)

    def append(self, qreg: QuantumRegister) -> None:
        self.quantum_registers.append(qreg)
        self.num_qudits += qreg.size
        self._dimensions += qreg.dimensions

        num_lines_stored = len(self.sitemap)
        try:
            for i in range(qreg.size):
                qreg.local_sitemap[i] = num_lines_stored + i
                self.sitemap[str(qreg.label), i] = (num_lines_stored + i, qreg.dimensions[i])
                self.inverse_sitemap[num_lines_stored + i] = (str(qreg.label), i)
        except IndexError:
            msg = "Check your Quantum Register to have the right number of lines and number of dimensions"
            raise IndexError(msg) from None

    def append_classic(self, creg: ClassicRegister) -> None:
        self.classic_registers.append(creg)
        self.num_cl += creg.size

        num_lines_stored = len(self.classic_site_map)
        for i in range(creg.size):
            creg.local_sitemap[i] = num_lines_stored + i
            self.classic_site_map[str(creg.label), i] = (num_lines_stored + i,)
            self.cl_inverse_sitemap[num_lines_stored + i] = (str(creg.label), i)

    @add_gate_decorator
    def csum(self, qudits: list[int]) -> CSum:
        return CSum(
            self, "CSum" + str([self.dimensions[i] for i in qudits]), qudits, [self.dimensions[i] for i in qudits], None
        )

    @add_gate_decorator
    def cu_one(self, qudits: int, parameters: NDArray, controls: ControlData | None = None) -> CustomOne:
        return CustomOne(
            self, "CUo" + str(self.dimensions[qudits]), qudits, parameters, self.dimensions[qudits], controls
        )

    @add_gate_decorator
    def cu_two(self, qudits: list[int], parameters: NDArray, controls: ControlData | None = None) -> CustomTwo:
        return CustomTwo(
            self,
            "CUt" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            controls,
        )

    @add_gate_decorator
    def cu_multi(self, qudits: list[int], parameters: NDArray, controls: ControlData | None = None) -> CustomMulti:
        return CustomMulti(
            self,
            "CUm" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            controls,
        )

    @add_gate_decorator
    def cx(self, qudits: list[int], parameters: list[int | float] | None = None) -> CEx:
        return CEx(
            self,
            "CEx" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    # @add_gate_decorator # decide to make it usable for computations but only for constructions
    def gellmann(self, qudit: int, parameters: list[int | str], controls: ControlData | None = None) -> GellMann:
        # warnings.warn("Using this matrix in a circuit will not allow simulation.", UserWarning)
        return GellMann(self, "Gell" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def h(self, qudit: int, controls: ControlData | None = None) -> H:
        return H(self, "H" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def rh(self, qudit: int, parameters: list[int], controls: ControlData | None = None) -> Rh:
        return Rh(self, "Rh" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def ls(self, qudits: list[int], parameters: list[float]) -> LS:
        return LS(
            self,
            "LS" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def ms(self, qudits: list[int], parameters: list[float]) -> MS:
        return MS(
            self,
            "MS" + str([self.dimensions[i] for i in qudits]),
            qudits,
            parameters,
            [self.dimensions[i] for i in qudits],
            None,
        )

    @add_gate_decorator
    def pm(self, qudits: int, parameters: list[int]) -> Perm:
        return Perm(self, "Pm" + str(self.dimensions[qudits]), qudits, parameters, self.dimensions[qudits], None)

    @add_gate_decorator
    def r(self, qudit: int, parameters: list[int | float], controls: ControlData | None = None) -> R:
        return R(self, "R" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def randu(self, qudits: list[int]) -> RandU:
        return RandU(
            self, "RandU" + str([self.dimensions[i] for i in qudits]), qudits, [self.dimensions[i] for i in qudits]
        )

    @add_gate_decorator
    def rz(self, qudit: int, parameters: list[int | float], controls: ControlData | None = None) -> Rz:
        return Rz(self, "Rz" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def noisex(self, qudit: int, parameters: list[int], controls: ControlData | None = None) -> NoiseX:
        return NoiseX(self, "NoiseX" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def noisey(self, qudit: int, parameters: list[int], controls: ControlData | None = None) -> NoiseY:
        return NoiseY(self, "NoiseY" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def noisez(self, qudit: int, level: int, controls: ControlData | None = None) -> VirtRz:
        return VirtRz(
            self, "NoiseZ" + str(self.dimensions[qudit]), qudit, [level, np.pi], self.dimensions[qudit], controls
        )

    @add_gate_decorator
    def virtrz(self, qudit: int, parameters: list[int | float], controls: ControlData | None = None) -> VirtRz:
        return VirtRz(self, "VirtRz" + str(self.dimensions[qudit]), qudit, parameters, self.dimensions[qudit], controls)

    @add_gate_decorator
    def s(self, qudit: int, controls: ControlData | None = None) -> S:
        return S(self, "S" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def x(self, qudit: int, controls: ControlData | None = None) -> X:
        return X(self, "X" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    @add_gate_decorator
    def z(self, qudit: int, controls: ControlData | None = None) -> Z:
        return Z(self, "Z" + str(self.dimensions[qudit]), qudit, self.dimensions[qudit], controls)

    def replace_gate(self, gate_index: int, sequence: Sequence[Gate]) -> None:
        self.instructions[gate_index : gate_index + 1] = sequence
        self.number_gates = (self.number_gates - 1) + len(sequence)

    def set_instructions(self, sequence: Sequence[Gate]) -> QuantumCircuit:
        self.instructions = []
        self.instructions += sequence
        self.number_gates = len(sequence)
        return self

    def set_final_mappings(self, mappings: list[list[int]]) -> QuantumCircuit:
        self.final_mappings = mappings
        return self

    def set_initial_mappings(self, mappings: list[list[int]]) -> QuantumCircuit:
        self.initial_mappings = mappings
        return self

    def from_qasm(self, qasm_prog: str) -> None:
        """Create a circuit from qasm text."""
        self.reset()
        qasm_parser = QASM().parse_ditqasm2_str(qasm_prog)
        instructions = qasm_parser["instructions"]
        temp_sitemap = qasm_parser["sitemap"]
        cl_sitemap = qasm_parser["sitemap_classic"]

        for qreg in QuantumRegister.from_map(temp_sitemap):
            self.append(qreg)

        for creg in ClassicRegister.from_map(cl_sitemap):
            self.append_classic(creg)

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
                    if is_not_none_or_empty(op["params"]):
                        if op["controls"]:
                            gate = function(qudits_call, op["params"], op["controls"])
                        else:
                            gate = function(qudits_call, op["params"])
                    elif op["controls"]:
                        gate = function(qudits_call, op["controls"])
                    else:
                        gate = function(qudits_call)
                else:
                    msg = "the required gate_matrix is not available anymore."
                    raise NotImplementedError(msg)
                if op["dagger"]:
                    gate.dag()

    def to_qasm(self) -> str:
        text = ""
        text += "DITQASM 2.0;\n"
        for qreg in self.quantum_registers:
            text += qreg.__qasm__()
            text += "\n"
        text += f"creg meas[{len(self.dimensions)}];\n"

        for op in self.instructions:
            if op.to_qasm() is None:
                pass
            text += op.to_qasm()

        cregs_indices = iter(list(range(len(self.dimensions))))
        for qreg in self.quantum_registers:
            for i in range(qreg.size):
                text += f"measure {qreg.label}[{i}] -> meas[{next(cregs_indices)}];\n"

        return text

    def save_to_file(self, file_name: str, file_path: str = ".") -> str:
        """Save qasm into a file with the specified name and path.

        Args:
            text (str): The text to be saved into the file.
            file_name (str): The name of the file.
            file_path (str, optional): The path where the file will be saved. Defaults to "." (current directory).

        Returns:
            str: The full path of the saved file.
        """
        # Combine the file path and name to get the full file path
        self.path_save = file_path
        full_file_path = Path(file_path) / (file_name + ".qasm")

        # Write the text to the file
        with full_file_path.open("w+") as file:
            file.write(self.to_qasm())

        self.path_save = None
        return str(full_file_path)

    def load_from_file(self, file_path: str) -> None:
        """Load text from a file.

        Args:
            file_path (str): The path of the file to load.

        Returns:
            str: The text loaded from the file.
        """
        with Path(file_path).open("r", encoding=locale.getpreferredencoding(False)) as file:
            text = file.read()
        self.from_qasm(text)

    @property
    def gate_set(self) -> str:
        return "\n".join(self.qasm_to_gate_set_dict.values())

    def simulate(self) -> NDArray:
        from mqt.qudits.simulation import MQTQuditProvider

        provider = MQTQuditProvider()
        backend = provider.get_backend("tnsim")
        job = backend.run(self)
        result = job.result()
        return result.get_state_vector()

    def compileO0(self, backend_name: str) -> QuantumCircuit:  # noqa: N802
        from mqt.qudits.compiler import QuditCompiler
        from mqt.qudits.simulation import MQTQuditProvider

        qudit_compiler = QuditCompiler()
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend(backend_name)

        return qudit_compiler.compile_O0(backend_ion, self)

    def compileO1(self, backend_name: str, mode: str = "resynth") -> QuantumCircuit:  # noqa: N802
        from mqt.qudits.compiler import QuditCompiler
        from mqt.qudits.simulation import MQTQuditProvider

        qudit_compiler = QuditCompiler()
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend(backend_name)

        if mode == "adapt":
            new_circuit = qudit_compiler.compile_O1_adaptive(backend_ion, self)
        elif mode == "resynth":
            new_circuit = qudit_compiler.compile_O1_resynth(backend_ion, self)
        else:
            msg = f"mode {mode} not supported"
            raise ValueError(msg)

        return new_circuit

    def compileO2(self, backend_name: str) -> QuantumCircuit:  # noqa: N802
        from mqt.qudits.compiler import QuditCompiler
        from mqt.qudits.simulation import MQTQuditProvider

        qudit_compiler = QuditCompiler()
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend(backend_name)

        return qudit_compiler.compile_O2(backend_ion, self)

    def set_initial_state(self, state: ArrayLike, approx: bool = False) -> QuantumCircuit:
        from mqt.qudits.compiler.state_compilation.state_preparation import StatePrep

        preparation = StatePrep(self, state, approx)
        new_circuit = preparation.compile_state()
        self.set_instructions(new_circuit.instructions)
        return self
