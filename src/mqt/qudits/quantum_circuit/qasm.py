from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .components.extensions.controls import ControlData

if TYPE_CHECKING:
    from .components.classic_register import ClSitemap
    from .components.quantum_register import SiteMap


class QASM:
    """Class that manages the parsing of QASM programs."""

    def __init__(self) -> None:
        self.program: dict[str, Any] = {}

    @staticmethod
    def parse_nonspecial_lines(line: str, rgxs: dict[str, re.Pattern[str]], in_comment_flag: bool) -> tuple[bool, bool]:
        in_comment = in_comment_flag

        if not line:
            # blank line
            return True, in_comment
        if rgxs["comment"].match(line):
            # single comment
            return True, in_comment
        if rgxs["comment_start"].match(line):
            # start of multiline comments
            in_comment = True
        if in_comment:
            # in multiline comment, check if its ending
            in_comment = not bool(rgxs["comment_end"].match(line))
            return True, in_comment
        if rgxs["header"].match(line):
            # ignore standard header lines
            return True, in_comment
        return False, in_comment

    @staticmethod
    def parse_qreg(
        line: str, rgxs: dict[str, re.Pattern[str]], sitemap: dict[tuple[str, int], tuple[int, int]]
    ) -> bool:
        match = rgxs["qreg"].match(line)
        if match:
            name, nqu, qdims = match.groups()
            if nqu is not None:
                int_matchings = re.search(r"\[(\d+)\]", nqu)
                if int_matchings is not None:
                    nq = int(int_matchings.group(1))

            if qdims:
                qdims = qdims.split(",") if qdims else []
                qdims = [int(num) for num in qdims]
            else:
                qdims = [2] * nq

            for i in range(int(nq)):
                sitemap[str(name), i] = len(sitemap), qdims[i]

            return True
        return False

    @staticmethod
    def parse_creg(line: str, rgxs: dict[str, re.Pattern[str]], sitemap_classic: ClSitemap) -> bool:
        match = rgxs["creg"].match(line)
        if match:
            name, nclassics = match.groups()
            for i in range(int(nclassics)):
                sitemap_classic[str(name), i] = (len(sitemap_classic),)
            return True
        return False

    @staticmethod
    def safe_eval_math_expression(expression: str) -> float | None:
        try:
            expression = expression.replace("pi", str(np.pi))
            # Define a dictionary of allowed names
            allowed_names = {"__builtins__": None, "pi": np.pi}
            # Use eval with restricted names
            return float(eval(expression, allowed_names))  # noqa: S307
        except (ValueError, SyntaxError):
            return None

    @staticmethod
    def parse_gate(
        line: str,
        rgxs: dict[str, re.Pattern[str]],
        sitemap: dict[tuple[str, int], tuple[int, int]],
        gates: list[dict[str, Any]],
    ) -> bool:
        match = rgxs["gate_matrix"].search(line)
        if match:
            label = match.group(1)
            params = match.group(2)
            qudits = match.group(3)
            ctl_pattern = match.group(5)
            ctl_qudits = match.group(6)
            ctl_levels = match.group(8)

            # Evaluate params using NumPy and NumExpr
            if params:
                if ".npy" in params:
                    params = np.load(params)
                else:
                    # TODO: This does not handle "custom_data" correctly
                    params = tuple(QASM.safe_eval_math_expression(param) for param in params.strip("()[]").split(","))
            else:
                params = ()

            qudits_list = []
            for dit in qudits.split(","):
                match = rgxs["qreg_indexing"].match(str(dit))
                if match:
                    name, reg_qudit_index = match.groups()
                    if reg_qudit_index:
                        extracted_reg = re.search(r"\[(\d+)\]", reg_qudit_index)
                        if extracted_reg:
                            reg_qudit_index = int(extracted_reg.groups()[0])
                            qudit = tuple(sitemap[name, reg_qudit_index])
                            qudits_list.append(qudit)

            qudits_control_list = []
            if ctl_pattern is not None:
                matches = rgxs["qreg_indexing"].findall(ctl_qudits)
                for match in matches:
                    if match:
                        namectl = match[0]
                        reg_qudit_indexctl = cast("str", match[1])
                        number = re.search(r"\[(\d+)\]", reg_qudit_indexctl)
                        if number:
                            extracted = number.groups()
                            reg_qudit_index_i = int(extracted[0])
                            qudit = tuple(sitemap[namectl, reg_qudit_index_i])
                            qudits_control_list.append(qudit[0])

            numbers = re.compile(r"\d+")
            qudits_levels_list = [int(level) for level in numbers.findall(ctl_levels)] if ctl_levels is not None else []

            if len(qudits_control_list) == 0 and len(qudits_levels_list) == 0:
                controls = None
            else:
                controls = ControlData(qudits_control_list, qudits_levels_list)
            gate_dict = {"name": label, "params": params, "qudits": qudits_list, "controls": controls}

            gates.append(gate_dict)

            return True
        return False

    @staticmethod
    def parse_ignore(line: str, rgxs: dict[str, re.Pattern[str]], warned: dict[str, bool]) -> bool:
        match = rgxs["ignore"].match(line)
        if match:
            # certain operations we can just ignore and warn about
            (op,) = match.groups()
            if not warned.get(op, False):
                # warnings.warn(f"Unsupported operation ignored: {op}", SyntaxWarning, stacklevel=2)
                warned[op] = True
            return True
        return False

    def parse_ditqasm2_str(self, contents: str) -> dict[str, Any]:
        """Parse the string contents of an OpenQASM 2.0 file.

        This parser only supports basic gate_matrix definitions, and is not guaranteed to check the full OpenQASM grammar.
        """
        # define regular expressions for parsing
        rgxs: dict[str, re.Pattern[str]] = {
            "header": re.compile(r"(DITQASM\s+2.0;)|(include\s+\"qelib1.inc\";)"),
            "comment": re.compile(r"^//"),
            "comment_start": re.compile(r"/\*"),
            "comment_end": re.compile(r"\*/"),
            "qreg": re.compile(r"qreg\s+(\w+)\s+(\[\s*\d+\s*\])(?:\s*\[(\d+(?:,\s*\d+)*)\])?;"),
            "creg": re.compile(r"creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;"),
            # "ctrl_id":       re.compile(r"\s+(\w+)\s*(\[\s*\other_size+\s*\])\s*(\s*\w+\s*\[\other_size+\])*\s*"),
            "qreg_indexing": re.compile(r"\s*(\w+)\s*(\[\s*\d+\s*\])"),
            # "gate_matrix":
            # re.compile(r"(\w+)\s*(?:\(([^)]*)\))?\s*(\w+\[\other_size+\]\s*(,\s*\w+\[\other_size+\])*)\s*;"),
            "gate_matrix": re.compile(
                r"(\w+)\s*(?:\(([^)]*)\))?\s*(\w+\[\d+\]\s*(,\s*\w+\[\d+\])*)\s*"
                r"(ctl(\s+\w+\[\d+\]\s*(\s*\w+\s*\[\d+\])*)\s*(\[(\d+(,\s*\d+)*)\]))?"
                r"\s*;"
            ),
            "error": re.compile(r"^(gate_matrix|if)"),
            "ignore": re.compile(r"^(measure|barrier)"),
        }

        # initialise number of qubits to zero and an empty list for instructions
        sitemap: SiteMap = {}
        sitemap_classic: ClSitemap = {}

        gates: list[dict[str, Any]] = []
        # only want to warn once about each ignored instruction
        warned: dict[str, bool] = {}

        # Process each line
        in_comment = False
        for current_line in contents.split("\n"):
            line = current_line.strip()

            continue_flag, in_comment = self.parse_nonspecial_lines(line, rgxs, in_comment)
            if continue_flag:
                continue

            if self.parse_qreg(line, rgxs, sitemap):
                continue

            if self.parse_creg(line, rgxs, sitemap_classic):
                continue

            if self.parse_ignore(line, rgxs, warned):
                continue

            if self.parse_gate(line, rgxs, sitemap, gates):
                continue

            if rgxs["error"].match(line):
                # raise hard error for custom tate defns etc
                msg = f"Custom gate_matrix definitions are not supported: {line}"
                raise NotImplementedError(msg)

            # if not covered by previous checks, simply raise
            msg = f"{line}"
            raise SyntaxError(msg)
        self.program = {
            "circuits_size": len(sitemap),
            "sitemap": sitemap,
            "sitemap_classic": sitemap_classic,
            "instructions": gates,
            "n_gates": len(gates),
        }
        return self.program

    def parse_ditqasm2_file(self, fname: str) -> dict[str, Any]:
        """Parse an OpenQASM 2.0 file."""
        path = Path(fname)
        with path.open() as f:
            return self.parse_ditqasm2_str(f.read())
