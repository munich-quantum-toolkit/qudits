from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.controls import ControlData


class QASM:
    def __init__(self) -> None:
        self._program = None

    def parse_nonspecial_lines(self, line, rgxs, in_comment_flag):
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

    def parse_qreg(self, line, rgxs, sitemap) -> bool:
        match = rgxs["qreg"].match(line)
        if match:
            name, nq, qdims = match.groups()
            nq = int(*re.search(r"\[(\d+)\]", nq).groups())

            if qdims:
                qdims = qdims.split(",") if qdims else []
                qdims = [int(num) for num in qdims]
            else:
                qdims = [2] * nq

            for i in range(int(nq)):
                sitemap[(str(name), i)] = len(sitemap), qdims[i]

            return True
        return False

    def safe_eval_math_expression(self, expression):
        try:
            expression = expression.replace("pi", str(np.pi))
            # Define a dictionary of allowed names
            allowed_names = {"__builtins__": None, "pi": np.pi}
            # Use eval with restricted names
            return eval(expression, allowed_names)
        except (ValueError, SyntaxError):
            return None

    def parse_gate(self, line, rgxs, sitemap, gates) -> bool:
        match = rgxs["gate_matrix"].search(line)
        if match:
            label = match.group(1)
            params = match.group(2)
            qudits = match.group(3)
            ctl_pattern = match.group(5)
            ctl_qudits = match.group(6)
            ctl_levels = match.group(8)

            # params = (
            #     tuple(sp.sympify(param.replace("pi", str(sp.pi))) for param in params.strip("()").split(","))
            #     if params
            #     else ()
            # )
            # Evaluate params using NumPy and NumExpr
            params = (
                tuple(self.safe_eval_math_expression(param) for param in params.strip("()").split(","))
                if params
                else ()
            )

            qudits_list = []
            for dit in qudits.split(","):
                match = rgxs["qreg_indexing"].match(str(dit))
                if match:
                    name, reg_qudit_index = match.groups()
                    reg_qudit_index = int(*re.search(r"\[(\d+)\]", reg_qudit_index).groups())
                    qudit = tuple(sitemap[(name, reg_qudit_index)])
                    qudits_list.append(qudit)

            qudits_control_list = []
            if ctl_pattern is not None:
                matches = rgxs["qreg_indexing"].findall(ctl_qudits)
                for match in matches:
                    name, reg_qudit_index = match
                    reg_qudit_index = int(*re.search(r"\[(\d+)\]", reg_qudit_index).groups())
                    qudit = tuple(sitemap[(name, reg_qudit_index)])
                    qudits_control_list.append(qudit[0])

            qudits_levels_list = []
            if ctl_levels is not None:
                numbers = re.compile(r"\d+")
                matches = numbers.findall(ctl_levels)
                for level in matches:
                    qudits_levels_list.append(int(level))

            if len(qudits_control_list) == 0 and len(qudits_levels_list) == 0:
                controls = None
            else:
                controls = ControlData(qudits_control_list, qudits_levels_list)
            gate_dict = {"name": label, "params": params, "qudits": qudits_list, "controls": controls}

            gates.append(gate_dict)

            return True
        return False

    def parse_ignore(self, line, rgxs, warned) -> bool:
        match = rgxs["ignore"].match(line)
        if match:
            # certain operations we can just ignore and warn about
            (op,) = match.groups()
            if not warned.get(op, False):
                # warnings.warn(f"Unsupported operation ignored: {op}", SyntaxWarning, stacklevel=2)
                warned[op] = True
            return True
        return False

    def parse_ditqasm2_str(self, contents):
        """Parse the string contents of an OpenQASM 2.0 file. This parser only
        supports basic gate_matrix definitions, and is not guaranteed to check the full
        openqasm grammar.
        """
        # define regular expressions for parsing
        rgxs = {
            "header": re.compile(r"(DITQASM\s+2.0;)|(include\s+\"qelib1.inc\";)"),
            "comment": re.compile(r"^//"),
            "comment_start": re.compile(r"/\*"),
            "comment_end": re.compile(r"\*/"),
            "qreg": re.compile(r"qreg\s+(\w+)\s+(\[\s*\d+\s*\])(?:\s*\[(\d+(?:,\s*\d+)*)\])?;"),
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
            "ignore": re.compile(r"^(creg|measure|barrier)"),
        }

        # initialise number of qubits to zero and an empty list for instructions
        sitemap = {}
        gates = []
        # only want to warn once about each ignored instruction
        warned = {}

        # Process each line
        in_comment = False
        for current_line in contents.split("\n"):
            line = current_line.strip()

            continue_flag, in_comment = self.parse_nonspecial_lines(line, rgxs, in_comment)
            if continue_flag:
                continue

            if self.parse_qreg(line, rgxs, sitemap):
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
        self._program = {
            "circuits_size": len(sitemap),
            "sitemap": sitemap,
            "instructions": gates,
            "n_gates": len(gates),
        }
        return self._program

    def parse_ditqasm2_file(self, fname):
        """Parse an OpenQASM 2.0 file."""
        path = Path(fname)
        with path.open() as f:
            return self.parse_ditqasm2_str(f.read())
