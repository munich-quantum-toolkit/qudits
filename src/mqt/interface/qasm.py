import re
import warnings
from pathlib import Path

import sympy as sp


class QASM:
    def __init__(self):
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

    def parse_gate(self, line, rgxs, sitemap, gates):
        match = rgxs["gate"].search(line)
        if match:
            label, params, qudits = (
                match.group(1),
                match.group(2),
                match.group(3),
            )

            params = (
                tuple(sp.sympify(param.replace("pi", str(sp.pi))) for param in params.strip("()").split(","))
                if params
                else ()
            )
            for dit in qudits.split(","):
                match = rgxs["qreg_indexing"].match(str(dit))
                if match:
                    name, reg_qudit_index = match.groups()
                    reg_qudit_index = int(*re.search(r"\[(\d+)\]", reg_qudit_index).groups())
                    qudits = tuple(sitemap[(name, reg_qudit_index)])
                    gate_dict = {"name": label, "params": params, "qudits": qudits}
                    gates.append(gate_dict)

            return True
        return False

    def parse_ditqasm2_str(self, contents):
        """Parse the string contents of an OpenQASM 2.0 file. This parser only
        supports basic gate definitions, and is not guaranteed to check the full
        openqasm grammar.
        """
        # define regular expressions for parsing
        rgxs = {
            "header": re.compile(r"(DITQASM\s+2.0;)|(include\s+\"qelib1.inc\";)"),
            "comment": re.compile(r"^//"),
            "comment_start": re.compile(r"/\*"),
            "comment_end": re.compile(r"\*/"),
            "qreg": re.compile(r"qreg\s+(\w+)\s+(\[\s*\d+\s*\])(?:\s*\[(\d+(?:,\s*\d+)*)\])?;"),
            "qreg_indexing": re.compile(r"\s*(\w+)\s*(\[\s*\d+\s*\])"),
            "gate": re.compile(r"(\w+)\s*(?:\(([^)]*)\))?\s*(\w+\[\d+\]\s*(,\s*\w+\[\d+\])*)\s*;"),
            "error": re.compile(r"^(gate|if)"),
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

                continue

            match = rgxs["ignore"].match(line)
            if match:
                # certain operations we can just ignore and warn about
                (op,) = match.groups()
                if not warned.get(op, False):
                    warnings.warn(f"Unsupported operation ignored: {op}", SyntaxWarning, stacklevel=2)
                    warned[op] = True
                continue

            if rgxs["error"].match(line):
                # raise hard error for custom tate defns etc
                msg = f"Custom gate definitions are not supported: {line}"
                raise NotImplementedError(msg)

            if self.parse_gate(line, rgxs, sitemap, gates):
                continue

            # if not covered by previous checks, simply raise
            msg = f"{line}"
            raise SyntaxError(msg)
        self._program = {
            "n": len(sitemap),
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
