import math
import re
import warnings


class QASM:
    def __init__(self):
        self._program = None

    def parse_ditqasm2_str(self, contents):
        """Parse the string contents of an OpenQASM 2.0 file. This parser only
        supports basic gate definitions, and is not guaranteed to check the full
        openqasm grammar.
        """
        # define regular expressions for parsing
        rgxs = {
            "header":      re.compile(r"(DITQASM\s+2.0;)|(include\s+\"qelib1.inc\";)"),
            "comment":     re.compile(r"^//"),
            "comment_start": re.compile(r"/\*"),
            "comment_end": re.compile(r"\*/"),
            "qreg":        re.compile(r"qreg\s+(\w+)\s+(\[\s*\d+\s*\])(?:\s*\[(\d+(?:,\s*\d+)*)\])?;"),
            "gate":        re.compile(r"(\w+)\s*(?:\(([^)]*)\))?\s*(\w+\[\d+\]\s*(,\s*\w+\[\d+\])*)\s*;"),
            "error":       re.compile(r"^(gate|if)"),
            "ignore":      re.compile(r"^(creg|measure|barrier)"),
        }

        # initialise number of qubits to zero and an empty list for gates
        sitemap = {}
        gates = []
        # only want to warn once about each ignored instruction
        warned = {}

        # Process each line
        in_comment = False
        for line in contents.split("\n"):
            line = line.strip()

            if not line:
                # blank line
                continue
            if rgxs["comment"].match(line):
                # single comment
                continue
            if rgxs["comment_start"].match(line):
                # start of multiline comments
                in_comment = True
            if in_comment:
                # in multiline comment, check if its ending
                in_comment = not bool(rgxs["comment_end"].match(line))
                continue
            if rgxs["header"].match(line):
                # ignore standard header lines
                continue

            match = rgxs["qreg"].match(line)
            if match:
                name, nq, qsizes = match.groups()
                nq = int(*re.search(r"\[(\d+)\]", nq).groups())

                if qsizes:
                    qsizes = qsizes.split(",") if qsizes else []
                    qsizes = [int(num) for num in qsizes]
                else:
                    qsizes = [2] * nq

                for i in range(int(nq)):
                    sitemap[f"{name}[{i}]"] = len(sitemap), qsizes[i]

                continue

            match = rgxs["ignore"].match(line)
            if match:
                # certain operations we can just ignore and warn about
                (op,) = match.groups()
                if not warned.get(op, False):
                    warnings.warn(f"Unsupported operation ignored: {op}", SyntaxWarning)
                    warned[op] = True
                continue

            if rgxs["error"].match(line):
                # raise hard error for custom tate defns etc
                msg = f"Custom gate definitions are not supported: {line}"
                raise NotImplementedError(msg)

            match = rgxs["gate"].search(line)
            if match:
                # apply a gate
                label, params, qubits = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                )

                if params:
                    params = tuple(eval(param, {"pi": math.pi}) for param in params.strip("()").split(","))
                else:
                    params = ()

                qubits = tuple(sitemap[qubit.strip()] for qubit in qubits.split(","))
                gates.append((label, params, qubits))
                continue

            # if not covered by previous checks, simply raise
            msg = f"{line}"
            raise SyntaxError(msg)
        self._program = {
            "n":     len(sitemap),
            "sitemap": sitemap,
            "gates": gates,
            "n_gates": len(gates),
        }
        return self._program

    def parse_ditqasm2_file(self, fname, **kwargs):
        """Parse an OpenQASM 2.0 file."""
        with open(fname) as f:
            return self.parse_ditqasm2_str(f.read(), **kwargs)
