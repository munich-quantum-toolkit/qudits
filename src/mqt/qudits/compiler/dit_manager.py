from __future__ import annotations

from .onedit import LogLocAdaPass, LogLocQRPass, PhyLocAdaPass, PhyLocQRPass
from .twodit import LogEntQRCEXPass


class QuditCompiler:
    passes_enabled = {
        "PhyLocQRPass": PhyLocQRPass,
        "PhyLocAdaPass": PhyLocAdaPass,
        "LocQRPass": PhyLocQRPass,
        "LocAdaPass": PhyLocAdaPass,
        "LogLocAdaPass": LogLocAdaPass,
        "LogLocQRPass": LogLocQRPass,
        "LogEntQRCEXPass": LogEntQRCEXPass,
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend, circuit, passes_names):
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass]
            decomposition = compiler_pass(backend)
            circuit = decomposition.transpile(circuit)
        return circuit
