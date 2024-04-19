from mqt.qudits.compiler.onedit.mapping_aware_transpilation.phy_local_adaptive_decomp import PhyLocAdaPass
from mqt.qudits.compiler.onedit.mapping_aware_transpilation.phy_local_qr_decomp import PhyLocQRPass
from mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_adaptive_decomp import LogLocAdaPass
from mqt.qudits.compiler.onedit.mapping_un_aware_transpilation.log_local_qr_decomp import LogLocQRPass
from mqt.qudits.compiler.twodit.mapping_un_aware_transpilation.entanglement_qr.cex_decomposition.log_ent_qr_cex_decomp import (
    LogEntQRCEXPass,
)


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

    def __init__(self):
        pass

    def compile(self, backend, circuit, passes_names):
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass]
            decomposition = compiler_pass(backend)
            circuit = decomposition.transpile(circuit)
        return circuit
