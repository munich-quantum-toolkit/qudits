from mqt.qudits.compiler.onedit.local_adaptive_decomp import LocAdaPass
from mqt.qudits.compiler.onedit.local_qr_decomp import LocQRPass
from mqt.qudits.compiler.onedit.propagate_virtrz import ZPropagationPass
from mqt.qudits.compiler.onedit.remove_phase_rotations import ZRemovalPass


class QuditManager():
    def __init__(self):
        pass

    def compile(self, backend, circuit, passes_names):
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = globals()[compiler_pass]
            decomposition = compiler_pass(backend)
            circuit = decomposition.transpile(circuit)
        return circuit
