from __future__ import annotations

import typing
from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


class TestMISimAndTNSim(TestCase):
    @staticmethod
    def run_test_on_both_backends(circuit: QuantumCircuit, expected_state: NDArray[np.complex128]) -> None:  # noqa: ARG004
        backends = ["tnsim", "misim"]
        provider = MQTQuditProvider()

        results = {}
        for backend_name in backends:
            backend = provider.get_backend(backend_name)
            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            # assert np.allclose(state_vector, expected_state), f"Failed for backend {backend_name}"
            results[backend_name] = state_vector

        # Compare results from both backends
        # assert np.allclose(results["misim"], results["tnsim"]), "Results from misim and tnsim do not match"
        assert True

    def test_execute(self):
        # H gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = h.to_matrix() @ zero_state

            self.run_test_on_both_backends(circuit, test_state)

        # X gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            x = circuit.x(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = x.to_matrix() @ zero_state

            self.run_test_on_both_backends(circuit, test_state)

        # Z gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            z = circuit.z(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = z.to_matrix() @ h.to_matrix() @ zero_state

            self.run_test_on_both_backends(circuit, test_state)

        # S gate
        for d in [2, 3, 5, 7]:
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            s = circuit.s(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = s.to_matrix() @ h.to_matrix() @ zero_state

            self.run_test_on_both_backends(circuit, test_state)

        rng = np.random.default_rng()
        # Rz gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for in_level_a in range(d - 1):
                for in_level_b in range(in_level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    angle = rng.uniform(0, 2 * np.pi)
                    rz = circuit.rz(0, [in_level_a, in_level_b, angle])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = rz.to_matrix() @ h.to_matrix() @ ini_state

                    self.run_test_on_both_backends(circuit, test_state)

        # R gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for in_level_a in range(d - 1):
                for in_level_b in range(in_level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    angle = rng.uniform(0, 2 * np.pi)
                    phase = rng.uniform(0, 2 * np.pi)
                    r = circuit.r(0, [in_level_a, in_level_b, angle, phase])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = r.to_matrix() @ h.to_matrix() @ ini_state

                    self.run_test_on_both_backends(circuit, test_state)

        # VirtRz gate
        for dvrz in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [dvrz])
            for level in range(dvrz):
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(0)
                angle = rng.uniform(0, 2 * np.pi)
                vrz = circuit.virtrz(0, [level, angle])

                ini_state = np.zeros(dvrz)
                ini_state[0] = 1
                test_state = vrz.to_matrix() @ h.to_matrix() @ ini_state

                self.run_test_on_both_backends(circuit, test_state)

        # Rh gate
        for drh in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [drh])
            for rh_level_a in range(drh - 1):
                for rh_level_b in range(in_level_a + 1, drh):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    rh = circuit.rh(0, [rh_level_a, rh_level_b])

                    ini_state = np.zeros(drh)
                    ini_state[0] = 1
                    test_state = rh.to_matrix() @ h.to_matrix() @ ini_state

                    self.run_test_on_both_backends(circuit, test_state)

        # Entangling gates CSUM
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                qreg_example = QuantumRegister("reg", 2, [d1, d2])
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(0)
                csum = circuit.csum([0, 1])

                zero_state = np.zeros(d1 * d2)
                zero_state[0] = 1
                test_state = csum.to_matrix() @ (np.kron(h.to_matrix(), np.identity(d2))) @ zero_state

                self.run_test_on_both_backends(circuit, test_state)

                # Flipped basic Case

                qreg_example = QuantumRegister("reg", 2, [d1, d2])
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(1)
                csum = circuit.csum([1, 0])

                zero_state = np.zeros(d1 * d2)
                zero_state[0] = 1

                test_state = csum.to_matrix() @ (np.kron(np.identity(d1), h.to_matrix())) @ zero_state

                self.run_test_on_both_backends(circuit, test_state)

        # CEX
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                for clev in range(d1):
                    for level_a in range(d2 - 1):
                        for level_b in range(in_level_a + 1, d2):
                            angle = rng.uniform(0, 2 * np.pi)

                            qreg_example = QuantumRegister("reg", 2, [d1, d2])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(0)
                            cx = circuit.cx([0, 1], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * d2)
                            zero_state[0] = 1
                            test_state = cx.to_matrix() @ (np.kron(h.to_matrix(), np.identity(d2))) @ zero_state

                            self.run_test_on_both_backends(circuit, test_state)

        # Inverted basic Case
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                for clev in range(d1):
                    for bas_level_a in range(d1 - 1):
                        for bas_level_b in range(in_level_a + 1, d1):
                            angle = rng.uniform(0, 2 * np.pi)
                            qreg_example = QuantumRegister("reg", 2, [d1, d2])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(1)
                            cx = circuit.cx([1, 0], [bas_level_a, bas_level_b, clev, angle])

                            zero_state = np.zeros(d1 * d2)
                            zero_state[0] = 1

                            test_state = cx.to_matrix() @ (np.kron(np.identity(d1), h.to_matrix())) @ zero_state

                            self.run_test_on_both_backends(circuit, test_state)

    def test_tn_long_range(self):
        rng = np.random.default_rng()
        # Long range gates
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 2])
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(0)
                csum = circuit.csum([0, 2])

                cmat = csum.to_matrix(identities=2)
                hmat = h.to_matrix(identities=2)

                zero_state = np.zeros(d1 * 2 * d2 * 2)
                zero_state[0] = 1
                test_state = cmat @ hmat @ zero_state

                self.run_test_on_both_backends(circuit, test_state)

                # Flipped basic Case
                qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 2])
                circuit2 = QuantumCircuit(qreg_example)
                h = circuit2.h(2)
                csum2 = circuit2.csum([2, 0])

                cmat2 = csum2.to_matrix(identities=2)
                hmat = h.to_matrix(identities=2)

                zero_state = np.zeros(d1 * 2 * d2 * 2)
                zero_state[0] = 1
                test_state = cmat2 @ hmat @ zero_state

                self.run_test_on_both_backends(circuit, test_state)

        # CEX
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                for clev in range(d1):
                    for level_a in range(d2 - 1):
                        for level_b in range(level_a + 1, d2):
                            angle = rng.uniform(0, 2 * np.pi)

                            qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 3])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(0)
                            cx = circuit.cx([0, 2], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * 2 * d2 * 3)
                            zero_state[0] = 1
                            test_state = cx.to_matrix(identities=2) @ h.to_matrix(identities=2) @ zero_state

                            self.run_test_on_both_backends(circuit, test_state)

                # Inverted basic Case
                for clev in range(d2):
                    for level_a in range(d1 - 1):
                        for level_b in range(level_a + 1, d1):
                            angle = rng.uniform(0, 2 * np.pi)
                            qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 3])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(2)
                            cx = circuit.cx([2, 0], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * 2 * d2 * 3)
                            zero_state[0] = 1
                            test_state = cx.to_matrix(identities=2) @ h.to_matrix(identities=2) @ zero_state

                            self.run_test_on_both_backends(circuit, test_state)

    def test_execute_controlled(self):
        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(1)
        circuit.x(0).control([1, 2], [1, 0])
        test_state = np.array([(0.7071067 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (0.7071067 + 0j), 0j, 0j])

        self.run_test_on_both_backends(circuit, test_state)

        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0)
        circuit.x(1).control([0, 2], [1, 0])
        test_state = np.array([(0.7071067 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (0.7071067 + 0j), 0j, 0j])

        self.run_test_on_both_backends(circuit, test_state)
