from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestTNSim(TestCase):
    def test_execute(self):

        provider = MQTQuditProvider()
        backend = provider.get_backend("tnsim")

        # H gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = h.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(
                    state_vector,
                    test_state
            )

        # X gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            gate = circuit.x(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = gate.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(
                    state_vector,
                    test_state
            )

        # Z gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            gate = circuit.z(0)
            matrix_gate = gate.to_matrix()

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = gate.to_matrix() @ h.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(state_vector, test_state)

        # S gate
        for d in [2, 3, 5, 7]:
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            gate = circuit.s(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = gate.to_matrix() @ h.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(
                    state_vector,
                    test_state
            )

        # Rz gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for level_a in range(d - 1):
                for level_b in range(level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    gate = circuit.rz(0, [level_a, level_b, angle])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = gate.to_matrix() @ h.to_matrix() @ ini_state

                    job = backend.run(circuit)
                    result = job.result()
                    state_vector = result.get_state_vector()

                    assert np.allclose(state_vector, test_state)

        # Rh gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for level_a in range(d - 1):
                for level_b in range(level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    gate = circuit.rh(0, [level_a, level_b])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = gate.to_matrix() @ h.to_matrix() @ ini_state

                    job = backend.run(circuit)
                    result = job.result()
                    state_vector = result.get_state_vector()

                    assert np.allclose(state_vector, test_state)

        # R gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for level_a in range(d - 1):
                for level_b in range(level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    angle = np.random.uniform(0, 2 * np.pi)
                    phase = np.random.uniform(0, 2 * np.pi)
                    gate = circuit.r(0, [level_a, level_b, angle, phase])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = gate.to_matrix() @ h.to_matrix() @ ini_state

                    job = backend.run(circuit)
                    result = job.result()
                    state_vector = result.get_state_vector()

                    assert np.allclose(state_vector, test_state)

        # VirtRz gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for level in range(d):
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(0)
                angle = np.random.uniform(0, 2 * np.pi)
                gate = circuit.virtrz(0, [level, angle])
                matrix_gate = gate.to_matrix()

                ini_state = np.zeros(d)
                ini_state[0] = 1
                test_state = gate.to_matrix() @ h.to_matrix() @ ini_state

                job = backend.run(circuit)
                result = job.result()
                state_vector = result.get_state_vector()

                assert np.allclose(state_vector, test_state)

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

                job = backend.run(circuit)
                result = job.result()
                state_vector = result.get_state_vector()

                assert np.allclose(state_vector, test_state)

                # Flipped basic Case

                qreg_example = QuantumRegister("reg", 2, [d1, d2])
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(1)
                csum = circuit.csum([1, 0])

                zero_state = np.zeros(d1 * d2)
                zero_state[0] = 1

                test_state = csum.to_matrix() @ (np.kron(np.identity(d1), h.to_matrix())) @ zero_state

                job = backend.run(circuit)
                result = job.result()
                state_vector = result.get_state_vector()

                assert np.allclose(state_vector, test_state)

        # CEX
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                for clev in range(0, d1):
                    for level_a in range(d2 - 1):
                        for level_b in range(level_a + 1, d2):
                            angle = np.random.uniform(0, 2 * np.pi)

                            qreg_example = QuantumRegister("reg", 2, [d1, d2])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(0)
                            cx = circuit.cx([0, 1], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * d2)
                            zero_state[0] = 1
                            test_state = cx.to_matrix() @ (np.kron(h.to_matrix(), np.identity(d2))) @ zero_state

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

                # Inverted basic Case
                for clev in range(0, d2):
                    for level_a in range(d1 - 1):
                        for level_b in range(level_a + 1, d1):
                            angle = np.random.uniform(0, 2 * np.pi)
                            qreg_example = QuantumRegister("reg", 2, [d1, d2])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(1)
                            cx = circuit.cx([1, 0], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * d2)
                            zero_state[0] = 1

                            test_state = cx.to_matrix() @ (np.kron(np.identity(d1), h.to_matrix())) @ zero_state

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

    def test_tn_long_range(self):
        provider = MQTQuditProvider()
        backend = provider.get_backend("tnsim")
        # Long range gates
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                print("Test long range CSUM")
                qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 2])
                circuit = QuantumCircuit(qreg_example)
                h = circuit.h(0)
                csum = circuit.csum([0, 2])

                cmat = csum.to_matrix(identities=2)
                hmat = h.to_matrix(identities=2)

                zero_state = np.zeros(d1 * 2 * d2 * 2)
                zero_state[0] = 1
                test_state = cmat @ hmat @ zero_state

                job = backend.run(circuit)
                result = job.result()
                state_vector = result.get_state_vector()

                assert np.allclose(state_vector, test_state)

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

                job = backend.run(circuit2)
                result = job.result()
                state_vector = result.get_state_vector()

                assert np.allclose(state_vector, test_state)

        # CEX
        for d1 in range(2, 8):
            for d2 in range(2, 8):
                for clev in range(0, d1):
                    for level_a in range(d2 - 1):
                        for level_b in range(level_a + 1, d2):
                            print("Test long range CEX")
                            angle = np.random.uniform(0, 2 * np.pi)

                            qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 3])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(0)
                            cx = circuit.cx([0, 2], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * 2 * d2 * 3)
                            zero_state[0] = 1
                            test_state = cx.to_matrix(identities=2) @ h.to_matrix(identities=2) @ zero_state

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

                # Inverted basic Case
                for clev in range(0, d2):
                    for level_a in range(d1 - 1):
                        for level_b in range(level_a + 1, d1):
                            angle = np.random.uniform(0, 2 * np.pi)
                            print("Test long range Cex inverted")
                            qreg_example = QuantumRegister("reg", 4, [d1, 2, d2, 3])
                            circuit = QuantumCircuit(qreg_example)
                            h = circuit.h(2)
                            cx = circuit.cx([2, 0], [level_a, level_b, clev, angle])

                            zero_state = np.zeros(d1 * 2 * d2 * 3)
                            zero_state[0] = 1
                            test_state = cx.to_matrix(identities=2) @ h.to_matrix(identities=2) @ zero_state

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

    def test_execute_controlled(self):
        provider = MQTQuditProvider()
        backend = provider.get_backend("tnsim")

        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        h = circuit.h(1)
        x = circuit.x(0).control([1, 2], [1, 0])
        test_state = np.array([
            (0.7071067 + 0j),
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            (0.7071067 + 0j),
            0j,
            0j
        ])

        job = backend.run(circuit)
        result = job.result()
        state_vector = result.get_state_vector()

        assert np.allclose(state_vector, test_state)

        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        h = circuit.h(0)
        x = circuit.x(1).control([0, 2], [1, 0])
        test_state = np.array([
            (0.7071067 + 0j),
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            0j,
            (0.7071067 + 0j),
            0j,
            0j
        ])

        job = backend.run(circuit)
        result = job.result()
        state_vector = result.get_state_vector()

        assert np.allclose(state_vector, test_state)


"""    def test_tn_multi(self):
        self.assertFalse()

    def test_ion_ent_gates(self):
        self.assertFalse()"""
