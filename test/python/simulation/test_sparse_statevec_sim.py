from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider

from .._qudits.test_pymisim import is_quantum_state


class TestSparseStatevecSim(TestCase):
    @staticmethod
    def test_execute():
        rng = np.random.default_rng()
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

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

            assert np.allclose(state_vector, test_state)

        # X gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            x = circuit.x(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = x.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(state_vector, test_state)

        # Z gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            z = circuit.z(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = z.to_matrix() @ h.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(state_vector, test_state)

        # S gate
        for d in [2, 3, 5, 7]:
            qreg_example = QuantumRegister("reg", 1, [d])
            circuit = QuantumCircuit(qreg_example)
            h = circuit.h(0)
            s = circuit.s(0)

            zero_state = np.zeros(d)
            zero_state[0] = 1
            test_state = s.to_matrix() @ h.to_matrix() @ zero_state

            job = backend.run(circuit)
            result = job.result()
            state_vector = result.get_state_vector()

            assert np.allclose(state_vector, test_state)

        # Rz gate
        for d in range(2, 8):
            qreg_example = QuantumRegister("reg", 1, [d])
            for level_a in range(d - 1):
                for level_b in range(level_a + 1, d):
                    circuit = QuantumCircuit(qreg_example)
                    h = circuit.h(0)
                    angle = rng.uniform(0, 2 * np.pi)
                    rz = circuit.rz(0, [level_a, level_b, angle])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = rz.to_matrix() @ h.to_matrix() @ ini_state

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
                    rh = circuit.rh(0, [level_a, level_b])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = rh.to_matrix() @ h.to_matrix() @ ini_state

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
                    angle = rng.uniform(0, 2 * np.pi)
                    phase = rng.uniform(0, 2 * np.pi)
                    r = circuit.r(0, [level_a, level_b, angle, phase])

                    ini_state = np.zeros(d)
                    ini_state[0] = 1
                    test_state = r.to_matrix() @ h.to_matrix() @ ini_state

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
                angle = rng.uniform(0, 2 * np.pi)
                vrz = circuit.virtrz(0, [level, angle])

                ini_state = np.zeros(d)
                ini_state[0] = 1
                test_state = vrz.to_matrix() @ h.to_matrix() @ ini_state

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
                for clev in range(d1):
                    for level_a in range(d2 - 1):
                        for level_b in range(level_a + 1, d2):
                            angle = rng.uniform(0, 2 * np.pi)

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
                for clev in range(d2):
                    for level_a in range(d1 - 1):
                        for level_b in range(level_a + 1, d1):
                            angle = rng.uniform(0, 2 * np.pi)
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

    @staticmethod
    def test_long_range():
        rng = np.random.default_rng()
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")
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

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

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

                            job = backend.run(circuit)
                            result = job.result()
                            state_vector = result.get_state_vector()

                            assert np.allclose(state_vector, test_state)

    @staticmethod
    def test_execute_controlled():
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(1)
        circuit.x(0).control([1, 2], [1, 0])
        test_state = np.array([(0.7071067 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (0.7071067 + 0j), 0j, 0j])

        job = backend.run(circuit)
        result = job.result()
        state_vector = result.get_state_vector()

        assert np.allclose(state_vector, test_state)

        qreg_example = QuantumRegister("reg", 3, [2, 2, 3])
        circuit = QuantumCircuit(qreg_example)
        circuit.h(0)
        circuit.x(1).control([0, 2], [1, 0])
        test_state = np.array([(0.7071067 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (0.7071067 + 0j), 0j, 0j])

        job = backend.run(circuit)
        result = job.result()
        state_vector = result.get_state_vector()

        assert np.allclose(state_vector, test_state)

    @staticmethod
    def test_mathematical_properties():
        """Test mathematical properties: norm preservation, unitarity."""
        provider = MQTQuditProvider()
        backend = provider.get_backend("sparse_statevec")

        # Test norm preservation
        qreg = QuantumRegister("test", 3, [2, 2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])
        circuit.z(2)

        job = backend.run(circuit)
        state = job.result().get_state_vector().flatten()

        # State should be normalized
        assert np.isclose(np.linalg.norm(state), 1.0)

        # Test Bell state
        qreg = QuantumRegister("bell", 2, [2, 2])
        circuit = QuantumCircuit(qreg)
        circuit.h(0)
        circuit.cx([0, 1])

        job = backend.run(circuit)
        state = job.result().get_state_vector().flatten()

        # Should have 50/50 probabilities for |00⟩ and |11⟩
        assert np.isclose(np.abs(state[0]) ** 2, 0.5)
        assert np.isclose(np.abs(state[3]) ** 2, 0.5)
        assert np.isclose(np.abs(state[1]) ** 2, 0.0)
        assert np.isclose(np.abs(state[2]) ** 2, 0.0)
        assert is_quantum_state(state)
