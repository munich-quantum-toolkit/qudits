from __future__ import annotations

from collections import defaultdict
from unittest import TestCase

import numpy as np
import pytest

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation.noise_tools import Noise, NoiseModel, NoisyCircuitFactory, SubspaceNoise


def rand_0_5() -> int:
    rng = np.random.default_rng()
    return int(rng.integers(0, 6))


class TestNoisyCircuitFactoryPhysical(TestCase):
    def setUp(self) -> None:
        sub1 = SubspaceNoise(0.999, 0.999, (0, 1))
        sub2 = SubspaceNoise(0.0, 0.999, [(1, 2), (2, 3)])
        sub3 = SubspaceNoise(0.999, 0.0, [(1, 2), (2, 3)])
        dynamic_assigned = SubspaceNoise(0.999, 0.0, [])

        # Add errors to noise_tools model

        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate_matrix
        noise_model.add_all_qudit_quantum_error(sub1, ["csum"])
        # Local Gates
        noise_model.add_quantum_error_locally(sub1, ["z", "ms"])
        noise_model.add_quantum_error_locally(sub2, ["x", "h"])
        noise_model.add_quantum_error_locally(dynamic_assigned, ["rh", "rz", "rxy", "cx"])
        noise_model.add_quantum_error_locally(sub3, ["s"])
        self.noise_model = noise_model

        assert set(noise_model.basis_gates) == {"csum", "z", "ms", "rh", "h", "rz", "rxy", "x", "s", "cx"}

    def test_generate_circuit(self):
        qreg_example = QuantumRegister("reg", 6, 6 * [5])
        circ = QuantumCircuit(qreg_example)
        choice = rand_0_5()
        x = circ.x(choice)
        x.control([int(np.mod(choice + 1, 5))], [2])
        circ.rz(rand_0_5(), [0, 2, np.pi / 13])
        circ.cx([3, 4], [0, 3, 0, np.pi / 12])
        circ.x(rand_0_5()).dag()
        circ.s(rand_0_5())
        circ.z(rand_0_5())
        circ.csum([5, 1])
        circ.virtrz(rand_0_5(), [0, np.pi / 13]).dag()
        circ.virtrz(rand_0_5(), [1, -np.pi / 8])
        circ.csum([2, 5]).dag()
        circ.x(rand_0_5()).dag()
        circ.z(rand_0_5()).dag()
        circ.h(rand_0_5())
        circ.rz(rand_0_5(), [3, 4, np.pi / 13]).dag()
        circ.h(rand_0_5()).dag()
        circ.r(rand_0_5(), [0, 1, np.pi / 5 + np.pi, np.pi / 7])
        circ.rh(rand_0_5(), [1, 3])
        circ.r(rand_0_5(), [0, 4, np.pi, np.pi / 2]).dag()
        circ.r(rand_0_5(), [0, 3, np.pi / 5, np.pi / 7])
        circ.cx([1, 2], [0, 1, 1, np.pi / 2]).dag()
        circ.csum([0, 1])

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        instructions_new = new_circ.instructions

        tag_counts_list1: dict[str, int] = defaultdict(int)
        tag_counts_list2: dict[str, int] = defaultdict(int)
        insts = 0
        insts_new = 0
        for gate in instructions_og:
            insts += 1
            tag_counts_list1[gate.qasm_tag] += 1

        for gate in instructions_new:
            insts_new += 1
            tag_counts_list2[gate.qasm_tag] += 1

        keys_to_check = ["noisex", "noisey", "virtrz"]
        valid_stochasticity = True
        # Iterate over all keys
        for key in tag_counts_list1.keys() | tag_counts_list2.keys():
            if key in keys_to_check:
                if tag_counts_list1.get(key, 0) > tag_counts_list2.get(key, 0):
                    valid_stochasticity = False
                    print(key, "error")
            elif tag_counts_list1.get(key, 0) != tag_counts_list2.get(key, 0):
                valid_stochasticity = False
                print(key, "error")

        assert valid_stochasticity
        assert insts == circ.number_gates
        assert insts_new == new_circ.number_gates
        assert len(instructions_new) > len(instructions_og)

    def test_generate_circuit_isolated1(self):
        qreg_example = QuantumRegister("reg", 4, [5, 5, 5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        circ.z(1)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["z", "z"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"z", "virtrz", "noisex", "noisey"}
        for inst in new_circ.instructions:
            assert inst.target_qudits in {0, 1}
            if inst.qasm_tag in {"noisex", "noisey"}:
                assert inst.lev_a == 0
                assert inst.lev_b == 1
            if inst.qasm_tag == "virtz":
                assert inst.lev_a != 0
                assert inst.lev_b != 1

    def test_generate_circuit_isolated2(self):
        qreg_example = QuantumRegister("reg", 4, [5, 5, 5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.x(0)
        circ.x(1)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["x", "x"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"x", "virtrz"}
        for inst in new_circ.instructions:
            assert inst.target_qudits in {0, 1}

    def test_generate_circuit_isolated3(self):
        qreg_example = QuantumRegister("reg", 4, [5, 5, 5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.s(1)
        circ.s(2)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["s", "s"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"s", "noisex", "virtrz", "noisey"}
        for inst in new_circ.instructions:
            assert inst.target_qudits in {1, 2}
            if inst.qasm_tag in {"noisex", "noisey"}:
                assert (inst.lev_a == 2 and inst.lev_b == 3) or (inst.lev_a == 1 and inst.lev_b == 2)

    @staticmethod
    def test_str():
        sub1 = SubspaceNoise(0.999, 0.999, (0, 1))
        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(sub1, ["z"])
        assert "Gate: z, Mode: local, SubspaceNoise: 0<->1:0.999 0.999," in str(noise_model)

    @staticmethod
    def test_error():
        noise_model = NoiseModel()

        with pytest.raises(ValueError, match="The levels in the subspace noise should be different!"):
            SubspaceNoise(0.999, 0.999, (0, 0))

        err2 = SubspaceNoise(0.999, 0.999, (0, 1))
        noise_model.add_quantum_error_locally(err2, ["z"])
        with pytest.raises(ValueError, match="The same level physical noise is defined for multiple times!"):
            noise_model.add_quantum_error_locally(err2, ["z"])

    @staticmethod
    def test_invalid_level():
        err = SubspaceNoise(0.999, 0.999, (8, 9))
        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(err, ["z"])
        qreg_example = QuantumRegister("reg", 6, 6 * [5])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        factory = NoisyCircuitFactory(noise_model, circ)
        with pytest.raises(IndexError, match=r"Subspace levels exceed qudit dimensions.*"):
            factory.generate_circuit()

    @staticmethod
    def test_no_dephasing():
        err = SubspaceNoise(0.999, 0.999, (0, 1))
        noise_model = NoiseModel()
        noise_model.add_quantum_error_locally(err, ["z"])
        qreg_example = QuantumRegister("reg", 2, 2 * [2])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        factory = NoisyCircuitFactory(noise_model, circ)
        factory.generate_circuit()

    @staticmethod
    def test_invalid_gate():
        err = SubspaceNoise(0.999, 0.999, (0, 1))
        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error_on_control(err, ["z"])
        qreg_example = QuantumRegister("reg", 2, 2 * [2])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        factory = NoisyCircuitFactory(noise_model, circ)
        with pytest.raises(ValueError, match=r".* is incompatible for the desired operation."):
            factory.generate_circuit()

        noise_model = NoiseModel()
        noise_model.add_nonlocal_quantum_error(err, ["z"])
        factory = NoisyCircuitFactory(noise_model, circ)
        with pytest.raises(ValueError, match=r"Nonlocal mode not applicable for gate type: .*"):
            factory.generate_circuit()

    @staticmethod
    def test_invalid_mode():
        err = SubspaceNoise(0.999, 0.999, (0, 1))
        noise_model = NoiseModel()
        noise_model._add_quantum_error(err, ["z"], "error_mode")  # noqa: SLF001
        qreg_example = QuantumRegister("reg", 2, 2 * [2])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        factory = NoisyCircuitFactory(noise_model, circ)
        with pytest.raises(ValueError, match="Unknown mode: error_mode"):
            factory.generate_circuit()

    def test_generate_circuit_isolated4(self):
        qreg_example = QuantumRegister("reg", 4, [5, 5, 5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.ms([1, 2], [np.pi / 4])
        circ.ms([1, 2], [np.pi / 3])

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["ms", "ms"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"ms", "noisex", "virtrz", "noisey"}
        for inst in new_circ.instructions:
            if isinstance(inst.target_qudits, list):
                for t in inst.target_qudits:
                    assert t in {1, 2}
            else:
                assert inst.target_qudits in {1, 2}
            if inst.qasm_tag in {"noisex", "noisey"}:
                assert inst.lev_a == 0
                assert inst.lev_b == 1

    @staticmethod
    def test_generate_circuit_isolated5():
        dynamic_assigned = SubspaceNoise(0.999, 0.0, [])
        # Add errors to noise_tools model
        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        noise_model.add_nonlocal_quantum_error_on_target(dynamic_assigned, ["rh", "rz", "rxy", "cx"])
        qreg_example = QuantumRegister("reg", 4, [5, 5, 5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.r(1, [0, 1, np.pi, np.pi / 2]).dag()
        circ.r(1, [2, 3, np.pi, np.pi / 2]).control([2], [1])
        circ.cx([0, 1])

        factory = NoisyCircuitFactory(noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 3  # namely r, r, cx
        assert [i.qasm_tag for i in instructions_og] == ["rxy", "rxy", "cx"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"rxy", "cx", "noisex", "virtrz", "noisey"}
        for inst in new_circ.instructions:
            if isinstance(inst.target_qudits, list):
                for t in inst.target_qudits:
                    assert t in {0, 1}
            else:
                assert inst.target_qudits in {0, 1}
            if inst.qasm_tag in {"noisex", "noisey"}:
                assert (inst.lev_a == 0 and inst.lev_b == 1) or (inst.lev_a == 2 and inst.lev_b == 3)

    @staticmethod
    def test_invalid_dynamic_levels_construction():
        err = SubspaceNoise(0.999, 0.999, (0, 1))
        with pytest.raises(ValueError, match="Negative keys are for the dynamic assignment"):
            err.add_noise(-2, -1, Noise(0.99, 0.0))

        err = SubspaceNoise(0.999, 0.999, [])
        with pytest.raises(ValueError, match="Negative keys are already present"):
            err.add_noise(1, 2, Noise(0.99, 0.0))
