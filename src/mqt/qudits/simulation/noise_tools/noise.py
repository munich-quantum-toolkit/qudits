from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Noise:
    probability_depolarizing: float
    probability_dephasing: float


class NoiseModel:
    def __init__(self) -> None:
        self.quantum_errors = {}

    def add_recurrent_quantum_error_locally(self, noise, gates, qudits) -> None:
        set_of_qudits = tuple(sorted(qudits))
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate][set_of_qudits] = noise

    def add_quantum_error_locally(self, noise, gates) -> None:
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate]["local"] = noise

    def add_all_qudit_quantum_error(self, noise, gates) -> None:
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate]["all"] = noise

    def add_nonlocal_quantum_error(self, noise, gates) -> None:
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate]["nonlocal"] = noise
        # self.add_quantum_error_locally(noise, gates, crtl_qudits + target_qudits)

    def add_nonlocal_quantum_error_on_target(self, noise, gates) -> None:
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate]["target"] = noise

    def add_nonlocal_quantum_error_on_control(self, noise, gates) -> None:
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            self.quantum_errors[gate]["control"] = noise

    @property
    def basis_gates(self):
        return list(self.quantum_errors.keys())

    def __str__(self) -> str:
        info_str = "NoiseModel Info:\n"
        for gate_errors, qudits in self.quantum_errors.items():
            info_str += f"Qudits: {qudits}, Gates: {', '.join(gate_errors.keys())}, Error: {gate_errors}\n"
        return info_str
