from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit


def remap_result(result: np.ndarray, circuit: QuantumCircuit) -> np.ndarray:
    new_result = result.copy()
    if circuit.mappings:
        permutation = np.eye(circuit.dimensions[0])[:, circuit.mappings[0]]
        for i in range(1, len(circuit.mappings)):
            permutation = np.kron(permutation, np.eye(circuit.dimensions[i])[:, circuit.mappings[i]])
        return new_result @ np.linalg.inv(permutation)
    return new_result


class HistogramWithErrors:
    def __init__(self, labels, counts, errors, title="", xlabel="Labels", ylabel="Counts") -> None:
        self.labels = labels
        self.counts = counts
        self.errors = errors

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def generate_histogram(self) -> None:
        plt.bar(
            self.labels,
            self.counts,
            yerr=self.errors,
            capsize=5,
            color="b",
            alpha=0.7,
            align="center",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def save_to_png(self, filename) -> None:
        plt.bar(
            self.labels,
            self.counts,
            yerr=self.errors,
            capsize=5,
            color="b",
            alpha=0.7,
            align="center",
        )
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(filename, format="png")
        plt.close()


def state_labels(circuit):
    dimensions = circuit.dimensions  # reversed(circuit.dimensions)
    # it was in the order of the DD simulation now it is in circuit order
    logic = []
    lut = []
    for d in dimensions:
        logic.append(list(range(d)))

    for element in itertools.product(*logic):
        lut.append(list(element))

    string_states = []
    for item in lut:
        s = ""
        for state in item:
            s += str(state)
        string_states.append(s)

    return string_states


def plot_state(state_vector: np.ndarray, circuit: QuantumCircuit, errors=None) -> np.ndarray:
    labels = state_labels(circuit)

    state_vector_list = np.squeeze(state_vector).tolist()
    counts = [abs(coeff) for coeff in state_vector_list]
    counts = remap_result(counts, circuit)

    h_plotter = HistogramWithErrors(labels, counts, errors, title="Simulation", xlabel="States", ylabel="Sqrt(Pr)")
    h_plotter.generate_histogram()
    return counts


def plot_counts(measurements, circuit: QuantumCircuit) -> np.ndarray:
    labels = state_labels(circuit)
    counts = [measurements.count(i) for i in range(len(labels))]
    counts = remap_result(counts, circuit)

    errors = len(labels) * [0]

    h_plotter = HistogramWithErrors(labels, counts, errors, title="Simulation", xlabel="States", ylabel="Counts")
    h_plotter.generate_histogram()
    return counts
