import itertools

import matplotlib.pyplot as plt
import numpy as np

from mqt.qudits.qudit_circuits.circuit import QuantumCircuit


class HistogramWithErrors:
    def __init__(self, labels, counts, errors, title="Simulation"):
        self.labels = labels
        self.counts = counts
        self.errors = errors
        self.title = title

    def generate_histogram(self):
        plt.bar(self.labels, self.counts, yerr=self.errors, capsize=5, color="b", alpha=0.7, align="center")
        plt.xlabel("States")
        plt.ylabel("Pr")
        plt.title(self.title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def save_to_png(self, filename):
        plt.bar(self.labels, self.counts, yerr=self.errors, capsize=5, color="b", alpha=0.7, align="center")
        plt.xlabel("States")
        plt.ylabel("Pr")
        plt.title(self.title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(filename, format="png")
        plt.close()


def plot_histogram(result: np.ndarray, circuit: QuantumCircuit, errors=None) -> None:
    result = np.squeeze(result).tolist()
    if errors is None:
        errors = len(result) * [0]
    dimensions = circuit.dimensions
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

    result = [abs(coeff) for coeff in result]
    h_plotter = HistogramWithErrors(string_states, result, errors, title="Simulation")
    h_plotter.generate_histogram()
