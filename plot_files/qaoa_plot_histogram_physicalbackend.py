import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from QAOA import QAOA, create_good_solution_counts

if __name__ == "__main__":
    GRAPH = nx.Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 4], [2, 4], [0, 3], [3, 4]])

    FILENAME = abs_path + "/data_files" + "/qaoa_histogram_physicalbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    counts = file["counts"][()]
    shots = file["shots"]
    iterations, function_evaluations = file["iterations"], file["function_evaluations"]
    initial_gamma, initial_beta = file["initial_gamma"], file["initial_beta"]
    optimized_gamma, optimized_beta = file["optimized_gamma"], file["optimized_beta"]

    from qiskit.visualization import plot_histogram

    optimal, good, _ = create_good_solution_counts(QAOA(graph=GRAPH))
    optimal_bitstrings = optimal.keys()
    good_bitstrings = good.keys()

    print(counts)

    y = np.zeros(2**5)
    x = []

    y_optimal = np.zeros(2**5)
    y_good = np.zeros(2**5)

    for n in range(2**5):
        hexkey = hex(n)
        if hexkey in counts.keys():
            y[n] = counts[hexkey] / shots

        bitstr = "{0:05b}".format(n)
        x.append(bitstr[::-1])

        if bitstr in optimal_bitstrings:
            y_optimal[n] = 1.0
        if bitstr in good_bitstrings and not bitstr in optimal_bitstrings:
            y_good[n] = 1.0

    plt.bar(x, y_good, align="center", alpha=0.2, label=r"$z$  s.t. $C(z) = 5$")
    plt.bar(x, y_optimal, align="center", alpha=0.2, label=r"$z$  s.t. $C(z) = 6$")
    plt.bar(x, y, label=r"Estimated probabilities $p_z$") # $p_z = |<\gamma,\beta\,|\,z>|^2$

    plt.ylim(0, 0.5)
    plt.xticks(rotation="vertical")

    plt.ylabel(r"$\frac{n_z}{N}$", labelpad=10.0, rotation="horizontal", fontsize=15)
    plt.xlabel(r"computational basis states $|z>$", fontsize=12)

    plt.tight_layout()

    plt.legend()

    plt.show()