import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from QAOA import QAOA, create_good_solution_counts

if __name__ == "__main__":
    GRAPH = nx.Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 3], [2, 3], [0, 3], [3, 4]])

    FILENAME_SIM = abs_path + "/data_files" + "/qaoa_histogram_simulatorbackend.npz"
    FILENAME_PHYS = abs_path + "/data_files" + "/qaoa_histogram_physicalbackend.npz"

    file_sim = np.load(FILENAME_SIM, allow_pickle=True)
    file_phys = np.load(FILENAME_PHYS, allow_pickle=True)

    counts_sim = file_sim["counts"][()]
    shots_sim = file_sim["shots"]

    print("Sim backend solution:")
    print("gamma = ",file_sim["optimized_gamma"])
    print("beta = ",file_sim["optimized_beta"])
    print("F = ",file_sim["optimized_exp_val"])
    print("iter = ",file_sim["iterations"])

    print("Mock backend solution:")
    print("gamma = ", file_phys["optimized_gamma"])
    print("beta = ", file_phys["optimized_beta"])
    print("F = ",file_phys["optimized_exp_val"])
    print("iter = ",file_phys["iterations"])

    counts_phys = file_phys["counts"][()]
    shots_phys = file_phys["shots"]

    from qiskit.visualization import plot_histogram

    optimal, good, _ = create_good_solution_counts(QAOA(graph=GRAPH))
    optimal_bitstrings = optimal.keys()
    good_bitstrings = good.keys()

    y_sim = np.zeros(2**5)
    y_phys = np.zeros(2**5)
    x = []

    y_optimal = np.zeros(2**5)
    y_good = np.zeros(2**5)

    for n in range(2**5):
        hexkey = hex(n)
        if hexkey in counts_sim.keys():
            y_sim[n] = counts_sim[hexkey] / shots_sim
        if hexkey in counts_phys.keys():
            y_phys[n] = counts_phys[hexkey] / shots_phys

        bitstr = "{0:05b}".format(n)
        x.append(bitstr[::-1])

        if bitstr in optimal_bitstrings:
            y_optimal[n] = 1.0
        if bitstr in good_bitstrings and not bitstr in optimal_bitstrings:
            y_good[n] = 1.0

    plt.bar(x, y_good, align="center", alpha=0.2, label=r"$z$  s.t. $C(z) = 5$")
    plt.bar(x, y_optimal, align="center", alpha=0.2, label=r"$z$  s.t. $C(z) = 6$")
    plt.bar(x, y_sim, fill=False, linestyle="-", linewidth=1.0, label=r"Mmt. counts $n_z$/$N$, from QAOA w/ ideal simulator backend") # $p_z = |<\gamma,\beta\,|\,z>|^2$
    plt.bar(x, y_phys, fill=False, linestyle="--", linewidth=1.0, label=r"Mmt. counts $n_z$/$N$, from QAOA w/ mock backend FakeVigo")

    plt.ylim(0, 0.25)
    plt.xticks(rotation="vertical", fontsize=13)
    plt.yticks(fontsize=13)

    plt.ylabel(r"$\frac{n_z}{N}$", labelpad=10.0, rotation="horizontal", fontsize=18)
    plt.xlabel(r"computational basis states $|z>$", fontsize=14)

    plt.tight_layout()

    plt.legend()

    plt.show()

    import qiskit
    print(qiskit.__qiskit_version__)