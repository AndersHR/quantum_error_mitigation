import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from QAOA import QAOA

if __name__ == "__main__":
    FILENAME = abs_path + "/data_files" + "/qaoa_landscape.npz"

    file = np.load(FILENAME)

    GRAPH, shots, gammas, betas, costs = file["GRAPH"],file["shots"],file["gammas"],file["betas"],file["costs"]

    plt.xlabel(r"$\beta$", fontsize=12)
    plt.ylabel(r"$\gamma$", fontsize=12)

    betas, gammas = np.meshgrid(gammas, betas)

    plt.pcolormesh(betas, gammas, costs, shading="auto", vmin=1.5, vmax=5.0)

    cbar = plt.colorbar()
    cbar.set_label(r"$F(\gamma, \beta)$",size=12)
    cbar.ax.tick_params(labelsize=10, labeltop=True, labelbottom=True)

    plt.tight_layout()

    plt.show()