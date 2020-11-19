from qiskit import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from QAOA import QAOA

if __name__ == "__main__":
    GRAPH = nx.Graph([[0, 1], [0, 2], [0, 4], [1, 2], [1, 4], [2, 4], [0, 3], [3, 4]])

    FILENAME = abs_path + "/data_files" + "/qaoa_landscape.npz"

    sim_backend = Aer.get_backend("qasm_simulator")
    shots = 8192

    qaoa = QAOA(graph=GRAPH, backend=sim_backend, shots=shots)

    num = 20

    gammas = np.linspace(0, np.pi, num=num)
    betas = np.linspace(0, np.pi, num=num)

    costs = np.zeros((num,num))

    for i in range(num):
        for j in range(num):
            result = qaoa.execute_circuit([gammas[i]],[betas[j]])
            costs[i,j] = qaoa.compute_exp_val(result)

    np.savez(FILENAME, GRAPH=GRAPH, shots=shots, gammas=gammas, betas=betas, costs=costs)