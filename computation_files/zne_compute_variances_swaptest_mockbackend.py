from qiskit import *
from qiskit.test.mock import FakeVigo
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from zne_swaptest_circuit import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import *

def compute_richardson_extrapolation_coefficients(c: np.ndarray) -> np.ndarray:
    n = np.shape(c)[0]
    A = zeros((n, n))
    b = zeros((n, 1))
    # must sum to 1
    A[0, :] = 1
    b[0] = 1
    for k in range(1, n):
        A[k, :] = c ** k
    gamma = solve(A, b)
    return gamma

def swaptest_variance():
    return

if __name__ == "__main__":
    FILENAME = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    FILENAME_OBJ = abs_path + "/data_files" + "/zne_convergence_mockbackend_obj.npz"

    mock_backend=FakeVigo()

    file_obj = np.load(FILENAME_OBJ, allow_pickle=True)
    qem = file_obj["qem"][()]

    file_data = np.load(FILENAME, allow_pickle=True)

    repeats = file_data["repeats"]
    all_exp_vals = file_data["all_exp_vals"]

    n_max = 10
    gammas = []
    for n in range(2, n_max+1):
        amp_factors = np.asarray([1+2*i for i in range(n)])
        gammas.append(compute_richardson_extrapolation_coefficients(amp_factors))
    print(gammas)

    circuit_variances = np.zeros(n_max)

    for n in range(n_max):
        e = np.average(all_exp_vals[:,n])
        e2 = np.average(all_exp_vals[:,n]**2)
        circuit_variances[n] = e2 - e**2
    print(np.sqrt(circuit_variances))

    n = 8

    print(np.sqrt(circuit_variances[0:8].dot(gammas[n-2]**2)))

    print("done")