from qiskit import *
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/computation_files")

from zne_circuits import qc_swaptest, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import ZeroNoiseExtrapolation, richardson_extrapolate

def richardson_extrapolate_get_coeffs(c) -> float:
    """
    Code taken from github.com/OpenQuantumComputing/error_mitigation/ -> zero_noise_extrapolation.py and slightly modified
    :param E: Expectation values
    :param c: Noise amplification factors
    :return: Extrapolation to the zero-limit
    """

    n = c.shape[0]
    if c.shape[0] != n:
        raise ValueError('E and c must have the same dimension.')
    if n <= 1:
        raise ValueError('the dimension of E and c must be larger than 1.')
    A = np.zeros((n, n))
    b = np.zeros((n, 1))
    # must sum to 1
    A[0, :] = 1
    b[0] = 1
    for k in range(1, n):
        A[k, :] = c ** k
    x = np.linalg.solve(A, b)
    return x

if __name__ == "__main__":

    FILENAME = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    backend_name = file["backend_name"]
    n_noise_amplification_factors = file["n_noise_amplification_factors"]
    counts = file["counts"]
    repeats, shots_per_repeat = file["repeats"], file["shots_per_repeat"]
    all_exp_vals, bare_exp_vals, mitigated_exp_vals = file["all_exp_vals"], \
                                                  file["bare_exp_vals"], \
                                                  file["mitigated_exp_vals"]

    repeats_included = 5000

    exp_vals = all_exp_vals[0:repeats_included, :]

    max_amp_factors = 10

    gammas = []
    for n in range(2,max_amp_factors+1):
        gammas.append((richardson_extrapolate_get_coeffs(np.asarray([1+2*i for i in range(n)])))[:,0])

    print(gammas)

    E = np.zeros(max_amp_factors)

    variances = np.zeros(max_amp_factors)
    variances_samples = np.zeros(max_amp_factors)

    mitigated = np.zeros(max_amp_factors)

    mitigated[0] = np.average(bare_exp_vals)

    for i in range(max_amp_factors):
        variances[i] = np.average(exp_vals[:, i] ** 2) - np.average(exp_vals[:, i]) ** 2
        E[i] = np.average(exp_vals[:, i])
        if i >= 1:
            mitigated[i] = richardson_extrapolate(E[0:i+1], np.asarray([1+2*k for k in range(i+1)]))

    for i in range(max_amp_factors):
        e = swaptest_exp_val_func(counts[i])
        variances_samples[i] = 1 - e**2

    #print(variances)
    #print(variances*8192)
    print("variances in each circuit i:\n", variances_samples)

    print("exp vals:\n", E)

    # Error tolerance
    tol = 0.01

    N_s = np.zeros(max_amp_factors-1)
    N_tot = np.zeros(max_amp_factors-1)
    sigmasquared = np.zeros(max_amp_factors-1)

    for i in range(0, max_amp_factors-1):
        sigmasquared[i] = (gammas[i]**2).dot(variances_samples[0:2+i])
        N_s[i] = (gammas[i]**2).dot(variances_samples[0:2+i]) / (0.01**2)
        N_tot[i] = N_s[i]*(i+2)

    print("N_s:\n", N_s)
    print("N_tot:\n", N_tot)
    print("sigma^2:\n", sigmasquared/8192)
    print("sqrt(sigma^2):\n", np.sqrt(sigmasquared/8192))

    repeats_s = N_s / 8192


    print("mitigated exp vals:\n", mitigated)

    var_n8 = 0
    repeats_n8 = 1000
    for i in range(repeats_n8):
        mit_n8 = richardson_extrapolate(all_exp_vals[i,0:8], np.asarray([1+2*k for k in range(8)]))
        var_n8 += (mit_n8 - mitigated[7])**2
    var_n8 = var_n8 / repeats_n8

    print("var in n=8:\n", var_n8)
    print("std in n=8:\n", np.sqrt(var_n8))

    FILENAME_DATA = abs_path + "/data_files" + "/zne_swaptest_statistics.npz"

    np.savez(FILENAME_DATA, variances=variances_samples, exp_vals=E, error_sampled_shots=N_s,
             error_sampled_total_shots=N_tot, gammas=gammas, mitigated=mitigated)

    print(richardson_extrapolate_get_coeffs(np.asarray([1,3,5,7,11,13,15])))