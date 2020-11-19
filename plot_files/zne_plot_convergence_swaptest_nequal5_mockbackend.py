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

if __name__ == "__main__":

    FILENAME = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    file = np.load(FILENAME, allow_pickle=True)

    backend_name = file["backend_name"]
    n_noise_amplification_factors = file["n_noise_amplification_factors"]
    counts = file["counts"]
    repeats, shots_per_repeat = file["repeats"], file["shots_per_repeat"]
    all_exp_vals, bare_exp_vals, mitigated_exp_vals = file["all_exp_vals"],\
                                                      file["bare_exp_vals"],\
                                                      file["mitigated_exp_vals"]

    n_amp_factors_included = 5

    dx = 5
    max_repeats = 600
    num = max_repeats // dx

    rep = np.linspace(1,max_repeats,num=num, endpoint=True,dtype=int)
    tot_shots = rep * 8192

    amp_factors = np.asarray([1+2*i for i in range(n_amp_factors_included)])

    averaged_exp_vals = np.zeros((n_amp_factors_included, num))

    mitigated = np.zeros(num)

    for x, r in enumerate(rep):
        for i in range(n_amp_factors_included):
            averaged_exp_vals[i,x] = np.average(all_exp_vals[0:r,i])
        mitigated[x] = richardson_extrapolate(averaged_exp_vals[:,x],amp_factors)

    plt.plot(tot_shots, np.zeros(num) + 0.5, 'k', linestyle="dashed", label=r"E$^*$, true exp val")

    plt.plot(tot_shots, mitigated, linestyle="solid", label=r"$E[1,3,5,7,9]_N$, mitigated")

    plt.plot(tot_shots, averaged_exp_vals[0, :], linestyle='dotted', label=r"E$[1\,\lambda]_N$, bare circuit")
    plt.plot(tot_shots, averaged_exp_vals[1, :], linestyle='dashdot', label=r"E$[3\lambda]_N$")
    plt.plot(tot_shots, averaged_exp_vals[2, :], linestyle=(0,(2,3,1,3,1,3)), label=r"E$[5 \lambda]_N$")
    plt.plot(tot_shots, averaged_exp_vals[3, :], linestyle=(0,(5,1,2,1)), label=r"E$[7 \lambda]_N$")
    plt.plot(tot_shots, averaged_exp_vals[4, :], linestyle=(0,(3,1,1,1,1,1)), label=r"E$[9 \lambda]_N$")

    plt.xlabel(r"$N$, shots included in averages", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=13, bbox_to_anchor=(1.02, 1.), loc=2, borderaxespad=0.)
    plt.tight_layout()

    plt.show()
