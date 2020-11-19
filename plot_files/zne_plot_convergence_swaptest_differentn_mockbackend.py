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

    n_amp_factors_included = 10

    dx = 10
    max_repeats = 5000
    num = max_repeats // dx

    rep = np.linspace(1,max_repeats,num=num, endpoint=True,dtype=int)
    tot_shots = rep * 8192

    amp_factors = np.asarray([1+2*i for i in range(n_amp_factors_included)])

    averaged_exp_vals = np.zeros((n_amp_factors_included, num))

    all_mitigated_exp_vals = np.zeros((n_amp_factors_included,num))

    for x, r in enumerate(rep):
        for i in range(n_amp_factors_included):
            averaged_exp_vals[i,x] = np.average(all_exp_vals[0:r,i])
        all_mitigated_exp_vals[0, x] = averaged_exp_vals[0, x]
        for j in range(2,n_amp_factors_included+1):
            all_mitigated_exp_vals[j-1,x] = richardson_extrapolate(averaged_exp_vals[0:j,x],amp_factors[0:j])

    plt.plot(tot_shots, np.zeros(num) + 0.5, 'k', linestyle="dashed", label=r"E$^*$, true exp val")

    #plt.plot(tot_shots, all_mitigated_exp_vals[0], linestyle="solid", label="$E[1]$, bare circuit")
    plt.plot(tot_shots, all_mitigated_exp_vals[1], linestyle=(0,(3,1,1,1,1,1)), label="$E[1,3]$")
    plt.plot(tot_shots, all_mitigated_exp_vals[2], linestyle="dotted", label="$E[1,3,5]$")
    plt.plot(tot_shots, all_mitigated_exp_vals[4], linestyle="dashdot", label="$E[1,3,5,7,9]$")
    plt.plot(tot_shots, all_mitigated_exp_vals[7], linestyle="solid", label="$E[1,3,5,7,9,11,13,15]$")

    #plt.plot(tot_shots, averaged_exp_vals[0, :], linestyle='dotted', label="E$[1\,\lambda]$, bare circuit")
    #plt.plot(tot_shots, averaged_exp_vals[1, :], linestyle='dashdot', label="E$[3\lambda]$")
    #plt.plot(tot_shots, averaged_exp_vals[2, :], linestyle=(0,(2,3,1,3,1,3)), label="E$[5 \lambda]$")
    #plt.plot(tot_shots, averaged_exp_vals[3, :], linestyle=(0,(5,1,2,1)), label="E$[7 \lambda]$")
    #plt.plot(tot_shots, averaged_exp_vals[4, :], linestyle=(0,(3,1,1,1,1,1)), label="E$[9 \lambda]$")

    plt.xlabel("$N$, shots included in averages")

    plt.ylim(0.45,0.57)

    #plt.legend(bbox_to_anchor=(1.02, 1.), loc=2, borderaxespad=0.)

    plt.legend(loc="upper right", fontsize=12)

    plt.tight_layout()

    plt.show()

    print(all_mitigated_exp_vals[:,-1])

    xticks = [i+1 for i in range(n_amp_factors_included)]

    plt.plot(xticks, all_mitigated_exp_vals[:,-1].transpose(),
             'ro--', label=r"$E[1,\dots,2n-1]$")

    plt.plot(xticks, np.zeros(np.shape(xticks)[0]) + 0.5, 'g--', label=r"$E^*$")

    plt.xticks(ticks=xticks)
    plt.xlabel(r"$n$, number of amplification factors included", fontsize=12)

    plt.legend(fontsize=12)

    plt.tight_layout()

    plt.show()
