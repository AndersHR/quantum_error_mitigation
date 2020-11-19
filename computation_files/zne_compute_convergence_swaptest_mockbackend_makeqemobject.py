from qiskit import *
from qiskit.test.mock import FakeVigo
import numpy as np

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)

from zne_swaptest_circuit import qc_swaptest
from zero_noise_extrapolation_cnot import *

def swaptest_exp_val_func(counts: dict):
    exp_val = 0
    tot = 0
    for key in counts.keys():
        if key == "0":
            eigenval = 1
        elif key == "1":
            eigenval = -1
        else:
            print("ERROR")
            return
        exp_val += counts[key] * eigenval
        tot += counts[key]
    return exp_val / tot

if __name__ == "__main__":
    FILENAME_OBJ = abs_path + "/data_files" + "/zne_convergence_mockbackend_obj.npz"
    FILENAME_DATA = abs_path + "/data_files" + "/zne_convergence_mockbackend.npz"

    mock_backend = FakeVigo()
    n_noise_amplification_factors = 20

    shots_per_repeat = 8192
    repeats = 500

    qem = ZeroNoiseExtrapolation(qc_swaptest, exp_val_func=swaptest_exp_val_func, backend=mock_backend,
                                 n_amp_factors=n_noise_amplification_factors, shots=shots_per_repeat)

    qem.mitigate(repeats=repeats)

    np.savez(FILENAME_OBJ, qem=qem)

    np.savez(FILENAME_DATA, backend_name=mock_backend.name(), n_noise_amplification_factors=n_noise_amplification_factors,
             repeats=repeats, shots_per_repeat=shots_per_repeat, depths=qem.depths, pauli_twirl=qem.pauli_twirl,
             counts=qem.counts, all_exp_vals=qem.all_exp_vals, mitigated_exp_vals=qem.mitigated_exp_vals,
             bare_exp_vals=qem.bare_exp_vals)
