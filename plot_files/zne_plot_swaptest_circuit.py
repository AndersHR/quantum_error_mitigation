from qiskit import *
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path)
path.append(abs_path + "/computation_files/")

from zne_circuits import qc_swaptest, qc_swaptest_transpiled, qc_swaptest_toffoli, swaptest_exp_val_func
from zero_noise_extrapolation_cnot import *

if __name__ == "__main__":

    qc_swaptest_toffoli.draw(output="mpl")

    plt.tight_layout()

    plt.savefig(abs_path + "/figures" + "/zne_swaptest_circuit_toffoli.pdf")

    plt.show()

    qc_swaptest.draw(output="mpl")

    plt.tight_layout()

    plt.savefig(abs_path + "/figures" + "/zne_swaptest_circuit.pdf")

    plt.show()

    qc_swaptest_transpiled.draw(output="mpl")

    plt.tight_layout()

    plt.savefig(abs_path + "/figures" + "/zne_swaptest_circuit_transpiled.pdf")

    plt.show()

    qem = ZeroNoiseExtrapolation(qc_swaptest, None, backend=None, n_amp_factors=2)

    qc_amplified = qem.noise_amplify_and_pauli_twirl_cnots(qc_swaptest, amp_factor=3, pauli_twirl=False)

    qc_amplified.draw(output="mpl")

    plt.tight_layout()

    plt.show()