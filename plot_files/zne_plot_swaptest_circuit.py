from qiskit import *
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path + "/data_files/")

from zne_swaptest_circuit import qc_swaptest, qc_swaptest_transpiled, qc_swaptest_toffoli

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