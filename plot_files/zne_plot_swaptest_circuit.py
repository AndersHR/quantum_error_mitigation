from qiskit import *
import matplotlib.pyplot as plt

from os.path import dirname
from sys import path

abs_path = dirname(dirname(__file__))
path.append(abs_path + "/data_files/")

from zne_swaptest_circuit import qc_swaptest

if __name__ == "__main__":

    qc_swaptest.draw(output="mpl")

    plt.tight_layout()

    plt.savefig("zne_swaptest_circuit.pdf")

    plt.show()