from qiskit import *
from qiskit.test.mock import FakeVigo
import matplotlib.pyplot as plt

if __name__ == "__main__":
    q = QuantumRegister(4, name="q")
    c = ClassicalRegister(4, name="c")

    qc = QuantumCircuit(q, c)

    qc.h(0)
    qc.x(2)
    qc.h(2)

    qc.barrier()

    qc.cx(2, 3)
    qc.cx(1, 2)
    qc.cx(0, 3)
    qc.cx(0, 1)

    qc.barrier()

    qc.measure((0, 1, 2, 3), (0, 1, 2, 3))

    qc.draw(output="mpl")

    plt.tight_layout()

    plt.savefig("qcircuit_example.pdf")
    plt.show()

    qc_transpiled = transpile(qc, backend=FakeVigo(), optimization_level=1)

    qc_transpiled.draw(output="mpl")

    plt.tight_layout()

    plt.savefig("qcircuit_example_transpiled.pdf")
    plt.show()