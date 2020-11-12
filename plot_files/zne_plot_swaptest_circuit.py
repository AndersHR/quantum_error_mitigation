from qiskit import *
import matplotlib.pyplot as plt

def add_toffoli(qc, c1, c2, t):
    qc.barrier()
    qc.h(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(t)
    qc.cx(c2, t)
    qc.tdg(t)
    qc.cx(c1, t)
    qc.t(c2)
    qc.t(t)
    qc.cx(c1, c2)
    qc.h(t)
    qc.t(c1)
    qc.tdg(c2)
    qc.cx(c1, c2)


def add_swap(qc, q1, q2, p):
    add_toffoli(qc, p, q1, q2)
    add_toffoli(qc, p, q2, q1)
    add_toffoli(qc, p, q1, q2)


def create_swap_circuit(state1_qubits, state2_qubits, probe, n_qubits):
    qc = QuantumCircuit(n_qubits, 1)

    qc.h(probe)

    qc.h(state1_qubits[0])
    for i in range(1, len(state1_qubits)):
        qc.cx(state1_qubits[0], state1_qubits[i])

    for i in range(len(state1_qubits)):
        add_swap(qc, state1_qubits[i], state2_qubits[i], probe)

    qc.barrier()

    qc.h(probe)

    qc.measure(probe, 0)

    return qc


def swap_exp_val_func(counts):
    exp_val = 0
    tot = 0
    for key in counts.keys():
        if key == "0":
            eigenval = 1
        elif key == "1":
            eigenval = -1
        else:
            print("ERROR")
        exp_val += counts[key] * eigenval
        tot += counts[key]
    return exp_val / tot

if __name__ == "__main__":
    qc_swap = create_swap_circuit([1], [2], 0, 3)

    qc_swap.draw(output="mpl")

    plt.tight_layout()

    plt.savefig("zne_swaptest_circuit.pdf")

    plt.show()