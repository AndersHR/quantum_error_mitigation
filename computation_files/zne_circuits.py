from qiskit import *
from qiskit.test.mock import FakeVigo

mock_backend = FakeVigo()

def create_simple_circuit():
    qc = QuantumCircuit(2,2)

    qc.h(0)
    qc.cx(0,1)
    qc.measure([0,1],[0,1])

    return qc

def simple_circuit_exp_val_func(counts: dict):
    e = 0
    tot = 0
    for key in counts.keys():
        if key == "00" or key == "11":
            eigenval = +1
        else:
            eigenval = -1
        e += eigenval*counts[key]
        tot += counts[key]
    return e / tot


def create_swaptest_toffoli_circuit():
    qc = QuantumCircuit(3, 1)
    qc.h(0)
    qc.h(1)

    qc.toffoli(0, 1, 2)
    qc.toffoli(0, 2, 1)
    qc.toffoli(0, 1, 2)

    qc.h(0)

    qc.measure(0, 0)

    return qc


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


qc_simple = create_simple_circuit()

qc_swaptest_toffoli = create_swaptest_toffoli_circuit()

qc_swaptest = create_swap_circuit([1], [2], 0, 3)

qc_swaptest_transpiled = transpile(qc_swaptest, backend=mock_backend, optimization_level=3)