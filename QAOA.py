from qiskit import QuantumCircuit, ClassicalRegister, Aer, execute

from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeLondon, FakeMelbourne

from qiskit.visualization import plot_histogram
from time import time

from numpy import *

from scipy.optimize import optimize, minimize

import matplotlib.pyplot as plt

from networkx import Graph, draw, draw_shell, draw_networkx

"""
Quantum Approximate Optimization Algorithm for the MAX CUT problem

PROBLEM:
Given a graph G with v vertices i = 0,1,...,v-1 and m edges (i,k).
Find a partition of the vertices into two disjoint subsets which maximizes the edges between the two partitions.



"""

class QAOA:

    def __init__(self, graph: Graph, backend = None, noise_model: NoiseModel = None, shots: int = 8192, vertex_qubit_mapping: list = [], n_qubits: int = None):
        """ CONSTRUCTOR
        QAOA class for MAX CUT problem

        :param graph: Graph of networkx.Graph type, to be examined
        :param backend: Backend for execution of quantum circuits. If none given, use "qasm_simulator"
        :param noise_model: Own defined noise model for noisy simulation
        :param shots: Number of shots of the quantum circuit to be executed
        :param vertex_qubit_mapping: Mapping from vertex indices to qubit indices, to account for
        :param n_qubits: Max number of qubits
        """

        # Graph parameters
        self.v = graph.number_of_nodes()
        self.m = graph.number_of_edges()
        self.graph = graph

        # If no specific backend is given, use the qasm_simulator (possibly with an own, specified noise model)
        if backend == None:
            self.backend = Aer.get_backend("qasm_simulator", noise_model=noise_model)
        else:
            self.backend = backend

        self.shots = shots

        # n_qubits and vertex_qubit_mapping allows for a mapping from graph vertex indices to physical qubit indices on
        # a quantum device which may require a different coupling map
        if n_qubits == None:
            self.n_qubits = self.v
        else:
            self.n_qubits = n_qubits

        # self.q[i] contains the index of the physical qubit assigned to graph vertex i
        if len(vertex_qubit_mapping) == 0:
            self.q = [i for i in range(self.v)]
        else:
            self.q = [None]*self.n_qubits
            for map in vertex_qubit_mapping:
                self.q[map[0]] = map[1]

        # Variables keeping track of the optimization process and found results
        self.exp_vals = []
        self.all_counts = []

        self.optimized_exp_val = None
        self.optimized_gamma = None
        self.optimized_beta = None

    def maxcut_cost(self, bitstr: str):
        """ COST FUNCTION

        Cost function as a function of a given bitstring.

        The cost associated with two vertices i, k connected by an edge is 1 if they are in opposite partitions,
        0 if they are in the same. This is represented by the operator:
        C_{i,k} = 1/2 * (id - Z_i x Z_k)

        The total cost is the sum over all C_{i,k} for all edges (i,k) in the graph.

        :param bitstr: bit string z which represents the partition of the graph vertices into
                       two separate subsets (denoted by 0 and 1)
        :return: The cost of the input bit string.
        """
        n = len(bitstr) - 1
        cost = 0
        for (i, j) in self.graph.edges():
            cost += (1 - (1 - 2*int(bitstr[n-i])) * (1 - 2*int(bitstr[n-j]))) / 2
        return cost

    def build_circuit(self, gamma: list, beta: list, barriers: bool = False) -> QuantumCircuit:
        """
        Build quantum circuit for QAOA, parametrized by 2p phases (gamma_{p-1},...,gamma_0) and (beta_{p-1},...,beta_0)

        The total unitary: U(gamma_{p-1},C)*U(beta_{p-1},B)*...*U(gamma_0,C)*U(beta_0,B)

        Where:
        U(gamma, C) = e^(-i * gamma * C)    ,    C being the cost function
        U(beta, B) = e^(-i * beta * B)      ,    Choice of B: B = sum_{i=0}^v B_i,  B_i = X_i (Farhi et.al.)

        Applied on the state |cat> = sum_{x in {0,1}^v} |x>, equal position of all v-bit bitstrings

        :param gamma: List of p phases gamma
        :param beta: List of p phases beta
        :return: quantum circuit executing the above unitary on the |cat> state
        """

        # Quantum register of size n_qubits, as a physical device might have more qubits than we have vertices
        # Reminder: self.q contains the mapping from vertex i to the index of its assigned physical qubit
        self.qc = QuantumCircuit(self.n_qubits, self.v)

        # Prepate the |cat_v> state
        self.qc.h([self.q[i] for i in range(self.v)])

        p = min(len(gamma), len(beta))

        for k in range(p):

            self.qc.barrier()

            # APPLY U(B, beta_k)
            for i in range(self.v):
                self.qc.rx(beta[k], self.q[i])

            self.qc.barrier()

            # APPLY U(C,gamma_k)
            for (i, j) in self.graph.edges():
                self.qc.cnot(self.q[i], self.q[j])
                self.qc.rz(gamma[k], self.q[j])
                self.qc.cnot(self.q[i],self.q[j])

        self.qc.barrier()

        # Add measurements
        for i in range(self.v):
            self.qc.measure(self.q[i], i)

        return self.qc

    def execute_circuit(self, gamma: list, beta: list) -> dict:
        self.build_circuit(gamma=gamma, beta=beta)

        #t_0 = time()
        counts = execute(self.qc, backend=self.backend, shots=self.shots).result().get_counts()
        #print("Execution time:",time() - t_0)

        return counts

    def compute_exp_val(self, counts: dict, shots: int = 0):

        if shots == 0:
            shots = self.shots

        exp_val = 0

        #t_0 = time()
        for bitstring in counts.keys():
            # NOTE: the bitstrings from counts are right-endian, i.e. c[0] on the right end side
            cost = self.maxcut_cost(bitstring)
            exp_val += cost * (counts[bitstring] / shots)
        #print("Postprocessing time:",time()-t_0)

        return exp_val

    # ALTERNATIVE IMPLEMENTATION:

    """
    def pre_divide_edges(self):
        rest = list(self.graph.edges)
        divs = []

        tmp = None
        while len(rest) != 0:
            new = [rest.pop(0)]
            vertices_used = [new[0][0], tmp[0][1]]
            for i in len(rest):
                if edge[0] not in vertices_used and edge[1] not in vertices_used:

            break
    """

    def build_circuit_alt(self, gamma: list, beta: list):
        self.circuits = []

        qc_temp = QuantumCircuit(self.n_qubits,2)

        p = min(len(gamma), len(beta))

        for i in range(self.v):
            qc_temp.h(self.q[i])

        for k in range(p):
            qc_temp.barrier()

            # APPLY U(C,gamma_k)
            for (i, j) in self.graph.edges():
                qc_temp.cnot(self.q[i], self.q[j])
                qc_temp.rz(gamma[k], self.q[j])
                qc_temp.cnot(self.q[i], self.q[j])

            qc_temp.barrier()

            # APPLY U(B, beta_k)
            for i in range(self.v):
                qc_temp.rx(2 * beta[k], self.q[i])

        qc_temp.barrier()

        for (i,j) in self.graph.edges():
            qc_edge = qc_temp.copy()

            qc_edge.measure((self.q[i],self.q[j]),(0,1))

            self.circuits.append(qc_edge.copy())



    def build_circuit_alt_alt(self, gamma: list, beta: list):
        self.qc = QuantumCircuit(self.n_qubits)

        cregs = [ClassicalRegister(2)]*self.m
        for i,c in enumerate(cregs):
            print(i)
            self.qc.add_register(c)

        # Prepate the |cat_v> state
        self.qc.h([self.q[i] for i in range(self.v)])

        p = min(len(gamma), len(beta))

        for k in range(p):

            self.qc.barrier()

            # APPLY U(C,gamma_k)
            for (i, j) in self.graph.edges():
                self.qc.cnot(self.q[i], self.q[j])
                self.qc.rz(gamma[k], self.q[j])
                self.qc.cnot(self.q[i], self.q[j])

            self.qc.barrier()

            # APPLY U(B, beta_k)
            for i in range(self.v):
                self.qc.rx(2 * beta[k], self.q[i])

        self.qc.barrier()

        # APPLY MEASUREMENTS

        for k, (i,j) in enumerate(self.graph.edges):
            self.qc.measure((i,j),cregs[k])

        return self.qc

    def execute_circuit_alt(self, gamma: list, beta: list):
        self.build_circuit_alt(gamma=gamma, beta=beta)

        counts = execute(self.circuits, backend=self.backend, shots=self.shots).result().get_counts()

        return counts

    def execute_circuit_alt_alt(self, gamma: list, beta: list):
        self.build_circuit_alt_alt(gamma=gamma, beta=beta)

        counts = execute(self.qc, backend=self.backend, shots=self.shots).result().get_counts()

        return counts

    def compute_exp_val_alt(self, counts: dict, shots: int = 0):
        if shots == 0:
            shots = self.shots

        exp_val = 0

        for edge_counts in counts:

            zz_exp_val = 0

            for key in edge_counts.keys():
                if key == "00" or key == "11":
                    zz_exp_val += edge_counts[key]
                else:
                    zz_exp_val -= edge_counts[key]
            edge_cost_exp_val = (1/2) * (1 - (zz_exp_val / shots))

            exp_val += edge_cost_exp_val

        return exp_val

    def objective_func(self,var_params: list, version: int = 1):
        """
        Objective function for use in optimization. Returns the negative of the the expectation value of the cost func,
        to be minimized w.r.t the phases gamma_i and beta_i

        :param var_params: variational parameters, [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        :return: -E where E is the expectation value of the cost function w.r.t the state created with the betas and gammas
        """

        p = len(var_params) // 2

        if version == 1:
            counts = self.execute_circuit(gamma=var_params[0:p], beta=var_params[p::])
            exp_val = self.compute_exp_val(counts)
        else:
            counts = self.execute_circuit_alt(gamma=var_params[0:p], beta=var_params[p::])
            exp_val = self.compute_exp_val_alt(counts)

        # Statistics from the optimization process:
        self.all_counts.append(counts)
        self.exp_vals.append(- exp_val)

        return - exp_val

    def run_optimization(self,gamma_start: list, beta_start: list, method: str = 'Nelder-Mead', version: int = 1, save_statistics: bool = True):
        """


        :param gamma_start: start point for optimzation, list of p initial phases gamma
        :param beta_start: start point for optimization, list of p initial phases gamma
        :param method: specification of optimization algorithm for scipy.optimization.minimize. See:
                       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        :param save_statistics:
        :return: scipy.optimization.OptimizeResult object containing the result
        """

        p = min(len(gamma_start), len(beta_start))


        self.exp_vals = []
        self.all_counts = []

        var_params_start = concatenate((gamma_start, beta_start))

        # scipy.optimize.minimize
        res = minimize(fun=self.objective_func, x0=var_params_start, method=method,
                       args=(version,),
                       options={'xatol': 1e-2, 'fatol': 1e-1, 'disp': True})

        # Save the found minimized (maximized) solution with the associated parameters {gamma_i} and {beta_i}
        self.optimized_exp_val = - res.fun
        self.optimized_gamma = res.x[0:p]
        self.optimized_beta = res.x[p:]

        return res

    def landscape_plot(self, gamma_bounds: tuple, beta_bounds: tuple, n: int, version: int = 1):
        gamma = linspace(gamma_bounds[0], gamma_bounds[1], n)
        beta = linspace(beta_bounds[0], beta_bounds[1], n)

        gamma, beta = meshgrid(gamma, beta)

        cost_exp_vals_1 = zeros(shape=shape(gamma))
        cost_exp_vals_2 = zeros(shape=shape(gamma))

        t_start_1 = time()
        for i in range(n):
            for j in range(n):
                counts = self.execute_circuit([gamma[i][j]], [beta[i][j]])
                cost_exp_vals_1[i][j] = self.compute_exp_val(counts)
                print(i,j)
        t_end_1 = time()

        t_start_2 = time()
        for i in range(n):
            for j in range(n):
                counts_2 = self.execute_circuit_alt([gamma[i][j]], [beta[i][j]])
                cost_exp_vals_2[i][j] = self.compute_exp_val_alt(counts_2)
                print(i,j)
        t_end_2 = time()

        print(t_end_1 - t_start_1)
        print(t_end_2 - t_start_2)

        fig, (ax1, ax2) = plt.subplots(2,1,constrained_layout=True,sharex=True)

        ax1.imshow(cost_exp_vals_1, interpolation='bicubic', origin='lower',
                   extent=[gamma_bounds[0],gamma_bounds[1],beta_bounds[0],beta_bounds[1]])
        ax2.imshow(cost_exp_vals_2, interpolation='bicubic', origin='lower',
                   extent=[gamma_bounds[0],gamma_bounds[1],beta_bounds[0],beta_bounds[1]])

        #ax1.colorbar(orientation="horizontal", pad=0.2)

        plt.xlabel(r'$\gamma$')
        ax1.set_ylabel(r'$\beta$')
        ax2.set_ylabel(r'$\beta$')

        ax1.set_title(r"QAOA - single quantum execution, exponential mmt post processing",size=8)
        ax2.set_title(r"QAOA - abs(E) quantum executions, efficient mmt post processing",size=8)

        plt.show()

    def get_optimized_solution(self) -> (float, list, list):
        return self.optimized_exp_val, self.optimized_gamma, self.optimized_beta

    def get_optimized_solution_counts(self):
        if self.optimized_exp_val == None:
            print("run optimization first")
            return

        return self.execute_circuit(self.optimized_gamma, self.optimized_beta)


def create_good_solution_counts(qaoa: QAOA, threshold: int = 0.8) -> (dict, dict, dict):
    """
    Brute force all v-bit bitstrings to find optimal and "good" solutions.
    A solution with cost C >= threshold * C_max is considered "good"

    :param qaoa: QAOA object from which to get the graph and cost function
    :param threshold: Percentile cut off of cost function which counts as a "good" solution
    :return:
    """
    all_costs = {}

    for i in range(2**qaoa.v):
        bitstr = '{0:0{1}b}'.format(i,qaoa.v)
        cost = int(qaoa.maxcut_cost(bitstr))

        all_costs[bitstr] = cost

    optimal = {}
    good = {}

    max_cost = max(all_costs.values())

    for bitstr in all_costs.keys():
        if all_costs[bitstr] == max_cost:
            optimal[bitstr] = 1000
        if all_costs[bitstr] >= threshold * max_cost:
            good[bitstr] = 1000

    return optimal, good, all_costs

if __name__ == "__main__":
    g = Graph([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(3,4)])

    qaoa = QAOA(g)

    qaoa.landscape_plot((0,2*pi),(0,pi),n=100)

    #qaoa.landscape_plot(gamma_bounds=(0,2*pi), beta_bounds=(0,pi),n=10)

    #res = qaoa.run_optimization([.7],[1.],version=1)
    #print(res)

    #print(qaoa.compute_exp_val(qaoa.execute_circuit([.4375], [1.29375])))

    #print(qaoa.compute_exp_val_alt(qaoa.execute_circuit_alt([.4375],[1.29375])))

    #print(qaoa.execute_circuit_alt_alt([.4375],[1.29375]))

    """
    g_2 = Graph([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(2,3),(3,4),(3,5),(4,5),(2,5),(0,6),(1,6),(2,6),(4,7),(5,7),(6,7),(3,7),(3,8),(5,8),(6,9),(1,9),(2,9),
                 (1,10),(2,10),(4,10),(6,10),(8,10),(9,10),(0,11),(2,11),(3,11),(5,11),(7,11),(8,11),(10,11)])

    qaoa_2 = QAOA(g_2)

    t_start_1 = time()
    print(qaoa.compute_exp_val(qaoa.execute_circuit([.4375], [1.29375])))
    t_end_1 = time()

    t_start_2 = time()
    print(qaoa.compute_exp_val_alt(qaoa.execute_circuit_alt([.4375],[1.29375])))
    t_end_2 = time()

    print(t_end_1 - t_start_1)
    print(t_end_2 - t_start_2)

    print("do landscape")

    qaoa_2.landscape_plot(gamma_bounds=(0, 2 * pi), beta_bounds=(0, pi), n=10)
    """
