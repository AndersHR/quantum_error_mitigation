from qiskit import QuantumCircuit, Aer, execute

from qiskit.providers.aer.noise import NoiseModel

from numpy import *

from scipy.optimize import minimize

from networkx import Graph

"""
Quantum Approximate Optimization Algorithm for the MAX CUT problem

PROBLEM:
Given a graph G with v vertices i = 0,1,...,v-1 and m edges (i,k).
Find a partition of the vertices into two disjoint subsets which maximizes the edges between the two partitions.



"""

class QAOA:

    def __init__(self, graph: Graph, backend=None, noise_model: NoiseModel = None, shots: int = 8192, xatol: float = 1e-2, fatol: float = 1e-1):
        """ CONSTRUCTOR
        QAOA class for MAX CUT problem

        :param graph: Graph of networkx.Graph type, to be examined
        :param backend: Backend for execution of quantum circuits. If none given, use "qasm_simulator"
        :param noise_model: Own defined noise model for noisy simulation
        :param shots: Number of shots of the quantum circuit to be executed
        :param xatol: Absolute error tolerance used in optimization procedure, in the variational parameters
        :param fatol: Absolute error tolerance used in optimization procedure, in the cost function value f(x)
        """

        # Graph parameters
        self.v = graph.number_of_nodes()    # number of vertices/nodes
        self.m = graph.number_of_edges()    # number of edges
        self.graph = graph

        # If no specific backend is given, use the qasm_simulator (possibly with an own, specified noise model)
        if backend == None:
            self.backend = Aer.get_backend("qasm_simulator", noise_model=noise_model)
        else:
            self.backend = backend

        self.shots = shots

        # Error tolerances for optimization
        self.xatol = xatol
        self.fatol = fatol

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

    def build_circuit(self, gamma: list, beta: list, barriers: bool = True) -> QuantumCircuit:
        """
        Build quantum circuit for QAOA, parametrized by 2p phases (gamma_{p-1},...,gamma_0) and (beta_{p-1},...,beta_0)

        The total unitary: U(gamma_{p-1},C)*U(beta_{p-1},B)*...*U(gamma_0,C)*U(beta_0,B)

        Where:
        U(gamma, C) = e^(-i * gamma * C)    ,    C being the cost function
        U(beta, B) = e^(-i * beta * B)      ,    Choice of B: B = sum_{i=0}^v B_i,  B_i = X_i (Farhi et.al.)

        Applied on the state |cat> = sum_{x in {0,1}^v} |x>, equal position of all v-bit bitstrings

        :param gamma: List of p phases gamma
        :param beta: List of p phases beta
        :param barriers: Include barriers, True / False
        :return: quantum circuit executing the above unitary on the |cat> state
        """

        # For a Graph of v vertices we need v qubits
        self.qc = QuantumCircuit(self.v, self.v)

        # Prepate the |cat_v> state
        self.qc.h([i for i in range(self.v)])

        p = min(len(gamma), len(beta))

        for k in range(p):

            if barriers:
                self.qc.barrier()

            # APPLY U(C,gamma_k)
            for (i, j) in self.graph.edges():
                self.qc.cnot(i, j)
                self.qc.rz(gamma[k], j)
                self.qc.cnot(i, j)

                if barriers:
                    self.qc.barrier()

            # APPLY U(B, beta_k)
            for i in range(self.v):
                self.qc.rx(beta[k], i)

        if barriers:
            self.qc.barrier()

        # Add measurements
        self.qc.measure([i for i in range(self.v)], [i for i in range(self.v)])

        return self.qc

    def execute_circuit(self, gamma: list, beta: list):
        self.build_circuit(gamma=gamma, beta=beta)

        result = execute(self.qc, backend=self.backend, shots=self.shots).result()

        return result

    def compute_exp_val(self, result):

        shots = result.results[0].shots

        exp_val = 0

        counts = result.get_counts()

        for bitstring in counts.keys():
            # NOTE: the bitstrings from counts are right-endian, i.e. c[0] on the right end side
            cost = self.maxcut_cost(bitstring)
            exp_val += cost * (counts[bitstring] / shots)

        return exp_val

    def objective_func(self, var_params: list):
        """
        Objective function for use in optimization. Returns the negative of the the expectation value of the cost func,
        to be minimized w.r.t the phases gamma_i and beta_i

        :param var_params: variational parameters, [gamma_0, ..., gamma_{p-1}, beta_0, ..., beta_{p-1}]
        :return: -E where E is the expectation value of the cost function w.r.t the state created with the betas and gammas
        """

        p = len(var_params) // 2

        result = self.execute_circuit(gamma=var_params[0:p], beta=var_params[p::])
        exp_val = self.compute_exp_val(result)

        print(var_params[0:p], var_params[p::])
        print(exp_val)

        # Statistics from the optimization process:
        self.all_counts.append(result.get_counts())
        self.exp_vals.append(- exp_val)

        return - exp_val

    def run_optimization(self,gamma_start: list, beta_start: list, method: str = 'Nelder-Mead'):
        """
        Run optimization for Max cut

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
                       options={'xatol': self.xatol, 'fatol': self.fatol, 'disp': True})

        # Save the found minimized (maximized) solution with the associated parameters {gamma_i} and {beta_i}
        self.optimized_exp_val = - res.fun
        self.optimized_gamma = res.x[0:p]
        self.optimized_beta = res.x[p:]

        return res

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