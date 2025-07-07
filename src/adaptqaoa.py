from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from scipy.optimize import minimize
import numpy as np
import itertools
from .helper_gates import *

class AdaptQAOA():
    """ The Adapt-QAOA class."""
    def __init__(
        self,
        problem_hamiltonian = None,
        mixer_pool = None,
        mixer_pool_type = None,
        reference_circuit = None,
        max_num_layers = 3,
        estimator = None,
        optimizer = None,
        optimizer_options = None,
        threshold = 1e-5,
        error_threshold = 1e-4
    ):
        """
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
            mixer_pool : The list of mixer Hamiltonians. Each Hamiltonian must be a SparsePauliOp instance.
            mixer_pool_type : One of 'qaoa', 'single' and 'double'. If mixer_pool is not provided, a new mixer pool
                will be created depending on the value of mixer_pool_type. Please refer to 'https://arxiv.org/pdf/2005.10258'
                for more information.
            reference_circuit : A QuantumCircuit as the reference circuit to create the reference state.
            max_num_layers :  Maximum number of repetitions of H_m·H_p layer
            estimator : An Estimator primitive that will be used to estimate the expectation values.
            optimizer : The name of the optimizer to be used to minimize the cost function. Default: 'COBYLA'
            optimizer_options : A dictionary of optimizer options.
            threshold : A threshold for the gradient of the mixers. If the gradient of any mixer is below the threshold,
                the algorithm is assumed to have converged and the algorithm stops.
            error_threshold : A threshold for the optimal values. If the difference of two consecutive runs is less than 
                this threshold, the algorithm is considered converged.
        """
        self.problem_hamiltonian = problem_hamiltonian
        if self.problem_hamiltonian is not None:
            self.num_qubits = problem_hamiltonian.num_qubits
        else:
            self.num_qubits = None
        if mixer_pool is not None and not isinstance(mixer_pool, list):
            raise TypeError("The provided mixer pool is not a list. Please provide a list for mixer pool.")
        self.mixer_pool = mixer_pool
        self.mixer_pool_type = mixer_pool_type
        
        if mixer_pool_type not in [None, 'qaoa', 'single', 'double']:
            raise Exception(f"The provided mixer_pool_type ({mixer_pool_type}) is invalid. It should be one of 'qaoa', 'single', or 'double'.")
        if mixer_pool is not None:
            self.mixer_pool_type = None
        elif problem_hamiltonian is not None and mixer_pool_type is None:
            self.mixer_pool_type = 'single'
        
        if self.mixer_pool_type == 'qaoa':
            iden_term = ['I' for j in range(self.num_qubits)]
            mixer_term = 0
            for i in range(self.num_qubits):
                xterm = iden_term.copy()
                xterm[self.num_qubits - i - 1] = 'X'
                mixer_term = mixer_term + SparsePauliOp(''.join(xterm), 1)
            self.mixer_pool = [mixer_term]
        elif self.mixer_pool_type is not None:
            self.mixer_pool = []
            mixer_pool = []
            iden_term = ['I' for j in range(self.num_qubits)]
            xterms = []
            yterms = []
            for i in range(self.num_qubits):
                xterm = iden_term.copy()
                xterm[self.num_qubits - i - 1] = 'X'
                yterm = iden_term.copy()
                yterm[self.num_qubits - i - 1] = 'Y'
                xterms.append(SparsePauliOp(''.join(xterm), 1))
                yterms.append(SparsePauliOp(''.join(yterm),1))
            mixer_pool.append(sum(xterms))
            mixer_pool.append(sum(yterms))
            for i in range(self.num_qubits):
                mixer_pool.append(xterms[i])
                mixer_pool.append(yterms[i])
            self.mixer_pool = mixer_pool
            if self.mixer_pool_type == 'double':
                for (oper1, oper2) in itertools.product(['X','Y','Z'], ['X','Y','Z']):
                    for i in range(self.num_qubits):
                        for j in range(i):
                            term = iden_term.copy()
                            term[i] = oper1
                            term[j] = oper2
                            self.mixer_pool.append(SparsePauliOp(''.join(term), 1))
         
        if reference_circuit is not None and not isinstance(reference_circuit, QuantumCircuit):
            raise TypeError("The provided reference circuit is not of QuantumCircuit type.")
        self.reference_circuit = reference_circuit
        self.max_num_layers = max_num_layers
        self.estimator = estimator
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.threshold = threshold
        self.error_threshold = error_threshold

        self.optimal_value = np.inf
        self.optimal_parameters = []
        self.optimal_ansatz = None
        self.optimal_mixer_list = []
        self.cost_list = []

    def set_problem_hamiltonian(self, problem_hamiltonian):
        """
        A method to set the problem Hamiltonian of the instance.
        
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
        """
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = problem_hamiltonian.num_qubits

    def set_mixer_pool(self, mixer_pool):
        """
        A method to set the mixer pool of the instance.

        Args:
            mixer_pool :  The list of mixer Hamiltonians where each Hamiltonian is an instance
                of SparsePauliOp.
        """
        if mixer_pool is not None and not isinstance(mixer_pool, list):
            raise TypeError("The provided mixer pool is not a list. Please provide a list for mixer pool.")
        self.mixer_pool = mixer_pool

    def set_optimizer(self, optimizer):
        """
        A method to set the optimizer for the instance.

        Args:
            optimizer : The name of the optimizer to be used to minimize the cost function.
        """
        self.optimizer = optimizer

    def set_reference_circuit(self, circuit):
        """
        A method to set the reference circuit for the instance.

        Args:
            circuit : A QuantumCircuit instance that will construct the reference state.
        """
        if reference_circuit is not None and not isinstance(reference_circuit, QuantumCircuit):
            raise TypeError("The provided reference circuit is not of QuantumCircuit type.")
        self.reference_circuit = reference_circuit

    def _operator_commutator(self, operator1, operator2):
        """
        Returns the commutator of operator1 and operator2.
        """
        commutator = self._operator_product(operator1, operator2) - self._operator_product(operator2, operator1)
        return commutator.simplify()
    
    @staticmethod
    def _pauli_product(pauli1, pauli2):
        """
        Returns a pair (pauli, coeff) of operator and complex number such that the product 
        of pauli1 and pauli2 is coeff*pauli.
        """
        pauli1 = pauli1.to_label()
        pauli2 = pauli2.to_label()
        product_dict = {
            'II' : ('I', 1), 'IX' : ('X', 1), 'IY' : ('Y', 1), 'IZ' : ('Z', 1),
            'XI' : ('X', 1), 'XX' : ('I', 1), 'XY' : ('Z', 1j), 'XZ' : ('Y', -1j),
            'YI' : ('Y', 1), 'YX' : ('Z', -1j), 'YY' : ('I', 1), 'YZ' : ('X', 1j),
            'ZI' : ('Z', 1), 'ZX' : ('Y', 1j), 'ZY' : ('X', -1j), 'ZZ' : ('I', 1)
        }
        product_list = [ p+q for p, q in zip(pauli1, pauli2)]
        product_string = ''
        coeff = 1
        for pair in product_list:
            if pair in product_dict.keys():
                prod_pair = product_dict[pair]
                product_string = product_string + prod_pair[0]
                coeff = coeff*prod_pair[1]
            else:
                print(f"Pair ({pair}) is not in keys")

        return product_string, coeff
    
    def _operator_product(self, operator1, operator2):
        """
        Returns the product of operator1 and operator2.
        """
        paulis1 = operator1.paulis
        coeffs1 = operator1.coeffs
        
        paulis2 = operator2.paulis
        coeffs2 = operator2.coeffs
    
        product_list = []
        
        for i, op1 in enumerate(paulis1):
            for j, op2 in enumerate(paulis2):
                product, coeff = self._pauli_product(op1, op2)
                product_list.append((product, coeffs1[i]*coeffs2[j]*coeff))
    
        return SparsePauliOp.from_list(product_list)
    
    @staticmethod
    def _single_hamil_term_to_gate(term, time):
        ''' A function that returns the gate corresponding to the evolution of
        the single term Hamiltonian over time.'''
        operator, coeff = term[0][::-1], term[1]
        circuit = QuantumCircuit(len(operator))
        op_list = [(op, pos) for pos, op in enumerate(operator) if op != 'I']
        gate_param = coeff*time/2
        op_gate_map = {
            'X' : RXGate(gate_param),
            'Y' : RYGate(gate_param),
            'Z' : RZGate(gate_param),
            'XX': RXXGate(gate_param),
            'XY': RXYGate(gate_param),
            'XZ': RZXGate(gate_param),
            'YX': RYXGate(gate_param),
            'YY': RYYGate(gate_param),
            'YZ': RYZGate(gate_param),
            'ZX': RZXGate(gate_param),
            'ZY': RZYGate(gate_param),
            'ZZ': RZZGate(gate_param),
            
        }
        if len(op_list) == 1:
            op = op_list[0][0]
            qubits = [op_list[0][1]]
            gate = op_gate_map[op]
        elif len(op_list) == 2:
            op = op_list[0][0]+op_list[1][0]
            qubits = [op_list[0][1],op_list[1][1]]
            gate = op_gate_map[op]
            if op not in op_gate_map.keys():
                raise Exception(f"Circuit for the operator {operator} cannot be implemented currently since it requires R{op} gate.")
            if op in ['XZ']:
                qubits = list(reversed(qubits))
        else:
            raise Exception(f"Circuit for the operator {operator} cannot be implemented currently since it has more than two Pauli operators.")
    
        return gate, qubits
    
    
    def hamiltonian_to_circuit(self, hamiltonian, trotter_steps=1, time=None):
        '''Obtaining the circuit corresponding to the hamiltonian using
        the trotter formula given by
                e^{-i(H1+H2+...+Hk)t} = (e^{-iH1t/n}e^{-iH2t/n}...e^{-iHkt/n})^n
        where n equals trotter_steps and t equals time.'''
        if time is None:
            time = Parameter('t')
        terms = hamiltonian.to_list()
        h = 1/trotter_steps
        
        circuit = QuantumCircuit(hamiltonian.num_qubits)
        for step in range(trotter_steps):
            for term in terms:
                gate, qubits = self._single_hamil_term_to_gate(term, time*h)
                circuit.append(gate, qubits)
    
        return circuit

    
    def prepare_ansatz(self, step=0):
        """
        Prepares the ansatz corresponding to the step.
        """
        if self.reference_circuit is not None:
            ansatz = reference_circuit.copy()
        else:
            ansatz = QuantumCircuit(self.num_qubits)
            ansatz.h(range(self.num_qubits))

        for i in range(step):
            ansatz.append(
                self.hamiltonian_to_circuit(self.problem_hamiltonian, time = Parameter('gamma_'+str(i))),
                range(self.num_qubits)
            )
            ansatz.append(
                self.hamiltonian_to_circuit(self.optimal_mixer_list[i], time = Parameter('beta_'+str(i))),
                range(self.num_qubits)
            )

        ansatz.append(
            self.hamiltonian_to_circuit(self.problem_hamiltonian, time = Parameter('gamma_'+str(step))),
            range(self.num_qubits)
        )

        return ansatz

    def cost_function(self, params, ansatz, hamiltonian, estimator=None):
        """
        A method to compute the expectation value of a Hamiltonian with respect to
        the state created by the ansatz using estimator.
        
        Args:
            params : The list parameters for the ansatz,
            ansatz : The ansatz that creates the state with respect to which the expectation
                value is obtained.
            hamiltonian : The hamiltonian whose expectation value is calculated.
            estimator : The Estimator primitive which is to be used to obtain the expectation value.

        Returns:
            cost : The expectation value of the Hamiltonian with respect to the state created by the ansatz.
        """
        if estimator is None:
            estimator = self.estimator
        cost = estimator.run([(ansatz, hamiltonian, [params])]).result()[0].data.evs
        return cost
    
    def run(self):
        """
        The method that performs the Adapt-QAOA optimization.
        """
        if self.estimator is None:
            print("The estimator is not provided. Using StatevectorEstimator for estimation.")
            self.estimator = StatevectorEstimator()
        if self.optimizer is None:
            print("The optimizer is set to COBYLA since optimizer is not provided.")
            self.optimizer = 'COBYLA'
        
        prev_parameters = []
        current_ansatz = None
        step = 0
        prev_cost = np.inf
        
        while step < self.max_num_layers:
            step_ansatz = self.prepare_ansatz(step)
            ansatz_parameters = prev_parameters + [0.01]
            mixer_gradient = []
            for mixer in self.mixer_pool:
                observable = -1j*self._operator_commutator(self.problem_hamiltonian, mixer)
                gradient = self.cost_function(ansatz_parameters, step_ansatz, observable, self.estimator)
                mixer_gradient.append(gradient)
            gradient_abs = np.linalg.norm(mixer_gradient)
            if gradient_abs <= self.threshold:
                print("A solution is reached.")
                return
            optimal_mixer = self.mixer_pool[mixer_gradient.index(max(mixer_gradient))]
            self.optimal_mixer_list.append(optimal_mixer)
            step_ansatz.append(
                self.hamiltonian_to_circuit(optimal_mixer, time = Parameter('beta_'+str(step))),
                range(self.num_qubits)
            )
            
            mixer_params_init = prev_parameters[:len(prev_parameters)//2] + ['0.0'] + prev_parameters[len(prev_parameters)//2:] + ['0.01']

            step_result = minimize(
                self.cost_function,
                mixer_params_init,
                args = (step_ansatz, self.problem_hamiltonian, self.estimator),
                method = self.optimizer,
                options = self.optimizer_options
            )

            self.cost_list.append(step_result.fun)
            prev_parameters = list(step_result.x)
            if step_result.fun < self.optimal_value:
                self.optimal_parameters = list(step_result.x)
                self.optimal_value = step_result.fun
                self.optimal_ansatz = step_ansatz.copy()
            step = step + 1
            if step > 1 and np.abs(self.cost_list[-1]-self.cost_list[-2]) < self.error_threshold:
                return