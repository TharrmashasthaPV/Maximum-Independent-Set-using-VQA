from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator, BackendEstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from scipy.optimize import minimize, basinhopping, differential_evolution
import numpy as np
import itertools
import warnings
from src.helper_gates import *


class DCQAOA():
    """The DC-QAOA Class."""
    def __init__(
        self,
        problem_hamiltonian = None,
        mixer_hamiltonian = None,
        cd_hamiltonian = None,
        cd_hamiltonian_type = None,
        reference_circuit = None,
        ansatz = None,
        num_layers = 1,
        optimizer = 'COBYLA',
        optimizer_options = {'maxiter' : 1000},
        backend = None,
        transpiler_optimization_level = 3,
        verbose = False
    ):
        """
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
            mixer_hamiltonian : The mixer Hamiltonian as a SparsePauliOp instance.
            cd_hamiltonian : The Counterdiabatic Hamiltonian as a SparsePauliOp Instance.
            cd_hamiltonian_type : If a CD Hamiltonian is not provided, then cd_hamiltonian_type is used to construct a CD Hamiltonian.
                The value can be either the string 'commutator' or a sublist of ['Y', 'XY', 'YX', 'YZ', 'ZY']. If the value is
                'commutator' the CD Hamiltonian is the commutator of the problem Hamiltonian and the mixer Hamiltonian.
                If the value is a sublist of ['Y', 'XY', 'YX', 'YZ', 'ZY'], then the CD Hamiltonian will contain the terms in
                the sublist.
            reference_circuit : A QuantumCircuit as the reference circuit to create the reference state.
            ansatz : A (possibly parameterized) QuantumCircuit as the ansatz.
            num_layers :  Maximum number of repetitions of H_m·H_p layer
            backend : A backend that will be used to estimate the expectation values.
            optimizer : The name of the optimizer to be used to minimize the cost function. Default: 'COBYLA'
            optimizer_options : A dictionary of optimizer options.
        """
        self.problem_hamiltonian = problem_hamiltonian
        if problem_hamiltonian is not None:
            self.num_qubits = problem_hamiltonian.num_qubits
        else:
            self.num_qubits = None
        self.mixer_hamiltonian = mixer_hamiltonian
        if reference_circuit is None:
            reference_circuit = QuantumCircuit(self.num_qubits)
            reference_circuit.x(range(self.num_qubits))
            reference_circuit.h(range(self.num_qubits))

        if self.problem_hamiltonian is not None and self.mixer_hamiltonian is not None:
            self.cd_hamiltonian = self._operator_commutator(
                self.mixer_hamiltonian, self.problem_hamiltonian
            )
        else:
            self.cd_hamiltonian = None
        self.cd_hamiltonian_type = cd_hamiltonian_type
        
        self.reference_circuit = reference_circuit
        self.ansatz = ansatz
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.backend = backend
        if backend is not None:
            self.estimator = Estimator(backend = self.backend)
        else:
            self.estimator = None
        self.transpiler_optimization_level = transpiler_optimization_level

        self.optimal_value = np.inf
        self.optimal_parameters = []
        self.intermediate_parameters_list = []
        self.intermediate_costs_list = None
        
        def tqdm(iterable, **kwargs):
            return iterable
        self._TQDM = False
        self.tqdm = tqdm
        if verbose == True:
            try:
                from tqdm.notebook import tqdm
                self.tqdm = tqdm
                self._TQDM = True
            except:
                pass

    def set_problem_hamiltonian(self, problem_hamiltonian):
        """
        A method to set the problem Hamiltonian of the instance.
        
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
        """
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = problem_hamiltonian.num_qubits

    def set_mixer_hamiltonian(self, mixer_hamiltonian):
        """
        A method to set the mixer Hamiltonian of the instance.

        Args:
            mixer_hamiltonian :  The mixer Hamiltonian as an instance of SparsePauliOp.
        """
        self.mixer_hamiltonian = mixer_hamiltonian

    def set_num_layers(self, num_layers):
        """
        A method to set the number of layers of the instance.

        Args:
            num_layers :  The number of layers for this instance.
        """
        if not isinstance(num_layers, int):
            raise TypeError(f"Argument provided for num_layers (={num_layers}) is not an integer.")
        self.num_layers = num_layers
    
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
            if op == 'XZ':
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
    
    
    def prepare_ansatz(self):
        ''' A custom function to generate the QAOA ansatz for a given problem hamiltonian
        and mixer hamiltonian. reps indicate the number of iterations the mixing is done.'''
        if self.problem_hamiltonian is None:
            raise Exception(
                "Problem Hamiltonian is not defined for this instance. Please use .set_problem_hamiltonian() method to set problem hamiltonian."
            )
        num_qubits = self.num_qubits
        if self.mixer_hamiltonian is None:
            warnings.warn("Mixer Hamiltonian is not provided. Using sum_i X_i as the mixer Hamiltonian.")
            mixer_terms = []
            for i in range(num_qubits):
                term = ['X' if j==i else 'I' for j in range(num_qubits)]
                mixer_terms.append((''.join(term), '1'))
            mixer_hamiltonian = SparsePauliOp.from_list(mixer_terms)
            self.mixer_hamiltonian = mixer_hamiltonian

        if self.cd_hamiltonian is None:
            if not isinstance(self.cd_hamiltonian_type, list) and self.cd_hamiltonian_type == "commutator":
                self.cd_hamiltonian = self._operator_commutator(
                    self.mixer_hamiltonian, self.problem_hamiltonian
                )
            elif isinstance(self.cd_hamiltonian_type, list):
                cd_terms = []
                for i in range(num_qubits):
                    if 'Y' in self.cd_hamiltonian_type:
                        term = ['Y' if j==i else 'I' for j in range(num_qubits)]
                        cd_terms.append((''.join(term), '1'))
                    for j in range(i):
                        if 'YZ' in self.cd_hamiltonian_type:
                            term = ['I' for j in range(num_qubits)]
                            term[i] = 'Y'
                            term[j] = 'Z'
                            cd_terms.append((''.join(term), '1'))
                        if 'ZY' in self.cd_hamiltonian_type:
                            term = ['I' for j in range(num_qubits)]
                            term[i] = 'Z'
                            term[j] = 'Y'
                            cd_terms.append((''.join(term), '1'))
                        if 'XY' in self.cd_hamiltonian_type:
                            term = ['I' for j in range(num_qubits)]
                            term[i] = 'X'
                            term[j] = 'Y'
                            cd_terms.append((''.join(term), '1'))
                        if 'YX' in self.cd_hamiltonian_type:
                            term = ['I' for j in range(num_qubits)]
                            term[i] = 'Y'
                            term[j] = 'X'
                            cd_terms.append((''.join(term), '1'))
                cd_hamiltonian = SparsePauliOp.from_list(cd_terms)
                self.cd_hamiltonian = cd_hamiltonian
        
        ansatz = self.reference_circuit.copy()
    
        for i in range(self.num_layers):
            gamma_i = Parameter('γ_'+str(i))
            beta_i = Parameter('β_'+str(i))
            alpha_i = Parameter('α_'+str(i))
            ansatz.barrier()
            if self.cd_hamiltonian_type == "commutator":
                ansatz.append(self.hamiltonian_to_circuit(1j*self.cd_hamiltonian, time=alpha_i), range(num_qubits))
            else:
                ansatz.append(self.hamiltonian_to_circuit(self.cd_hamiltonian, time=alpha_i), range(num_qubits))
            ansatz.barrier()
            ansatz.append(self.hamiltonian_to_circuit(self.problem_hamiltonian, time=gamma_i), range(num_qubits))
            ansatz.barrier()
            ansatz.append(self.hamiltonian_to_circuit(self.mixer_hamiltonian, time=beta_i), range(num_qubits))
            ansatz.barrier()
    
        return ansatz

    def tqdm_callback(self, maxiter):
        if self._TQDM:
            progress_bar = self.tqdm(total=maxiter, desc="Progress")
        else:
            progress_bar = None
        def save_intermediate_parameters(intermediate_parameters, *args):
            if self._TQDM:
                progress_bar.update(1)
            self.intermediate_parameters_list.append(intermediate_parameters)

        return save_intermediate_parameters, progress_bar

    def cost_function(self, params, ansatz, hamiltonian, estimator):
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
        cost = estimator.run([(ansatz, hamiltonian, [params])]).result()[0].data.evs
        return cost

    def optimize(self, cost_function, init_params, cost_fn_args, callback):
        """
        The classical optimizer function for the class. This function sets the optimizer to one of the
        COBYLA, Powell, L-BFGS-B, basinhopping and differential evolution.
        
        Args:
            cost_function : The cost function to be optimized.
            init_params : The initial parameters for the optimizer.
            cost_fn_args : A tuple containing the args for the cost function.

        Returns:
            result : The optimzer result.
        """
        if self.optimizer == 'basinhopping':
            result = basinhopping(
                cost_function, 
                init_params, 
                minimizer_kwargs = {'args': cost_fn_args}, 
                callback = callback
            )
        elif self.optimizer == 'differential_evolution':
            bounds = [(0, 2*np.pi) for _ in range(cost_fn_args[0].num_parameters)]
            if 'maxiter' in self.optimizer_options:
                maxiter = self.optimizer_options['maxiter']
            result = differential_evolution(
                func = cost_function, 
                bounds = bounds, 
                args = cost_fn_args,
                maxiter = maxiter,
                callback = callback
            )
        elif self.optimizer in ['COBYLA', 'Nelder-Mead', 'L-BFGS-B']:
            result = minimize(
                cost_function, 
                init_params, 
                args = cost_fn_args, 
                method = self.optimizer, 
                options = self.optimizer_options, 
                callback = callback
            )
        else:
            raise Exception(f"The provided optimizer ({self.optimizer}) is not available currently.")
        return result
    
    def run(self, estimator = None):
        """
        The method that performs the CD-QAOA optimization.
        """
        if self.estimator is None and self.backend is None:
            warnings.warn("Backend is not provided. Setting estimator as StatevectorEstimator.")
            self.backend = AerSimulator()
            self.estimator = Estimator(backend = self.backend)
        if self.ansatz is None:
            self.ansatz = self.prepare_ansatz()
        self.estimator = StatevectorEstimator()
        # Optimizing the circuit
        pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level = self.transpiler_optimization_level)
        transpiled_ansatz = pass_manager.run(self.ansatz)
        problem_hamiltonian = self.problem_hamiltonian.apply_layout(layout = transpiled_ansatz.layout)
        init_params = 2 * np.pi * np.random.rand(self.ansatz.num_parameters)
        maxiter = 1001
        if 'maxiter' in self.optimizer_options:
            maxiter = self.optimizer_options['maxiter']
        save_inter_parameters_callback, progress_bar = self.tqdm_callback(maxiter)
        optimizer_result = self.optimize(
            self.cost_function,
            init_params,
            (transpiled_ansatz, problem_hamiltonian, self.estimator),
            callback = save_inter_parameters_callback
        )
        if self._TQDM:
            progress_bar.update(maxiter - progress_bar.n)
            progress_bar.close()
        self.optimal_value = optimizer_result.fun
        self.optimal_parameters = optimizer_result.x

    def intermediate_costs(self):
        if None in [self.ansatz, self.problem_hamiltonian, self.estimator] or len(self.intermediate_parameters_list) == 0:
            raise Exception("Please use .run() method to run the optimization before calling intermediate_costs() method.")
        if self.intermediate_costs_list is None:
            self.intermediate_costs_list = [
                self.cost_function(params, self.ansatz, self.problem_hamiltonian, self.estimator) for params in self.tqdm(self.intermediate_parameters_list, desc="Computing Costs")
            ]
        return self.intermediate_costs_list