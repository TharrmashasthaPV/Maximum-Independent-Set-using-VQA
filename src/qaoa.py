from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import *
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimatorV2 as Estimator, StatevectorEstimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from scipy.optimize import minimize
import numpy as np
import itertools
from .helper_gates import *
try:
    from tqdm.notebook import tqdm
    _TQDM = True
except:
    _TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

class QAOA():
    """The QAOA class."""
    def __init__(
        self,
        problem_hamiltonian = None,
        mixer_hamiltonian = None,
        reference_circuit = None,
        ansatz = None,
        num_layers = 1,
        backend = None,
        optimizer = 'COBYLA',
        optimizer_options = {},
    ):
        """
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
            mixer_hamiltonian : The mixer Hamiltonian as a SparsePauliOp instance.
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
        self.reference_circuit = reference_circuit
        self.ansatz = ansatz
        self.num_layers = num_layers
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.backend = backend
        if self.backend is not None:
            self.estimator = Estimator(backend = self.backend)
        else:
            self.backend = None

        self.optimal_value = np.inf
        self.optimal_parameters = []
        self.intermediate_parameters_list = []
        self.intermediate_costs_list = None

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
            mixer_pool :  The mixer Hamiltonians as an instance of SparsePauliOp.
        """
        self.mixer_hamiltonian = mixer_hamiltonian

    def set_num_layers(self, num_layers):
        """
        A method to set the number of layers for the instance.

        Args:
            num_layers : The number of layers of (H_m·H_p) repetition.
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

    def set_reference_circuit(self, reference_circuit):
        """
        A method to set the reference circuit for the instance.

        Args:
            circuit : A QuantumCircuit instance that will construct the reference state.
        """
        if reference_circuit is not None and not isinstance(reference_circuit, QuantumCircuit):
            raise TypeError("The provided reference circuit is not of QuantumCircuit type.")
        self.reference_circuit = reference_circuit
    
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
            print("Mixer Hamiltonian is not provided. Using sum_i X_i as the mixer Hamiltonian.")
            mixer_terms = []
            for i in range(num_qubits):
                term = ['X' if j==i else 'I' for j in range(num_qubits)]
                mixer_terms.append((''.join(term), '1'))
            mixer_hamiltonian = SparsePauliOp.from_list(mixer_terms)
            self.mixer_hamiltonian = mixer_hamiltonian
        
        ansatz = self.reference_circuit.copy()
    
        for i in range(self.num_layers):
            gamma_i = Parameter('γ_'+str(i))
            beta_i = Parameter('β_'+str(i))
            ansatz.barrier()
            ansatz.append(self.hamiltonian_to_circuit(self.problem_hamiltonian, time=gamma_i), range(num_qubits))
            ansatz.barrier()
            ansatz.append(self.hamiltonian_to_circuit(self.mixer_hamiltonian, time=beta_i), range(num_qubits))
            ansatz.barrier()
    
        return ansatz

    def tqdm_callback(self, maxiter):
        if _TQDM:
            progress_bar = tqdm(total=maxiter, desc="Progress")
        else:
            progress_bar = None
        def save_intermediate_parameters(intermediate_parameters):
            if _TQDM:
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

    def run(self, backend = None):
        """
        The method that performs the QAOA optimization.
        """
        if self.backend is None and backend is None:
            print("Backend is not provided. Setting estimator as StatevectorEstimator.")
            self.estimator = StatevectorEstimator()
        if self.ansatz is None:
            self.ansatz = self.prepare_ansatz()
        init_params = 2 * np.pi * np.random.rand(self.ansatz.num_parameters)
        maxiter = 1001
        if 'maxiter' in self.optimizer_options:
            maxiter = self.optimizer_options['maxiter']
        save_inter_parameters_callback, progress_bar = self.tqdm_callback(maxiter)
        optimizer_result = minimize(
            self.cost_function,
            init_params,
            args = (self.ansatz, self.problem_hamiltonian, self.estimator),
            method = self.optimizer,
            options = self.optimizer_options,
            callback = save_inter_parameters_callback
        )
        if _TQDM:
            progress_bar.update(maxiter - progress_bar.n)
            progress_bar.close()
        self.optimal_value = optimizer_result.fun
        self.optimal_parameters = optimizer_result.x

    def intermediate_costs(self):
        """
        A method that returns the list of intermediate costs of the minimize function.
        """
        if None in [self.ansatz, self.problem_hamiltonian, self.estimator] or len(self.intermediate_parameters_list) == 0:
            raise Exception("Please use .run() method to run the optimization before calling intermediate_costs() method.")
        if self.intermediate_costs_list is None:
            self.intermediate_costs_list = [
                self.cost_function(params, self.ansatz, self.problem_hamiltonian, self.estimator) for params in tqdm(self.intermediate_parameters_list, desc="Computing Costs")
            ]
        return self.intermediate_costs_list