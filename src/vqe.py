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

class VQE():
    "The VQE class."
    def __init__(
        self,
        problem_hamiltonian = None,
        ansatz = None,
        reference_circuit = None,
        backend = None,
        optimizer = 'COBYLA',
        optimizer_options = {'maxiter' : 1000},
    ):
        """
        Args:
            problem_hamiltonian : The problem Hamiltonian as a SparsePauliOp instance.
            ansatz : A (possibly parameterized) QuantumCircuit as the ansatz.
            reference_circuit : A QuantumCircuit as the reference circuit to create the reference state.
            backend : A backend that will be used to estimate the expectation values.
            optimizer : The name of the optimizer to be used to minimize the cost function. Default: 'COBYLA'
            optimizer_options : A dictionary of optimizer options.
        """
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = problem_hamiltonian.num_qubits
        self.ansatz = ansatz
        self.reference_circuit = reference_circuit
        self.backend = backend
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options

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

    def set_ansatz(self, ansatz):
        """
        A method to set the ansatz for the instance.

        Args:
            circuit : An ansatz as a QuantumCircuit instance.
        """
        if ansatz is not None and not isinstance(ansatz, QuantumCircuit):
            raise TypeError("The provided ansatz is not of QuantumCircuit type.")
        self.ansatz = ansatz

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
        The method that performs the VQE optimization.
        """
        if self.backend is None and backend is None:
            print("Backend is not provided. Setting estimator as StatevectorEstimator.")
            self.estimator = StatevectorEstimator()
        if self.ansatz is None:
            print(
                "Ansatz is not provided. Choosing TwoLocal ansatz with [RX, RY] for rotation and [CZ] for entanglement"\
                " with full entanglement and 3 repetition of layers."
            )
            self.ansatz = TwoLocal(
                self.num_qubits,
                rotation_blocks = ['rx', 'ry'],
                entanglement_blocks = ['cz'],
                entanglement = 'full',
                reps = 3,
            )
        if self.reference_circuit is not None:
            self.ansatz = self.reference_circuit.compose(self.ansatz, inplace = False)
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