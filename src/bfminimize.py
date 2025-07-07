from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator

class BFMinimizeCombinatorial():
    def __init__(
        self,
        problem_hamiltonian
    ):
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = problem_hamiltonian.num_qubits
        self.optimal_value = None
        self.optimal_assignments = None
        self.estimator = StatevectorEstimator()
        self.cost_dict = {}
        self.sorted_cost_list = []

    def run(self):
        for i in range(2**self.num_qubits):
            assignment = bin(i)[2:].zfill(self.num_qubits)
            assignment_circuit = QuantumCircuit(self.num_qubits)
            assignment_circuit.initialize(assignment)
            cost = self.estimator.run([(assignment_circuit, self.problem_hamiltonian)]).result()[0].data.evs
            self.cost_dict[assignment] = float(cost)
        self.optimal_value = self.cost_dict[min(self.cost_dict, key = self.cost_dict.get)]
        self.optimal_assignments = [key for key, value in self.cost_dict.items() if value == self.optimal_value]
        self.sorted_cost_list = sorted(list(set(self.cost_dict.values())))

    def spectral_gap(self):
        if len(self.sorted_cost_list) == 0:
            raise Exception("Cost list is missing. Please use the .run()  method to generate the cost list.")
        return self.sorted_cost_list[1] - self.sorted_cost_list[0]

    def get_nth_least_cost(self, n):
        if len(self.sorted_cost_list) == 0:
            raise Exception("Cost list is missing. Please use the .run()  method to generate the cost list.")
        if n >= len(self.sorted_cost_list):
            print(f"The provided value for n (={n}) is larger than the number of unique cost values (={len(self.sorted_cost_list)}). "\
            f"The value of n should be between 0 and {len(self.sorted_cost_list)-1}")
            return
        return self.sorted_cost_list[n]