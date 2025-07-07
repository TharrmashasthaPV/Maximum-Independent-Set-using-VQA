from qiskit.quantum_info import SparsePauliOp

def get_hamiltonian_terms(n):
    terms_list = []
    for i in range(n):
        term = ["I" for _ in range(n)]
        term[n-i-1] = "Z"
        terms_list.append("".join(term))
    for i in range(n):
        for j in range(n):
            term = ["I" for _ in range(n)]
            term[n-i-1] = "Z"
            term[n-j-1] = "Z"
            terms_list.append("".join(term))
    return terms_list

def get_hamiltonian_from_graph(graph):
    penalty = graph.num_nodes()
    n = graph.num_nodes()
    coeffs_single = []
    coeffs_double = []
    for i in range(n):
        coeffs_single.append(1-((penalty*graph.degree(i)/4)))
        for j in range(n):
            if graph.has_edge(i,j):
                coeffs_double.append(penalty/8)
            else:
                coeffs_double.append(0)
    
    coeffs_list = coeffs_single + coeffs_double
    hamiltonian_terms = [ (term, coeff) for term, coeff in zip(get_hamiltonian_terms(n), coeffs_list) if coeff!=0]
    cost_hamiltonian = SparsePauliOp.from_list(hamiltonian_terms)
    
    return cost_hamiltonian