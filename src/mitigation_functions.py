from qiskit import QuantumCircuit

def circuit_folding(circuit, scaling_factor):
    if scaling_factor%2 != 0 and scaling_factor != 1:
        raise ValueError(f"Only allowed values for scaling factor is 1 and even. Given scaling factor {scaling_factor} is not even or 1.")
    new_circuit = circuit.copy()
    transposed_new_circuit = new_circuit.inverse()
    for _ in range(scaling_factor//2):
        new_circuit.append(transposed_new_circuit, range(circuit.num_qubits))
        new_circuit.append(circuit.copy(), range(circuit.num_qubits))

    return new_circuit

def gate_folding(circuit, scaling_factor):
    if scaling_factor%2 != 0 and scaling_factor != 1:
        raise ValueError(f"Only allowed values for scaling factor is 1 and even. Given scaling factor {scaling_factor} is not even or 1.")
    new_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for inst in circuit.decompose(reps=2).data:
        new_circuit.append(inst)
        for _ in range(scaling_factor//2):
            new_circuit.append(inst.operation.inverse(), inst.qubits, inst.clbits)
            new_circuit.append(inst)

    return new_circuit