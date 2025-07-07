"""Defining four gates that are not available in the qiskit circuit library.
These gate are RXY, RYX, RZY, and RYZ gates. """

import numpy
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate


class RXYGate(Gate):
    '''Defining the RXY gate.'''
    def __init__(self, theta, label='RXY'):
        super().__init__("rxy", 2, [theta], label=label)

    def _define(self):
        rxy_circuit = QuantumCircuit(2)
        rxy_circuit.h(0)
        rxy_circuit.sx(1)
        rxy_circuit.cx(1,0)
        rxy_circuit.rz(2.0*self.params[0], 0)
        rxy_circuit.cx(1,0)
        rxy_circuit.h(0)
        rxy_circuit.sxdg(1)
        self.definition = rxy_circuit

    def inverse(self):
        return RXYGate(-self.params[0])

class RYXGate(Gate):
    '''Defining the RYX gate.'''
    def __init__(self, theta, label='RYX'):
        super().__init__("ryx", 2, [theta], label=label)

    def _define(self):
        ryx_circuit = QuantumCircuit(2)
        ryx_circuit.h(1)
        ryx_circuit.sx(0)
        ryx_circuit.cx(1,0)
        ryx_circuit.rz(2.0*self.params[0], 0)
        ryx_circuit.cx(1,0)
        ryx_circuit.h(1)
        ryx_circuit.sxdg(0)
        self.definition = ryx_circuit

    def inverse(self):
        return RYXGate(-self.params[0])

class RZYGate(Gate):
    '''Defining the RZY gate.'''
    def __init__(self, theta, label='RZY'):
        super().__init__("rzy", 2, [theta], label=label)

    def _define(self):
        rzy_circuit = QuantumCircuit(2)
        rzy_circuit.sx(1)
        rzy_circuit.cx(1,0)
        rzy_circuit.rz(2.0*self.params[0], 0)
        rzy_circuit.cx(1,0)
        rzy_circuit.sxdg(1)
        self.definition = rzy_circuit

    def inverse(self):
        return RZYGate(-self.params[0])

class RYZGate(Gate):
    '''Defining the RYZ gate.'''
    def __init__(self, theta, label='RYZ'):
        super().__init__("ryz", 2, [theta], label=label)

    def _define(self):
        ryz_circuit = QuantumCircuit(2)
        ryz_circuit.sx(0)
        ryz_circuit.cx(1,0)
        ryz_circuit.rz(2.0*self.params[0], 0)
        ryz_circuit.cx(1,0)
        ryz_circuit.sxdg(0)
        self.definition = ryz_circuit

    def inverse(self):
        return RYZGate(-self.params[0])