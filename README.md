# Maximum-Independent-Set-using-VQA
In this project, we perform a comparison study of various Variational Quantum Algorithms for the Maximum Independent Set (MIS) problem. More specifically, we aim to compare the use of Variational Quantum Eigensolve (VQE), Quantum Approximate Optimization Algorithm (QAOA), Adaptive Derivative Assembled Problem Tailored-QAOA (ADAPT-QAOA), and Digitized Counterdiabatic-QAOA (DC-QAOA) in solving the MIS problem. 

We aim to perform the comparative analysis of the performance of the variational methods with respected to the following three settings:
##### 1. Ansatz Depth
An important reason to use quantum variational algorithms, In this setting, we investigate how the number of layers and the depth of the ansatz affect the performance of the variational algorithms.

##### 2. Classical Optimizer
A classical optimizer is one of the powerhouse of the variational algorithm. So, we next try to address the effect of the classical optimizers on the variational algorithms. Do gradient-free optimizer have any advantages over the gradient-based optimizers? How good are local optimizers perform in comparision with the global optimizers?

##### 3. Graph Classes
In a lot of the computational problems of interest, it is known that the average-case complexity is far lesser than the worst-case complexity. So, given that the MIS problem is NP-Hard, we investigate if there are any special graph classes for which the variational algorithms offer an advantage over the expenential search.

For these comparision, all the experiments will be conducted on noiseless simulators.

### Introducing Noise and the Use of Error Mitigation

Although a comparison in a noiseless setting is a good starting point to understand the variational algorithms, their performance in a real-life setting can be understood only in the presense of noise. For this, we also perform the comparisons in the presense of noise by using noisy simulators. Apart from investigating how noise affects the algorithms, we also try and use error mitigation techniques on the noisy outcomes to understand the extent of usefulness of the error mitigation techniques in the presense of noise.

### Current Stage of the project:

In the current state, this repository contains the source code for functions that generate random graph instances and the class definitions of classes corresponding to the VQE, QAOA, ADAPT-QAOA, and DC-QAOA algorithms, in addition to some helper functoins. 

In the notebook, we show how we use the custom `VQE`, `QAOA`, `AdaptQAOA`, and `DCQAOA` classes to solve the MIS problem for some arbitrarily generated graph instance.