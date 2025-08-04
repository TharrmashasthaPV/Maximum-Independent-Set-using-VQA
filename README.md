# Maximum-Independent-Set-using-VQA
In this project, we perform a comparison study of various Variational Quantum Algorithms for the Maximum Independent Set (MIS) problem. More specifically, we aim to compare the use of Variational Quantum Eigensolve (VQE), Quantum Approximate Optimization Algorithm (QAOA), Adaptive Derivative Assembled Problem Tailored-QAOA (ADAPT-QAOA), and Digitized Counterdiabatic-QAOA (DC-QAOA) in solving the MIS problem. 

We aim to perform a comparative analysis of the performance of the variational methods with respect to the following three settings:
##### 1. Ansatz Depth
An important reason to use quantum variational algorithms is the advantage offered by quantum circuits for certain estimations. In the variational algorithms, these quantum circuits come in the form of ansatzes. In this setting, we investigate how the number of layers in the ansatz, the depth of the ansatz, and the number of parameters in the ansatz affect the performance of the variational algorithms.

##### 2. Classical Optimizer
A classical optimizer is one of the powerhouses of the variational algorithm. So, we next try to address the effect of the classical optimizers on the variational algorithms. Do gradient-free optimizers have any advantages over the gradient-based optimizers? How well do local optimizers perform in comparison with the global optimizers?

##### 3. Graph Classes
In a lot of the computational problems of interest, it is known that the average-case complexity is far less than the worst-case complexity. So, given that the MIS problem is NP-hard, we investigate if there are any special graph classes for which the variational algorithms offer an advantage over the exponential search.

For these comparisons, all the experiments will be conducted on noiseless simulators.

### Introducing Noise and the Use of Error Mitigation

Although a comparison in a noiseless setting is a good starting point to understand the variational algorithms, their performance in a real-life setting can be understood only in the presence of noise. For this, we also perform the comparisons in the presence of noise by using noisy simulators. Apart from investigating how noise affects the algorithms, we also try to use error mitigation techniques on the noisy outcomes to understand the extent of usefulness of the error mitigation techniques in the presence of noise.

### Current Stage of the project:

In the current state, this repository contains the source code for functions that generate random graph instances and the class definitions of classes corresponding to the VQE, QAOA, ADAPT-QAOA, and DC-QAOA algorithms, in addition to some helper functions. 

#### Notebooks:

1. __1. Solving MIS using VQA.ipynb__: This notebook demonstrates how to use the custom `VQE`, `QAOA`, `AdaptQAOA`, and `DCQAOA` classes to solve the MIS problem for some arbitrarily generated graph instance.
2. __2.a. Comparing across Max-Repetitions.ipynb__: The notebook contains a comparison between the performance of the variational algorithms for fixed max-repetitions.
3. __2.b. Comparing across Fixed Depth.ipynb__: The notebook contains a comparison between the performance of the variational algorithms for fixed depths.
4.  __2.c. Comparing across Number of Parameters.ipynb__: Here, we study how the algorithms perform when all the algorithms have the same number of parameters.
5. __3. Dependence on Classical Optimizers.ipynb__: This notebook compares the classical optimizers and investigates how each variational algorithm performs with different classical optimizers.
6. __4. Comparison between various DCQAOA settings.ipynb__: In here, we study how different CD Hamiltonians affect the performance of DCQAOA when solving the MIS problem.