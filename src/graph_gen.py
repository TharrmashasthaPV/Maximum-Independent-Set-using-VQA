import rustworkx as rx
import numpy as np
from rustworkx.visualization import mpl_draw
import itertools
import networkx as nx

def generate_random_graph(n, p):
    ''' A function that generates a  Erdős-Rényi graph with n nodes
    and any edge between two nodes is present with probability p.'''

    if p > 1 or p < 0:
        raise ValueError(f"The value for p (={p}) is not between 0 and 1.")
    
    return rx.undirected_gnp_random_graph(n, p)

def generate_random_regular_graph(n, d):
    ''' A function to generate a random d-regular graph on n-nodes.
    This uses the networkx random_regular_graph function.'''

    if d > n-1:
        raise ValueError(f"Value of d cannot be larger than n-1.")
    if n*d % 2 != 0:
        raise ValueError(f"A {d}-regular graph on {n} nodes is not possible. The values of d and n should satisfy d*n is even.")

    nx_graph = nx.random_regular_graph(d, n)
    elist = list(nx_graph.edges())
    rx_graph = rx.PyGraph()
    rx_graph.add_nodes_from(list(range(n)))
    rx_graph.add_edges_from([(i,j,1) for (i,j) in elist])
    
    return rx_graph

def generate_complete_graph(n):
    ''' A function to generate an n-node complete graph.'''
    return rx.complete_graph(n)