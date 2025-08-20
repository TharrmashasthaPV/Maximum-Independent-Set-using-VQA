import numpy as np
from scipy.optimize import curve_fit

def top_k_counts(counts, k):
    """
    A method to return the list of top k keys sorted with 
    respect to counts[key].
    """
    counts_dict = counts.copy()
    top_counts = []
    for i in range(k):
        max_item = max(counts_dict, key=counts_dict.get)
        top_counts.append(max_item)
        del counts_dict[max_item]

    return top_counts

def lin_func(x, a, b):
    """A template for a linear function."""
    return (a * x) + b

def quad_func(x, a, b, c):
    """A template for a quadratic function."""
    return (a * x * x) + b * x + c

def exp_func(x, a, b, c):
    """A template for an exponential function."""
    return (a * np.exp(-b*x)) + c

def exptrapolate_for_zero(x, y, method = 'quadratic'):
    """
    A method to extrapolate the given data to the point 0. This is done by first interpolating the given data
    as a linear, quadratic or exponential function and then using that function to obtain the value at 0.
    Returns:
        A tuple (h, fun) where h is the value of the function at point 0 and fun is the interpolated function.
    """
    if len(x) != len(y):
        raise Exception("The lengths of the input do not match.")

    if method == 'linear':
        fun = lin_func
    elif method == 'quadratic':
        fun = quad_func
    elif method == 'exp':
        fun = exp_func
    else:
        raise Exception(f"The methods should be one of 'linear', 'quadratic', 'exp'. The given method {method} not available.")
    
    opt, cov = curve_fit(fun, x, y)
    def function(x):
        return (opt[0] * np.exp(-opt[1]*x)) + opt[2]

    return fun(0, *opt), lambda x: fun(x, *opt)