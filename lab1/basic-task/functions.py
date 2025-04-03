import math
import numpy as np

C = 6.73994940902456754567
# def f(x):
#     return 99.728399393 * x[0]**2 + 2 * x[1]**2 + C

def f(x):
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1] + C

# def f(x):
#     return 7 * x[0] ** 2 + 2 * x[1] ** 2 + C

# def f(x):
#     return 20 + x[0] ** 2 + x[1] ** 2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))

def get_approx_grad(x: np.ndarray) -> np.ndarray:
    h = 1e-8
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.eye(n)[i]
        hi = x + e_i * h
        lo = x - e_i * h
        grad[i] = (f(hi) - f(lo)) / (2 * h)
    return grad

# def grad_f(x):
#     return np.array([2 * 99.728399393 * x[0], 4 * x[1]])