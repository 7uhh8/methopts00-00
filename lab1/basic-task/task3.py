import numpy as np
import scipy.optimize as opt
from functions import *

np.set_printoptions(suppress=True, precision=40)

def steepest_gradient_descent_scipy(f, grad_f, x0, eps=1e-15, max_iter=15_000):
    x = x0.copy()
    for i in range(max_iter):
        grad = grad_f(x)
        norm = np.linalg.norm(grad)
        if norm < eps:
            break
        d = -grad
        phi = lambda alpha: f(x + alpha * d)
        # method -- golden section, range -- (-1, 1), options -- precision (norm < eps -- break)
        res = opt.minimize_scalar(phi, method='golden', bracket=(-1, 1), options={'xtol': eps})
        alpha = res.x
        x = x + alpha * d
        print(f"Iteration {i}: x = {x}, f(x) = {f(x):.40f}, ||grad|| = {norm:.40f}")
    return x

x_start = np.array([0.5, 0.5])
argmin = steepest_gradient_descent_scipy(f, grad_f, x_start)
print("Minimum at:", argmin)
print("Minimum: ", f(argmin))

print(opt.minimize(f, x0=x_start, method='CG').x)