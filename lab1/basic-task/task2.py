import math
import numpy as np
from functions import *
import timeit

GOLDEN_RATIO = 1.618033988749895

global iter_count
global func_count
global grad_count

def golden_section_method(f, a, b, eps, mode ="min"):
    global iter_count, func_count, grad_count

    x1 = b - (b - a) / GOLDEN_RATIO
    x2 = a + (b - a) / GOLDEN_RATIO

    y1 = f(x1)
    func_count += 1

    y2 = f(x2)
    func_count += 1

    if mode != "max" and mode != "min": return "Unsupported mode"

    while abs(b - a) >= eps:
        iter_count += 1
        if (mode == "max" and y1 <= y2) or (mode == "min" and y1 >= y2):
            a = x1
            x1 = x2
            y1 = y2
            x2 = a + (b - a) / GOLDEN_RATIO
            y2 = f(x2)
        else:
            b = x2
            x2 = x1
            y2 = y1
            x1 = b - (b - a) / GOLDEN_RATIO
            y1 = f(x1)

    return (a + b) / 2

# additional task 2
def dichotomy_method(f, a, b, eps, mode ="min"):
    global iter_count, func_count, grad_count

    while (abs(a - b) >= eps):
        iter_count += 1

        c = (a + b) / 2
        left = (a + c) / 2
        right = (c + b) / 2

        if mode == "min":
            if f(left) < f(c):
                func_count += 2
                b = c
            elif f(right) < f(c):
                func_count += 2
                a = c
            else:
                a, b = left, right
        elif mode == "max":
            if f(left) > f(c):
                func_count += 2
                b = c
            elif f(right) > f(c):
                func_count += 2
                a = c
            else:
                a, b = left, right

    return (a + b) / 2

def steepest_gradient_descent(f, grad_f, x0, search_method=golden_section_method, eps=1e-15, max_iter=15_000, log=True):
    global iter_count, func_count, grad_count

    iter_count = 0
    func_count = 0
    grad_count = 0

    x = x0.copy()
    for i in range(min(max_iter, 50_000)):
        grad = grad_f(x)
        grad_count += 1
        norm = np.linalg.norm(grad)
        if norm < eps:
            break
        d = -grad
        phi = lambda alpha: f(x + alpha * d)
        alpha = search_method(phi, a=-1, b=1, eps=eps)
        x = x + alpha * d
        if (log):
            print(f"Iteration {i}: x = {x}, f(x) = {f(x):.40f}, ||grad|| = {norm:.40f}")
        iter_count += 1
    return x

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=60)

    x_start = np.array([21, 40])

    argmin1 = steepest_gradient_descent(f, get_approx_grad, x_start, dichotomy_method, log=False)
    print(f"Dichotomy. Minimum at: {argmin1}")
    print(f"Iteration count: {iter_count}")
    print(f"Function calculation count: {func_count}")
    print(f"Gradient calculation count: {grad_count}")
    print("Dichotomy. Minimum: ", f(argmin1))

    print("\n")

    argmin2 = steepest_gradient_descent(f, get_approx_grad, x_start, golden_section_method, log=False)
    print(f"Golden Section. Minimum at: {argmin2}")
    print(f"Iteration count: {iter_count}")
    print(f"Function calculation count: {func_count}")
    print(f"Gradient calculation count: {grad_count}")
    print("Golden Section. Minimum: ", f(argmin1))
    # for _ in range(5):
    #     dichotomy_execution_time = timeit.timeit("steepest_gradient_descent(f, get_approx_grad, x_start, dichotomy_method, log=False)",
    #                                              globals=globals(),
    #                                              number=10)
    #
    #     golden_section_execution_time = timeit.timeit("steepest_gradient_descent(f, get_approx_grad, x_start, golden_section_method, log=False)",
    #                                              globals=globals(),
    #                                              number=10)
    #     print("Количество запусков каждой функции: 10")
    #     print(f"Среднее время выполнения градиентного спуска на основе дихотомии: {dichotomy_execution_time/1000:.6f} секунд")
    #     print(f"Среднее время выполнения градиентного спуска на основе золотого сечения: {golden_section_execution_time/1000:.6f} секунд")
    #     print("\n")


