import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Callable

np.random.seed(42)

def noisy_f(x: np.ndarray, scale: float = 1.5) -> float:
    value = 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    noise = np.random.normal(loc=0.0, scale=scale)
    return value + noise

def get_noisy_approx_grad(x: np.ndarray) -> np.ndarray:
    t = 1e-5
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.eye(n)[i]
        hi = x + e_i * t
        lo = x - e_i * t
        grad[i] = (noisy_f(hi, scale=1.0) - noisy_f(lo, scale=1.0)) / (2 * t)
    return grad

def gradient_descent_noisy(start: np.ndarray, get_learning_rate: Callable[[int], float],
                           iterations: int, eps=1e-8) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_noisy_approx_grad(points[i])) ** 2 >= eps:
        h = get_learning_rate(i)
        points[i + 1] = points[i] - h * get_noisy_approx_grad(points[i])
        i += 1
    return points[:i + 1]

def gradient_descent_armijo_noisy(start: np.ndarray, iterations: int, eps=1e-8,
                                  alpha_0=1.0, q=0.5, c=1e-4) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_noisy_approx_grad(points[i])) ** 2 >= eps:
        grad = get_noisy_approx_grad(points[i])
        d = -grad
        alpha = alpha_0
        while noisy_f(points[i] + alpha * d, scale=1.0) > noisy_f(points[i], scale=1.0) + c * alpha * np.dot(grad, d):
            alpha *= q
        points[i + 1] = points[i] + alpha * d
        i += 1
    return points[:i + 1]

def constant_learning_rate(h0: float) -> Callable[[int], float]:
    return lambda k: h0

def expon_decay_learning_rate(h0: float, lamb: float) -> Callable[[int], float]:
    return lambda k: h0 * np.exp(-lamb * k)

def traj_to_coords(traj, func) -> tuple:
    x_vals = traj[:, 0]
    y_vals = traj[:, 1]
    z_vals = np.array([func(np.array([x, y]), scale=1.0) for x, y in zip(x_vals, y_vals)])
    return x_vals, y_vals, z_vals

def noisy_f_graphic(x, y):
    shape = x.shape
    value = 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    noise = np.random.normal(loc=0.0, scale=30.0, size=shape)
    return value + noise


if __name__ == "__main__":
    x0 = np.array([10, 5])
    iterations = 5000

    const = constant_learning_rate(1e-2)
    expon = expon_decay_learning_rate(h0=1e-2, lamb=1e-2)

    algo_path_const = gradient_descent_noisy(x0, const, iterations)
    algo_path_expon = gradient_descent_noisy(x0, expon, iterations)
    algo_path_armijo = gradient_descent_armijo_noisy(x0, iterations)

    x_vals_c, y_vals_c, z_vals_c = traj_to_coords(algo_path_const, noisy_f)
    x_vals_e, y_vals_e, z_vals_e = traj_to_coords(algo_path_expon, noisy_f)
    x_vals_a, y_vals_a, z_vals_a = traj_to_coords(algo_path_armijo, noisy_f)


    all_x = np.concatenate([x_vals_c, x_vals_e, x_vals_a])
    all_y = np.concatenate([y_vals_c, y_vals_e, y_vals_a])
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    X = np.linspace(x_min - 1, x_max + 1, 100)
    Y = np.linspace(y_min - 1, y_max + 1, 100)
    X, Y = np.meshgrid(X, Y)
    Z_noisy = noisy_f_graphic(X, Y)


    def print_final_point(name: str, traj: np.ndarray):
        final = traj[-1]
        print(f"{name}:")
        print(f"Итераций: {len(traj) - 1}")
        print(f"x = {final[0]:.10f}, y = {final[1]:.10f}, noisy_f(x, y) = {noisy_f(final, scale=1.0):.10f}")

    print("==== Финальные точки (зашумленная функция) ====")
    print_final_point("Constant step", algo_path_const)
    print_final_point("Exponential decay", algo_path_expon)
    print_final_point("Armijo backtracking", algo_path_armijo)


    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_noisy, cmap='plasma', alpha=0.6)

    ax.plot(x_vals_c, y_vals_c, z_vals_c, color='red', marker='o', markersize=3, label='Constant step')
    ax.plot(x_vals_e, y_vals_e, z_vals_e, color='blue', marker='x', markersize=3, label='Exponential decay')
    ax.plot(x_vals_a, y_vals_a, z_vals_a, color='magenta', marker='^', markersize=3, label='Armijo backtracking')

    ax.scatter(x0[0], x0[1], noisy_f(x0, scale=1.0), color='green', s=50, label='Начальная точка')

    ax.set_title("3D-поиск минимума (зашумленно)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x, y)")
    ax.legend()

    plt.ion()
    plt.show(block=True)
