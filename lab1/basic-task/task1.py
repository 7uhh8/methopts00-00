import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Callable

CONST = 392.0987654356789876536757

def f(x: np.ndarray) -> float:
    return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))
    # return 10 + x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0])
    # return CONST + 7 * x[0] ** 2 + 2 * x[1] ** 2
    # return (x[0] ** 2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2
    # return  0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1] + CONST
    # return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

def noisy_f(x: np.ndarray) -> float:
    value = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    noise = np.random.normal(loc=0.0, scale=0.1)
    return value + noise


def get_approx_grad(x: np.ndarray) -> np.ndarray:
    t = 1e-8 # t -> 0
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.eye(n)[i]
        hi = x + e_i * t
        lo = x - e_i * t
        grad[i] = (f(hi) - f(lo)) / (2 * t)
    return grad


def gradient_descent(start: np.ndarray, get_learning_rate: Callable[[int], float], iterations: int, eps=1e-8) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_approx_grad(points[i])) ** 2 >= eps:
        h = get_learning_rate(i)
        points[i + 1] = points[i] - h * get_approx_grad(points[i])
        i += 1
    return points[:i + 1]


def constant_learning_rate(h0: float) -> Callable[[int], float]:
    return lambda k: h0

def expon_decay_learning_rate(h0: float, lamb: float) -> Callable[[int], float]:
    return lambda k: h0 * np.exp(-lamb * k)


def gradient_descent_armijo(start: np.ndarray, iterations: int, eps=1e-8,
                            alpha_0=1.0, q=0.5, c=1e-4) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_approx_grad(points[i])) ** 2 >= eps:
        grad = get_approx_grad(points[i])
        d = -grad
        alpha = alpha_0
        while f(points[i] + alpha * d) > f(points[i]) + c * alpha * np.dot(grad, d):
            alpha *= q

        points[i + 1] = points[i] + alpha * d
        i += 1
    return points[:i + 1]


if __name__ == "__main__":
    x0 = np.array([10, 5])
    iterations = 5000

    const = constant_learning_rate(1e-2)
    expon = expon_decay_learning_rate(h0=1e-2, lamb=1e-2)

    algo_path_const = gradient_descent(x0, const, iterations)
    algo_path_exp = gradient_descent(x0, expon, iterations)
    algo_path_armijo = gradient_descent_armijo(x0, iterations)

    def traj_to_coords(traj):
        x_vals = traj[:, 0]
        y_vals = traj[:, 1]
        z_vals = np.array([f(np.array([x, y])) for x, y in zip(x_vals, y_vals)])
        return x_vals, y_vals, z_vals

    x_vals_c, y_vals_c, z_vals_c = traj_to_coords(algo_path_const)
    x_vals_e, y_vals_e, z_vals_e = traj_to_coords(algo_path_exp)
    x_vals_a, y_vals_a, z_vals_a = traj_to_coords(algo_path_armijo)

    all_x = np.concatenate([x_vals_c, x_vals_e, x_vals_a])
    all_y = np.concatenate([y_vals_c, y_vals_e, y_vals_a])
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    X = np.linspace(x_min - 1, x_max + 1, 100)
    Y = np.linspace(y_min - 1, y_max + 1, 100)
    X, Y = np.meshgrid(X, Y)
    Z = 20 + X ** 2 + Y ** 2 - 10 * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
    # Z = CONST + 7 * X ** 2 + 2 * Y ** 2

    def print_final_point(name: str, traj: np.ndarray):
        final = traj[-1]
        print(f"{name}:")
        print(f"Итераций: {len(traj) - 1}")
        print(f"x = {final[0]:.10f}, y = {final[1]:.10f}, f(x, y) = {f(final):.10f}")

    print("==== Финальные точки алгоритмов ==== ")
    print_final_point("Constant step", algo_path_const)
    print_final_point("Exponential decay", algo_path_exp)
    print_final_point("Armijo backtracking", algo_path_armijo)


    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

    ax.plot(x_vals_c, y_vals_c, z_vals_c, color='red', marker='o', markersize=3, label='Constant step')
    ax.plot(x_vals_e, y_vals_e, z_vals_e, color='blue', marker='x', markersize=3, label='Exponential decay')
    ax.plot(x_vals_a, y_vals_a, z_vals_a, color='magenta', marker='^', markersize=3, label='Armijo backtracking')

    ax.scatter(x0[0], x0[1], f(x0), color='green', s=50, label='Начальная точка')

    ax.set_title("3D-поиск минимума функции\n(Разные стратегии выбора шага)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z = f(x, y)")
    ax.legend()
    plt.ion()
    plt.show(block=True)
