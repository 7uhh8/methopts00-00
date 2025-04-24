import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from typing import Callable

CONST = 392.0987654356789876536757


def f(x: np.ndarray) -> float:
    return 20 + x[0] ** 2 + x[1] ** 2 - 10 * (
        np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])
    )
    # Alternative test cases kept for quick experiments ↓↓↓
    # return 10 + x[0] ** 2 - 10 * np.cos(2 * np.pi * x[0])
    # return CONST + 7 * x[0] ** 2 + 2 * x[1] ** 2
    # return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    # return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1] + CONST
    # return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def noisy_f(x: np.ndarray) -> float:
    value = 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]
    noise = np.random.normal(loc=0.0, scale=0.1)
    return value + noise


def get_approx_grad(x: np.ndarray) -> np.ndarray:
    t = 1e-8  # step → 0
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.eye(n)[i]
        hi = x + e_i * t
        lo = x - e_i * t
        grad[i] = (f(hi) - f(lo)) / (2 * t)
    return grad


def gradient_descent(
    start: np.ndarray,
    get_learning_rate: Callable[[int], float],
    iterations: int,
    eps: float = 1e-8,
) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_approx_grad(points[i])) ** 2 >= eps:
        h = get_learning_rate(i)
        points[i + 1] = points[i] - h * get_approx_grad(points[i])
        i += 1
    return points[: i + 1]


def constant_learning_rate(h0: float) -> Callable[[int], float]:
    return lambda k: h0


def expon_decay_learning_rate(h0: float, lamb: float) -> Callable[[int], float]:
    return lambda k: h0 * np.exp(-lamb * k)


def gradient_descent_armijo(
    start: np.ndarray,
    iterations: int,
    eps: float = 1e-8,
    alpha_0: float = 1.0,
    q: float = 0.5,
    c: float = 1e-4,
) -> np.ndarray:
    n = start.shape[0]
    points = np.zeros((iterations + 1, n))
    points[0] = start
    i = 0
    while i < iterations and np.linalg.norm(get_approx_grad(points[i])) ** 2 >= eps:
        grad = get_approx_grad(points[i])
        d = -grad
        alpha = alpha_0
        # Armijo backtracking line search
        while f(points[i] + alpha * d) > f(points[i]) + c * alpha * np.dot(grad, d):
            alpha *= q
        points[i + 1] = points[i] + alpha * d
        i += 1
    return points[: i + 1]



def traj_to_coords(traj: np.ndarray):
    x_vals = traj[:, 0]
    y_vals = traj[:, 1]
    z_vals = np.array([f(np.array([x, y])) for x, y in zip(x_vals, y_vals)])
    return x_vals, y_vals, z_vals


def print_final_point(name: str, traj: np.ndarray):
    final = traj[-1]
    print(f"{name}:")
    print(f"\tИтераций: {len(traj) - 1}")
    print(
        f"\tx = {final[0]:.10f},\ty = {final[1]:.10f},\tf(x, y) = {f(final):.10f}"
    )


if __name__ == "__main__":
    x0 = np.array([10, 5])
    iterations = 5000

    const_lr = constant_learning_rate(1e-2)
    expon_lr = expon_decay_learning_rate(h0=1e-2, lamb=1e-2)

    path_const = gradient_descent(x0, const_lr, iterations)
    path_exp = gradient_descent(x0, expon_lr, iterations)
    path_armijo = gradient_descent_armijo(x0, iterations)

    x_c, y_c, z_c = traj_to_coords(path_const)
    x_e, y_e, z_e = traj_to_coords(path_exp)
    x_a, y_a, z_a = traj_to_coords(path_armijo)

    all_x = np.concatenate([x_c, x_e, x_a])
    all_y = np.concatenate([y_c, y_e, y_a])
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)

    X = np.linspace(x_min - 1, x_max + 1, 200)
    Y = np.linspace(y_min - 1, y_max + 1, 200)
    X, Y = np.meshgrid(X, Y)
    Z = 20 + X ** 2 + Y ** 2 - 10 * (
        np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y)
    )

    print("==== Финальные точки алгоритмов ==== ")
    print_final_point("Constant step", path_const)
    print_final_point("Exponential decay", path_exp)
    print_final_point("Armijo backtracking", path_armijo)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6, linewidth=0)

    ax.plot(x_c, y_c, z_c, color="red", marker="o", markersize=3, label="Constant step")
    ax.plot(x_e, y_e, z_e, color="blue", marker="x", markersize=3, label="Exponential decay")
    ax.plot(x_a, y_a, z_a, color="magenta", marker="^", markersize=3, label="Armijo backtracking")

    ax.scatter(x0[0], x0[1], f(x0), color="green", s=50, label="Начальная точка")

    ax.set_title("3D‑поиск минимума функции\n(Разные стратегии выбора шага)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z = f(x, y)")
    ax.legend()

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)

    contour_levels = 60
    CS = ax2.contour(X, Y, Z, levels=contour_levels, cmap="viridis")
    ax2.clabel(CS, inline=True, fontsize=7, fmt="%.0f")

    ax2.plot(x_c, y_c, color="red", marker="o", markersize=3, label="Constant step")
    ax2.plot(x_e, y_e, color="blue", marker="x", markersize=3, label="Exponential decay")
    ax2.plot(x_a, y_a, color="magenta", marker="^", markersize=3, label="Armijo backtracking")

    ax2.scatter(x0[0], x0[1], color="green", s=50, label="Начальная точка")

    ax2.set_title("Линии уровня функции\n(Траектории градиентного спуска)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.ion()
    plt.show(block=True)
