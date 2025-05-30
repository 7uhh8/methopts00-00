import math
import random
import numpy as np
import matplotlib.pyplot as plt

def f(x: np.ndarray):
    return np.sum((2 * x - 6.8499) ** 2)

def noisy_f(x: np.ndarray):
    noise = np.random.normal(0, 0.1)
    return f(x) + noise

def A(x: np.ndarray, i: int):
    step_size = 0.5
    return x + np.random.uniform(-step_size, step_size, size=x.shape)


def get_temperature(type="exp", **kwargs):
    if type == "exp":
        T0 = kwargs.get("T0", 10.0)
        alpha = kwargs.get("alpha", 0.99)
        return lambda n: T0 * (alpha ** n)

    elif type == "log":
        T0 = kwargs.get("T0", 10.0)
        c = kwargs.get("c", 2.0)
        return lambda n: T0 / math.log(n + c)

    elif type == "poly":
        T0 = kwargs.get("T0", 10.0)
        p = kwargs.get("p", 1.5)
        return lambda n: T0 / ((1 + n) ** p)


def annealing(f, x0, A, temp, max_iters=2000):
    x = np.copy(x0)
    f_x = f(x)
    best_x = np.copy(x)
    best_fx = f_x

    sequence = [f_x]

    for n in range(1, max_iters + 1):
        T = temp(n)
        if T <= 0:
            break

        x_new = A(x, n)
        fx_new = f(x_new)
        delta = fx_new - f_x

        if delta < 0 or random.random() < math.exp(-delta / T):
            x = x_new
            f_x = fx_new
            if f_x < best_fx:
                best_x = np.copy(x)
                best_fx = f_x

        sequence.append(f_x)

    return best_x, best_fx, sequence


if __name__ == "__main__":
    dim = 2
    x0 = np.zeros(dim)

    temp_fn = get_temperature(type="exp", T0=1000.0, alpha=0.9)
    best_x, best_fx, history = annealing(
        f=f,
        x0=x0,
        A=A,
        temp=temp_fn,
        max_iters=2000
    )

    print(f"Best x: {best_x}, f(x): {best_fx:.4f}")

    plt.plot(history)
    plt.xlabel("iteration")
    plt.ylabel("f(x)")
    plt.title("annealing...")
    plt.grid(True)
    plt.show()
