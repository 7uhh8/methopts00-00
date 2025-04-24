import numpy as np
import jax
import optuna

# 3x^2 + 2xy + 2y^2 - 4x + y

# global min = -43/20 = -2.15
def f(x: np.ndarray) -> float:
    return 3 * x[0]**2 + 2 * x[0] * x[1] + 2 * x[1]**2 - 4 * x[0] + x[1]

def get_approx_grad(f, x: np.ndarray) -> np.ndarray:
    t = 1e-8 # t -> 0
    n = x.shape[0]
    grad = np.zeros(n)
    for i in range(n):
        e_i = np.eye(n)[i]
        hi = x + e_i * t
        lo = x - e_i * t
        grad[i] = (f(hi) - f(lo)) / (2 * t)
    return grad

def newton_method_armijo(f, x0, eps=1e-10, max_iter=15_000, alpha_init=1.0, rho=0.5, c=1e-4):
    global iter_count, func_count, grad_count, hess_count

    iter_count = 0
    func_count = 0
    grad_count = 0
    hess_count = 0

    hess = jax.hessian(f)

    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = get_approx_grad(f, x)
        grad_count += 1

        if np.linalg.norm(grad) < eps:
            break

        Hx = np.array(hess(x))
        hess_count += 1
        hess_inv = np.linalg.inv(Hx)
        direction = -hess_inv.dot(grad)

        def phi(alpha):
            global func_count
            func_count += 1
            return f(x + alpha * direction)

        alpha = alpha_init
        f_x = f(x)
        func_count += 1

        while True:
            x_new = x + alpha * direction
            f_new = f(x_new)
            func_count += 1
            if f_new <= f_x + c * alpha * grad.dot(direction):
                break
            alpha *= rho

        x = x_new
        iter_count += 1

    return x

# OPTUNA HYPERPARAMS
def objective(trial):
    alpha_init = trial.suggest_float("alpha_init", 0.1, 2.0)
    rho = trial.suggest_float("rho", 0.1, 0.9)
    c = trial.suggest_float("c", 1e-6, 1e-2, log=True)

    x0 = np.array([0.0, 0.0])

    x_min = newton_method_armijo(f, x0, alpha_init=alpha_init, rho=rho, c=c)
    return f(x_min)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3)

print("Best hyperparameters: ", study.best_params)
print("Min value:", study.best_value)
