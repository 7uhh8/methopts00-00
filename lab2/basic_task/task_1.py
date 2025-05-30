import numpy as np
import jax
import jax.numpy as jnp
import scipy.optimize as opt
import warnings
warnings.filterwarnings("ignore")

GOLDEN_RATIO = 1.618033988749895

global iter_count
global func_count
global grad_count
global hess_count


# метод одномерного поиска на основе золотого сечения 
def golden_section_method(f, a, b, eps, mode = "min"):
    global iter_count, func_count, grad_count

    x1 = b - (b - a) / GOLDEN_RATIO
    x2 = a + (b - a) / GOLDEN_RATIO

    y1 = f(x1)
    func_count += 1

    y2 = f(x2)
    func_count += 1

    if mode != "max" and mode != "min": 
        return "Unsupported mode"

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


# функция поиска градиента в точке через центральную разностную аппроксимацию
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


# метод Ньютона на основе одномерного поиска золотым сечением
def newton_method_golden_section(f, x0, eps=1e-10, max_iter=15_000):
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

        alpha = golden_section_method(phi, 0, 1, eps)

        x = x + alpha * direction

        iter_count += 1

    return x


# метод Ньютона на основе выбора шага по правилу Армихо
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


# метод Ньютона на основе сопряженного градиента
def newton_method_CG(quadr_params, x0, eps=1e-6, max_iter = 10_000):
    global iter_count, func_count, grad_count, hess_count

    iter_count = 0
    func_count = 0
    grad_count = 0
    hess_count = 0

    A, b = quadr_params()
    A = np.asarray(A)
    b = np.asarray(b)
    xk = x0.astype(float)

    gk = A.dot(xk) - b
    d = -gk
    g0_norm = np.linalg.norm(gk)

    for k in range(max_iter):
        gk_norm = np.linalg.norm(gk)
        if gk_norm <= eps * g0_norm:
            break

        Ad = A.dot(d)
        alpha = np.transpose(gk).dot(gk) / np.transpose(d).dot(Ad)
        xk = xk + alpha * d
        gk_new = gk + alpha * Ad
        beta = np.transpose(gk_new).dot(gk_new) / (np.transpose(gk).dot(gk))
        d = -gk_new + beta * d
        gk = gk_new

        iter_count += 1

    return xk, k, gk_norm


# метод Ньютона с предобуславливанием на основе сопряженного градиента
def preconditioned_newton_method(quadr_params, x0, eps=1e-6, max_iter = 10_000):
    global iter_count, func_count, grad_count, hess_count

    iter_count = 0
    func_count = 0
    grad_count = 0
    hess_count = 0

    A, b = quadr_params()
    A = np.asarray(A)
    b = np.asarray(b)
    xk = x0.astype(float)
    M = np.diag(A.diagonal())
    gk = A.dot(xk) - b
    M_inv = np.linalg.inv(M)
    d = -M_inv.dot(gk)

    g0_norm = np.linalg.norm(gk)

    for k in range(max_iter):
        gk_norm = np.linalg.norm(gk)
        if gk_norm <= eps * g0_norm:
            break

        Ad = A.dot(d)
        alpha = np.transpose(gk).dot(M_inv).dot(gk) / np.transpose(d).dot(Ad)
        xk = xk + alpha * d
        gk_new = gk + alpha * Ad
        beta = np.transpose(gk_new).dot(M_inv).dot(gk_new) / (np.transpose(gk).dot(M_inv).dot(gk))
        d = -M_inv.dot(gk_new) + beta * d
        gk = gk_new

        iter_count += 1


    return xk, k, gk_norm


def steepest_gradient_descent(f, grad_f, x0, search_method=golden_section_method, eps=1e-10, max_iter=15_000, log=True):
    global iter_count, func_count, grad_count

    iter_count = 0
    func_count = 0
    grad_count = 0

    x = np.array(x0, dtype=float)
    for i in range(min(max_iter, 50_000)):
        grad = grad_f(f, x)
        grad_count += 1
        norm = np.linalg.norm(grad)
        if norm < eps:
            break
        d = -grad
        phi = lambda alpha: f(x + alpha * d)
        alpha = search_method(phi, a=-1, b=1, eps=eps)
        x = x + alpha * d
        # if (log):
        #     print(f"Iteration {i}: x = {x}, f(x) = {f(x):.40f}, ||grad|| = {norm:.40f}")
        iter_count += 1
    return x


def log(func_name, argmin, it_count=-1, fun_count=-1, g_count=-1, newton=False):
    print(f"{func_name}. Minimum at: {argmin}")
    if (it_count != -1): 
        print(f"Iterations count: {it_count}")
    else:
        print(f"Iterations count: {iter_count}")

    if (fun_count != -1): 
        print(f"Function calculations count: {fun_count}")
    else:
        print(f"Function calculations count: {func_count}")

    if (g_count != -1): 
        print(f"Gradient calculations count: {g_count}")
    else:
        print(f"Gradient calculations count: {grad_count}")

    if (newton):
        print(f"Hessian calculations count: {hess_count}")
    print("\n")


def f(x):
    return (x[0] - 2)**2 + (x[1] + 3)**2

def interesting_f(x: np.ndarray) -> float:
    return 20 + x[0]**2 + x[1]**2 - 10 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1]))

def quadratic_params():
    A = 2 * np.eye(2)
    b = np.array([4.0, -6.0])
    # из общего вида f(x) = 1/2 * x^T * A * x - b^T * x
    return A, b

def noisy_f(x: np.ndarray) -> float:
    value = 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]
    noise = np.random.normal(loc=0.0, scale=0.1)
    return value + noise



if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=60)
    # x0 = [0, 0]
    max_iter = 15_000


    def make_matrix_A_and_vector_b(n: int):
   
        A = np.zeros((n, n), dtype=float)
        b = np.ones(n, dtype=float)

        for i in range(n):
            A[i, i] = 1 + (i + 1) ** 1.2

        for i in range(n):
            if i + 1 < n:          
                A[i, i + 1] = 1
                A[i + 1, i] = 1
            if i + 100 < n:        
                A[i, i + 100] = 1
                A[i + 100, i] = 1

        return A, b

    n = 500
    A, b = make_matrix_A_and_vector_b(n)
    x0 = np.zeros(n)


    print("Функция (x - 2)**2 + (y + 3)**2 \n")
    x0 = np.array([0.0, 0.0])

    x1 = newton_method_golden_section(f, x0)
    log("Newton + Golden Section", x1, newton=True)

    x2 = newton_method_armijo(f, x0)
    log("Newton + Armijo", x2, newton=True)

    def quad(): return 2 * np.eye(2), np.array([4.,-6.])
    x3, it3, g3 = newton_method_CG(quad, x0)
    log("Newton-CG", x3)

    x4, it4, g4 = preconditioned_newton_method(quad, x0)
    log("Newton-CG-precond", x4)

    print("========================== Методы из scipy.optimize ==========================\n")

    print("newton_scipy (Newton-CG)\n")
    argmin_scipy_newton_cg = opt.minimize(
        fun=f,
        x0=x0,
        method='Newton-CG',
        jac=lambda x: get_approx_grad(f, x),
        tol=1e-10,
        options={'eps': 1e-6, 'maxiter': max_iter, 'disp': True}
    )
    log("newton_scipy (Newton-CG)\n", argmin_scipy_newton_cg.x, newton=True)

    # Квазиньютоновский метод BFGS из scipy.optimize
    print("scipy BFGS\n")
    argmin_scipy_bfgs = opt.minimize(
        fun=f,
        x0=x0,
        method='BFGS',
        jac=lambda x: get_approx_grad(f, x),
        tol=1e-10,
        options={'maxiter': max_iter, 'disp': True}
    )
    log("scipy BFGS", argmin_scipy_bfgs.x, newton=True)

    # Квазиньютоновский метод L-BFGS-B из scipy.optimize
    print("scipy L-BFGS-B\n")
    argmin_scipy_lbfgsb = opt.minimize(
        fun=f,
        x0=x0,
        method='L-BFGS-B',
        jac=lambda x: get_approx_grad(f, x),
        tol=1e-10,
        options={'maxiter': max_iter, 'disp': True}
    )
    log("scipy L-BFGS-B", argmin_scipy_lbfgsb.x, newton=True)


    print("Интересная функция \n")

    x0 = np.zeros(n)

    x1, it1, g1 = newton_method_CG(lambda: make_matrix_A_and_vector_b(n), x0)
    log("Newton-CG", x1, it1)

    x2, it2, g2 = preconditioned_newton_method(lambda: make_matrix_A_and_vector_b(n), x0)
    log("Newton-CG-precond", x2, it2)

