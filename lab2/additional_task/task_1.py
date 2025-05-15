import sys, pathlib

root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from basic_task.task_1 import *
from numpy import asarray
from numpy.linalg import norm as vecnorm
from scipy.optimize._optimize import _line_search_wolfe12
from scipy.optimize._optimize import _epsilon, _check_positive_definite, _check_unknown_options
from scipy.optimize._optimize import OptimizeResult
from scipy.optimize._optimize import _LineSearchError


def minimize_bfgs_custom(fun, x0, jac=None, gtol=1e-5, maxiter=None,
                          c1=1e-4, c2=0.9, disp=False):
    global iter_count, func_count, grad_count
    iter_count = func_count = grad_count = 0

    x = np.asarray(x0, float).flatten()
    n = x.size
    if maxiter is None:
        maxiter = 200 * n

    def f_wrapped(xi):
        global func_count
        val = fun(xi)
        func_count += 1
        return val

    def grad_wrapped(xi):
        global grad_count
        g = jac(xi) if jac else get_approx_grad(fun, xi)
        grad_count += 1
        return g

    fx = f_wrapped(x)
    gx = grad_wrapped(x)
    Hk = np.eye(n)
    k = 0

    old_old_fval = fx + vecnorm(gx) / 2

    while np.linalg.norm(gx, ord=np.inf) > gtol and k < maxiter:
        p = -Hk.dot(gx)
        try:
            alpha, fc, gc, fx_new, old_old_fval, gx_new = _line_search_wolfe12(
                f_wrapped, grad_wrapped, x, p, gx, fx, old_old_fval,
                amin=1e-8, amax=1e8, c1=c1, c2=c2)
        except _LineSearchError:
            alpha = 1e-3
            fx_new = f_wrapped(x + alpha * p)
            gx_new = grad_wrapped(x + alpha * p)

        s = alpha * p
        x_new = x + s
        y = gx_new - gx
        rho_k = 1.0 / np.dot(y, s)
        I = np.eye(n)
        Hk = (I - rho_k * np.outer(s, y)).dot(Hk).dot(I - rho_k * np.outer(y, s)) + rho_k * np.outer(s, s)

        x, fx, gx = x_new, fx_new, gx_new
        k += 1
        iter_count += 1

    res = OptimizeResult(x=x, fun=fx, jac=gx, hess_inv=Hk,
                         nfev=func_count, njev=grad_count,
                         nit=k, success=np.linalg.norm(gx, ord=np.inf) <= gtol)
    if disp:
        print(f"BFGS done: nit={k}, nfev={func_count}, njev={grad_count}")
    return res


if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    print("=== Custom BFGS ===")
    res = minimize_bfgs_custom(fun=f, x0=x0, gtol=1e-10, disp=True)
    print(f"Result: x={res.x}, f={res.fun}")