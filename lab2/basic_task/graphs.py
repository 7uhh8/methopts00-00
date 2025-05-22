import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def conj_grad_with_history(A, b, x0, eps=1e-6, max_iter=1000):
    x = x0.copy()
    g = A.dot(x) - b
    d = -g
    history = [x.copy()]
    for _ in range(max_iter):
        Ad = A.dot(d)
        alpha = g.dot(g) / d.dot(Ad)
        x = x + alpha * d
        history.append(x.copy())
        g_new = g + alpha * Ad
        if np.linalg.norm(g_new) < eps:
            break
        beta = g_new.dot(g_new) / g.dot(g)
        d = -g_new + beta * d
        g = g_new
    return x, np.array(history)

def precond_cg_with_history(A, b, x0, eps=1e-6, max_iter=1000):
    M = np.diag(np.diag(A))
    M_inv = np.linalg.inv(M)
    x = x0.copy()
    g = A.dot(x) - b
    z = M_inv.dot(g)
    d = -z
    history = [x.copy()]
    for _ in range(max_iter):
        Ad = A.dot(d)
        alpha = g.dot(z) / d.dot(Ad)
        x = x + alpha * d
        history.append(x.copy())
        g_new = g + alpha * Ad
        if np.linalg.norm(g_new) < eps:
            break
        z_new = M_inv.dot(g_new)
        beta = g_new.dot(z_new) / g.dot(z)
        d = -z_new + beta * d
        g, z = g_new, z_new
    return x, np.array(history)

def plot_trajectories(A, b, hist1, hist2):
    xs = np.linspace(-1,  3, 300)
    ys = np.linspace(-4,  4, 300)
    X, Y = np.meshgrid(xs, ys)

    Z = 3*X**2 + 2*X*Y + 2*Y**2 - 4*X + Y

    plt.figure(figsize=(6,5))
    plt.contour(X, Y, Z, levels=30)
    plt.plot(hist1[:,0], hist1[:,1], 'o-', label='CG')
    plt.plot(hist2[:,0], hist2[:,1], 'x-', label='Precond CG')
    plt.scatter(hist1[0,0], hist1[0,1], marker='s', color='k', label='Start')
    x_star = np.linalg.solve(A, b)
    plt.scatter(x_star[0], x_star[1], marker='*', color='r', s=100, label='Analytic Min')
    plt.legend(); plt.xlabel('x'); plt.ylabel('y')
    plt.title('Contour + Trajectories'); plt.grid(True)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=20, cstride=20, alpha=0.6)
    f_val = lambda v: 3*v[0]**2 + 2*v[0]*v[1] + 2*v[1]**2 - 4*v[0] + v[1]
    zs1 = [f_val(pt) for pt in hist1]
    zs2 = [f_val(pt) for pt in hist2]
    ax.plot(hist1[:,0], hist1[:,1], zs1, 'o-', label='CG')
    ax.plot(hist2[:,0], hist2[:,1], zs2, 'x-', label='Precond CG')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('f(x,y)')
    ax.legend(); ax.set_title('Surface + Trajectories')

    plt.show()

if __name__ == "__main__":
    A  = np.array([[6., 2.],
                   [2., 4.]])
    b  = np.array([4., -1.])
    x0 = np.zeros(2)

    x_cg,   hist_cg = conj_grad_with_history    (A, b, x0)
    x_pcg, hist_pcg = precond_cg_with_history   (A, b, x0)

    plot_trajectories(A, b, hist_cg, hist_pcg)
