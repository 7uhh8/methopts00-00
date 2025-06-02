import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

def grad_mse_reg(w, Xb, yb, alpha=0.0, l1_ratio=0.0):
    B = len(yb)
    grad = Xb.T.dot(Xb.dot(w) - yb) / B

    if alpha > 0:
        grad += alpha * (1 - l1_ratio) * w
        grad += alpha * l1_ratio * np.sign(w)
    return grad

def sgd(grad_fn, w0, X, y,
        lr=1e-3,
        batch_size=32,
        epochs=50,
        lr_schedule=None,
        alpha=0.0,
        l1_ratio=0.0):
    
    w = w0.copy()
    N = X.shape[0]
    history = []

    if lr_schedule is not None and not callable(lr_schedule):
        lr_schedule = np.array(lr_schedule)
        assert len(lr_schedule) == epochs

    for epoch in range(epochs):
        if lr_schedule is None:
            eta = lr
        else:
            eta = lr_schedule(epoch) if callable(lr_schedule) else lr_schedule[epoch]

        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            g = grad_fn(w, X[idx], y[idx], alpha=alpha, l1_ratio=l1_ratio)
            w -= eta * g

        mse = np.mean((X.dot(w) - y)**2) / 2
        reg = alpha*( (1-l1_ratio)/2 * np.sum(w**2) + l1_ratio * np.sum(np.abs(w)) )
        history.append(mse + reg)
        print(f"Epoch {epoch+1:2d}/{epochs} — loss={history[-1]:.6f}  lr={eta:.5f}")

    return w, history

np.random.seed(0)
N = 200
x = np.linspace(0, 5, N)
y = 2 + 0.5*x - 0.3*x**2 + 0.1*x**3 + np.random.randn(N)*0.5

poly = PolynomialFeatures(degree=3, include_bias=True)
X = poly.fit_transform(x.reshape(-1,1)) 

batch_sizes = [1, 16, 64, X.shape[0]]  
epochs = 100

def exp_decay(epoch, eta0=0.1, decay=0.97):
    return eta0 * (decay**epoch)

plt.figure(figsize=(12,8))
for B in batch_sizes:
    w0 = np.zeros(X.shape[1])
    w_est, hist = sgd(
        grad_mse_reg, w0, X, y,
        lr=0.1,
        batch_size=B,
        epochs=epochs,
        lr_schedule=lambda e: exp_decay(e, eta0=0.1, decay=0.98),
        alpha=0.01,           
        l1_ratio=0.2    
    )
    plt.plot(hist, label=f"B={B}")

plt.xlabel("Эпоха")
plt.ylabel("Loss = MSE/2 + Reg")
plt.legend()
plt.title("Сходимость при разных batch, с LR-scheduler и Elastic-регуляризацией")
plt.show()
