import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# —— 1) Функция-градиент с учётом регуляризации — L1, L2 или ElasticNet —— #
def grad_mse_reg(w, Xb, yb, alpha=0.0, l1_ratio=0.0):
    """
    alpha     — общая сила регуляризации (λ)
    l1_ratio  — доля L1 в ElasticNet (0=L2 only, 1=L1 only)
    """
    B = len(yb)
    # градиент MSE
    grad = Xb.T.dot(Xb.dot(w) - yb) / B

    if alpha > 0:
        # L2 часть: ∂(α*(1−l1_ratio)/2 * ||w||²) = α*(1−l1_ratio) * w
        grad += alpha * (1 - l1_ratio) * w
        # L1 часть: ∂(α*l1_ratio * ||w||₁) ≈ α*l1_ratio * sign(w)
        grad += alpha * l1_ratio * np.sign(w)
    return grad

# —— 2) SGD с LR-scheduler и регуляризацией —— #
def sgd(grad_fn, w0, X, y,
        lr=1e-3,
        batch_size=32,
        epochs=50,
        lr_schedule=None,
        alpha=0.0,
        l1_ratio=0.0):
    """
    grad_fn      — должен принимать (w, Xb, yb, alpha, l1_ratio)
    lr_schedule  — либо None (фиксированный lr), либо функция lr(epoch)->float,
                   либо список/массив из length=epochs
    alpha        — сила регуляризации
    l1_ratio     — доля L1 (все остальное — L2)
    """
    w = w0.copy()
    N = X.shape[0]
    history = []

    # если передали список lr, приводим к массиву
    if lr_schedule is not None and not callable(lr_schedule):
        lr_schedule = np.array(lr_schedule)
        assert len(lr_schedule) == epochs

    for epoch in range(epochs):
        # выбираем текущий learning rate
        if lr_schedule is None:
            eta = lr
        else:
            eta = lr_schedule(epoch) if callable(lr_schedule) else lr_schedule[epoch]

        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            # передаём alpha и l1_ratio в grad_fn
            g = grad_fn(w, X[idx], y[idx], alpha=alpha, l1_ratio=l1_ratio)
            w -= eta * g

        # полная MSE + рег. штраф (для логирования)
        mse = np.mean((X.dot(w) - y)**2) / 2
        reg = alpha*( (1-l1_ratio)/2 * np.sum(w**2) + l1_ratio * np.sum(np.abs(w)) )
        history.append(mse + reg)
        print(f"Epoch {epoch+1:2d}/{epochs} — loss={history[-1]:.6f}  lr={eta:.5f}")

    return w, history

# —— 3) Пример: полином 3-й степени —— #
#   (можно заменить на ваши многомерные X, y)
np.random.seed(0)
N = 200
x = np.linspace(0, 5, N)
y = 2 + 0.5*x - 0.3*x**2 + 0.1*x**3 + np.random.randn(N)*0.5

poly = PolynomialFeatures(degree=3, include_bias=True)
X = poly.fit_transform(x.reshape(-1,1))  # shape = (N, 4)

# —— 4) Эксперименты: разные batch, Scheduler и регуляризация —— #
batch_sizes = [1, 16, 64, X.shape[0]]  # от чистого SGD до GD
epochs = 100

# Пример scheduler: экспоненциальный спад η₀ * decay^epoch
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
        alpha=0.01,           # общая сила регуляризации
        l1_ratio=0.2          # 20% L1, 80% L2
    )
    plt.plot(hist, label=f"B={B}")

plt.xlabel("Эпоха")
plt.ylabel("Loss = MSE/2 + Reg")
plt.legend()
plt.title("Сходимость при разных batch, с LR-scheduler и Elastic-регуляризацией")
plt.show()
