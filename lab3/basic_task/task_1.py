import numpy as np

def grad_mse(w, Xb, yb):
    return Xb.T.dot(Xb.dot(w) - yb) / len(yb)

def sgd(grad_fn, w0, X, y, lr, batch_size, epochs):
    w = w0.copy()
    N = X.shape[0]
    history = []
    for _ in range(epochs):
        perm = np.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            grad = grad_fn(w, X[idx], y[idx])
            w -= lr * grad
        loss = np.mean((X.dot(w) - y)**2) / 2
        history.append(loss)
    return w, history

import matplotlib.pyplot as plt

batch_sizes = [i for i in range (1, X.shape[0] + 1)]
for B in batch_sizes:
    w0 = np.zeros(X.shape[1])
    _, hist = sgd(grad_mse, w0, X, y,
                  lr=0.01, batch_size=B, epochs=50)
    plt.plot(hist, label=f"B={B}")
plt.xlabel("Эпоха")
plt.ylabel("MSE/2")
plt.legend()
plt.show()
