import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')
X = data.drop("quality", axis=1).values
y = data["quality"].values

np.random.seed(0)
N = X.shape[0]
indices = np.random.permutation(N)
split = int(0.8 * N)
train_idx, test_idx = indices[:split], indices[split:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
d = X.shape[1]

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
            g = grad_fn(w, X[idx], y[idx])
            w -= lr * g
        loss = np.mean((X.dot(w) - y)**2) / 2
        history.append(loss)
    return w, history

def compute_flops(N, d, B, epochs):
    flops_epoch = 4*N*d + 2*d*(N/B)
    return flops_epoch * epochs

epochs = 50
batch_sizes = [b for b in range(1, N + 1) if b % 2 == 0]
results = []

for B in batch_sizes:
    w0 = np.zeros(d)
    tracemalloc.start()
    t0 = time.perf_counter()
    w_est, hist = sgd(grad_mse, w0, X_train, y_train, lr=0.01, batch_size=B, epochs=epochs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    train_mse = np.mean((X_train.dot(w_est) - y_train)**2) / 2
    test_mse = np.mean((X_test.dot(w_est) - y_test)**2) / 2
    total_flops = compute_flops(X_train.shape[0], d, B, epochs)

    results.append({
        "batch_size": B,
        "train_time_s": elapsed,
        "peak_mem_mb": peak / (1024**2),
        "total_flops": total_flops,
        "train_mse": train_mse,
        "test_mse": test_mse
    })

df = pd.DataFrame(results)
print(df)

plt.figure(figsize=(10, 6))
for B in batch_sizes:
    _, hist = sgd(grad_mse, np.zeros(d), X_train, y_train, lr=0.01, batch_size=B, epochs=epochs)
    plt.plot(hist, label=f"B={B}")
plt.xlabel("Epoch")
plt.ylabel("MSE/2")
plt.legend()
plt.title("Convergence for Different Batch Sizes")
plt.show()
