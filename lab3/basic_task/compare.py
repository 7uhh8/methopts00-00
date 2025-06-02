import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')
X = data.drop("quality", axis=1).values.astype(np.float32)
y = data["quality"].values.astype(np.float32).reshape(-1,1)

np.random.seed(0)
N = X.shape[0]
idx = np.random.permutation(N)
split = int(0.8 * N)
X_train, y_train = X[idx[:split]], y[idx[:split]]
X_test,  y_test  = X[idx[split:]], y[idx[split:]]

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

d = X.shape[1]
EPOCHS = 50

BATCH_SIZES = [1, 8, 32, X_train.shape[0]]

def train_tf_model(optimizer, batch_size):
    tf.random.set_seed(0)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(d,)),
        tf.keras.layers.Dense(1, activation=None)  
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
          .shuffle(buffer_size=X_train.shape[0], seed=0)
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))
    t0 = time.perf_counter()
    hist = model.fit(ds, epochs=EPOCHS, verbose=0)
    elapsed = time.perf_counter() - t0
    test_mse = model.evaluate(X_test, y_test, verbose=0)[0]  
    return hist.history['loss'], test_mse, elapsed

results = []

for B in BATCH_SIZES:
    sgd_opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
    hist_loss, test_mse, elapsed = train_tf_model(sgd_opt, batch_size=B)
    results.append({
        'optimizer': 'SGD(momentum=0.9, lr=1e-3)',
        'batch_size': B,
        'train_time_s': elapsed,
        'test_mse': test_mse,
        'train_loss': hist_loss
    })

for B in BATCH_SIZES:
    adam_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    hist_loss, test_mse, elapsed = train_tf_model(adam_opt, batch_size=B)
    results.append({
        'optimizer': 'Adam(lr=1e-3)',
        'batch_size': B,
        'train_time_s': elapsed,
        'test_mse': test_mse,
        'train_loss': hist_loss
    })

rows = []
for rec in results:
    row = {
        'optimizer': rec['optimizer'],
        'batch_size': rec['batch_size'],
        'train_time_s': rec['train_time_s'],
        'test_mse': rec['test_mse']
    }
    rows.append(row)
df = pd.DataFrame(rows)
print(df[['optimizer','batch_size','train_time_s','test_mse']])

plt.figure(figsize=(10,6))
colors = {'SGD(momentum=0.9, lr=1e-3)':'tab:blue', 'Adam(lr=1e-3)':'tab:orange'}
for rec in results:
    opt = rec['optimizer']
    B   = rec['batch_size']
    hist = np.array(rec['train_loss']) / 2.0
    plt.plot(range(1, EPOCHS+1), hist,
             color=colors[opt],
             linestyle='-' if opt.startswith('SGD') else '--',
             label=f"{opt}, B={B}")

plt.xlabel("Epoch")
plt.ylabel("MSE/2 на train")
plt.title("Сходимость (нормализованные фичи, снижен lr)")
plt.legend()
plt.grid(True)
plt.show()
