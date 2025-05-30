import random
import math

def pairs_hitting(x):
    n = len(x)
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                cnt += 1
    return cnt

def A(x, _):
    x_new = x[:]
    i, j = random.sample(range(len(x)), 2)
    x_new[i], x_new[j] = x_new[j], x_new[i]
    return x_new

def temp(n, T0=1.0, alpha=0.99):
    return T0 * (alpha ** n)

def annealing(f, x0, A, temp, max_iters=10000):
    x = x0
    fx = f(x)
    best_x = x
    best_fx = fx

    for n in range(1, max_iters + 1):
        T = temp(n)
        if T <= 0:
            break
        x_new = A(x, n)
        fx_new = f(x_new)
        delta = fx_new - fx

        if delta < 0 or random.random() < math.exp(-delta / T):
            x = x_new
            fx = fx_new
            if fx < best_fx:
                best_x = x
                best_fx = fx
            if best_fx == 0:
                break
    return best_x, best_fx


if __name__ == "__main__":
    n = 8
    x0 = list(range(n))
    random.shuffle(x0)

    best_x, best_fx = annealing(
        f=pairs_hitting,
        x0=x0,
        A=A,
        temp=temp,
        max_iters=10000
    )

    print("Best solution:", best_x)
    print("Conflicts:", best_fx)
