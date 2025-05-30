import numpy as np

def f(x):
    return np.sum((2 * x - 8.320249)**2)

def genetic_algorithm(f, bounds, dim=2, pop_size=50, generations=100, mutation_rate=0.1):
    population = np.random.uniform(low=[b[0] for b in bounds],
                                   high=[b[1] for b in bounds],
                                   size=(pop_size, dim))

    for _ in range(generations):
        scores = np.array([f(ind) for ind in population])
        best = population[np.argsort(scores)[:pop_size//2]]

        # скрещивание
        children = []
        while len(children) < pop_size:
            p1, p2 = best[np.random.choice(len(best), 2, replace=False)]
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2
            children.append(child)

        # мутация
        for c in children:
            if np.random.rand() < mutation_rate:
                i = np.random.randint(0, dim)
                c[i] += np.random.normal(0, 0.5)

        population = np.clip(np.array(children), [b[0] for b in bounds], [b[1] for b in bounds])

    best_x = min(population, key=f)
    return best_x, f(best_x)
