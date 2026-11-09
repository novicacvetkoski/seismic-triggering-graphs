import numpy as np
def simulate(seed=1, T=300.0, L=100.0, mu=3.3e-4, K=0.02, c=0.01, p=1.2, alpha=0.5, d=0.1, gamma=0.4, q=1.7, mc=2.0, b=1.0):
    rng = np.random.default_rng(seed)
    nbg = rng.poisson(mu * T * L * L)
    ev = [(rng.uniform(0, T), rng.uniform(0, L), rng.uniform(0, L), mc + rng.exponential(1 / (b * np.log(10))), -1) for _ in range(nbg)]
    out, queue = [], list(range(len(ev)))
    out = list(ev)
    i = 0
    while i < len(out):
        t, x, y, m, _ = out[i]
        x_ = m - mc
        n = rng.poisson(K * 10 ** (alpha * x_) * c ** (1 - p) / (p - 1))
        for _ in range(n):
            s = c * ((1 - rng.random()) ** (-1 / (p - 1)) - 1)
            if t + s > T: continue
            di = d * 10 ** (gamma * x_)
            r = np.sqrt(di * ((1 - rng.random()) ** (-1 / (q - 1)) - 1)); th = rng.uniform(0, 2 * np.pi)
            xx, yy = x + r * np.cos(th), y + r * np.sin(th)
            if 0 <= xx <= L and 0 <= yy <= L:
                out.append((t + s, xx, yy, mc + rng.exponential(1 / (b * np.log(10))), i))
        i += 1
    out_sorted = sorted(range(len(out)), key=lambda k: out[k][0])
    rank = {k: r for r, k in enumerate(out_sorted)}
    a = np.array([out[k][:4] for k in out_sorted])
    par = np.array([rank[out[k][4]] if out[k][4] >= 0 else -1 for k in out_sorted])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], par
