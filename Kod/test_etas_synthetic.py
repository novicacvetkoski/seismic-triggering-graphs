"""
test_etas_synthetic.py - validates seismic_kg_etas.ETAS:
  (1) the analytic M-step gradient against finite differences,
  (2) parameter recovery on a simulated ETAS catalogue (true parents known) with the true background imposed.
Usage: python test_etas_synthetic.py [--seed 1]
"""
import argparse, logging, time
import numpy as np
import seismic_kg_etas as E
from test_misd_synthetic import simulate
logging.disable(logging.CRITICAL)

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--bg-scale", type=float, default=1.0)
a = ap.parse_args()
t, x, y, m, par, truth = simulate(seed=a.seed)
N = len(t); L = 100.0
area = (x.max() - x.min()) * (y.max() - y.min()); bg_true = truth["bg_rate"] / L ** 2
K_sim, c_true, p_true = 0.3, 0.01, 1.2
K_etas = K_sim * (p_true - 1) / c_true ** (1 - p_true)       # simulation normalises Omori; ETAS amplitude does not
TRUE = dict(K=K_etas, c=c_true, p=p_true, alpha=0.5, d=0.1, gamma=0.4, q=1.7)

# (1) gradient check on a subset
sub = slice(0, 400)
et = E.ETAS(t[sub] - t[0], x[sub], y[sub], m[sub], mc=2.0, background="fixed", bg_rate=bg_true, area=area)
print("(1) gradient check (analytic vs central finite differences), incl. p < 1, p = 1 and p ~ 1")
for p_test in [0.8, 0.95, 0.99995, 1.0, 1.00003, 1.15, 1.6]:
    et.params = dict(K=0.03, c=0.02, p=p_test, alpha=0.45, d=0.2, gamma=0.3, q=1.6)
    et._estep(np.full(et.N, bg_true))
    ii, jj = np.where(et.W > 1e-8)
    w = et.W[ii, jj]
    data = (w, et._dt[ii, jj], et._dr2[ii, jj], et.xm[ii], float(w.sum()), float(w @ et.xm[ii]))
    z = E.z_from_params(et.params)
    f0, g = et._fg(z, data)
    num = np.zeros(7)
    for k in range(7):
        h = 1e-6 * max(1, abs(z[k])); zp, zm = z.copy(), z.copy(); zp[k] += h; zm[k] -= h
        num[k] = (et._fg(zp, data)[0] - et._fg(zm, data)[0]) / (2 * h)
    print(f"    p = {p_test:<8}  max relative error over the 7 parameters: {np.max(np.abs(g-num)/(np.abs(num)+1e-9)):.1e}   finite: {np.isfinite(f0)}")

# (2) full fit
t0 = time.time()
etas = E.ETAS(t - t[0], x, y, m, mc=2.0, background="fixed", bg_rate=bg_true * a.bg_scale, area=area, max_iter=40).fit()
print(f"\n(2) full EM fit: N={N}, converged={etas.converged} in {len(etas.trace)} iterations, {time.time()-t0:.0f} s  (background imposed at true rate x{a.bg_scale})")
print(f"    {'param':<7}{'true':>10}{'estimated':>12}")
for k in E.PARAM_NAMES:
    print(f"    {k:<7}{TRUE[k]:>10.4g}{etas.params[k]:>12.4g}")
true_bg = par < 0
trig = np.where(~true_bg)[0]
best = etas.W.argmax(axis=0)
print(f"    background: true fraction {true_bg.mean():.3f}   estimated mean P(bg) {etas.w_bg.mean():.3f}   "
      f"mean P(bg) on true-bg {etas.w_bg[true_bg].mean():.3f} / true-triggered {etas.w_bg[~true_bg].mean():.3f}")
print(f"    strongest parent == true parent: {np.mean(best[trig] == par[trig]):.3f}")
v = etas.verify(); print("    probability checks:", "ALL PASSED" if v["all_passed"] else v)
