"""
test_validation.py - validation of the implemented estimators on simulated ETAS catalogues with KNOWN parents
(replaces the missing test_misd_synthetic.py / test_etas_synthetic.py pair).  Truth: K=0.02, c=0.01, p=1.2,
alpha=0.5, d=0.1 km^2, gamma=0.4, q=1.7, b=1, 100x100 km, 300 d, background imposed at the TRUE rate.
Usage:  python test_validation.py [--seeds 1 2 3] [--json validation.json]
Reports (per seed and averaged): ETAS parameters (with and without study-box edge correction), expected background
fraction, share of triggered events whose strongest parent is the true parent, MISD kernel read-outs and parent accuracy.
"""
import argparse, json, logging, time
import numpy as np
import seismic_kg_etas as E
import seismic_kg_misd as M
from sim_etas import simulate
logging.disable(logging.CRITICAL)

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ap.add_argument("--json")
a = ap.parse_args()
TRUE = dict(K=0.02, c=0.01, p=1.2, alpha=0.5, d=0.1, gamma=0.4, q=1.7)
MU = 3.3e-4
KM = 6371.0 * np.pi / 180
rows = []
for seed in a.seeds:
    t, x, y, m, par = simulate(seed=seed, mu=MU)
    N = len(t); area = (x.max() - x.min()) * (y.max() - y.min()); trig = np.where(par >= 0)[0]
    r = dict(seed=seed, N=N, true_bg_frac=float(np.mean(par < 0)))
    for edge in (True, False):
        et = E.ETAS(t, x, y, m, mc=2.0, background="fixed", bg_rate=MU, area=area, max_iter=60, edge_correction=edge).fit()
        best = et.W.argmax(axis=0)
        r["etas_edge" if edge else "etas_noedge"] = dict(params=et.params, bg_frac=float(et.w_bg.mean()), parent_acc=float(np.mean(best[trig] == par[trig])),
                                                      checks=bool(et.verify()["all_passed"]), conv=bool(et.converged), iters=len(et.trace))
    lat0 = 33.0; lats = lat0 + y / KM; lons = 130.8 + x / (KM * np.cos(np.radians(lat0)))
    mi = M.MISD(t, lats, lons, m, mag_min=2.0, kernel="separable", productivity="loglinear", background="fixed", bg_rate=MU).fit()
    best_i = np.full(N, -1); bw = np.zeros(N)
    o = np.lexsort((-mi.omega, mi.child)); ch = mi.child[o]; first = np.r_[True, ch[1:] != ch[:-1]]
    best_i[ch[first]] = mi.parent[o][first]
    r["misd"] = dict(alpha=float(mi.alpha_hat), p=float(mi.p_hat), s=float(mi.s_hat), bg_frac=float(mi.w_bg.mean()),
                     parent_acc=float(np.mean(best_i[trig] == par[trig])), conv=bool(mi.converged))
    rows.append(r)
    print(f"seed {seed}: N={N}  true bg fraction {r['true_bg_frac']:.3f}")

def avg(f): return np.mean([f(r) for r in rows]), np.std([f(r) for r in rows])
print("\nETAS parameter recovery (mean over seeds; true value in brackets)")
for tag in ("etas_edge", "etas_noedge"):
    print(f"  {tag:<12}" + "  ".join(f"{k}={avg(lambda r: r[tag]['params'][k])[0]:.3g} [{TRUE[k]:g}]" for k in E.PARAM_NAMES))
for tag in ("etas_edge", "etas_noedge"):
    print(f"  {tag:<12} bg fraction {avg(lambda r: r[tag]['bg_frac'])[0]:.3f} (true {avg(lambda r: r['true_bg_frac'])[0]:.3f}), strongest parent correct {avg(lambda r: r[tag]['parent_acc'])[0]:.3f} +- {avg(lambda r: r[tag]['parent_acc'])[1]:.3f}, probability checks passed: {all(r[tag]['checks'] for r in rows)}, converged: {all(r[tag]['conv'] for r in rows)}")
print(f"  MISD separable: alpha={avg(lambda r: r['misd']['alpha'])[0]:.3f} [0.5]  p={avg(lambda r: r['misd']['p'])[0]:.3f} [1.2]  s={avg(lambda r: r['misd']['s'])[0]:.2f} [2q=3.4, tail]  bg fraction {avg(lambda r: r['misd']['bg_frac'])[0]:.3f}  strongest parent correct {avg(lambda r: r['misd']['parent_acc'])[0]:.3f} +- {avg(lambda r: r['misd']['parent_acc'])[1]:.3f}")
if a.json:
    json.dump(rows, open(a.json, "w"), indent=1, default=float)
