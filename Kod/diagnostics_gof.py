"""
diagnostics_gof.py - does the fitted model describe the data?  (mentor's point on 5.4 / 6.1)
Numerical consistency (probabilities sum to 1, EM converged) is not a goodness-of-fit test. Here the temporal conditional
intensity of each fitted model, lambda(t) = integral over the study box of lambda(t, x, y | history), is integrated to
Lambda(t) and compared with the observations:
  * compensator plot: observed cumulative number N(t) against Lambda(t_j);
  * time-rescaling residuals (Ogata 1988): if the model is right, the intervals Lambda(t_j) - Lambda(t_{j-1}) are
    independent unit-rate exponential variables; shown as a QQ plot and summarised by the Kolmogorov-Smirnov distance.
ETAS uses the stored parameters (no refit); MISD is re-fitted with the same settings (fast) because its kernel histograms are
not stored. Run from the folder you ran the models from (the catalogue path is read from the run JSON).

  python diagnostics_gof.py --results-dir results --seqs tohoku,kumamoto,noto
Outputs: tabeli/tab_gof.tex, figures/fig_gof_kompenzator.pdf, figures/fig_gof_qq.pdf, gof_sumarno.json
"""
import argparse, json, logging, os
import numpy as np
import pandas as pd
from scipy.stats import kstest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kg_common as C
import seismic_kg_etas as E
import seismic_kg_misd as M
logging.disable(logging.CRITICAL)
LABEL = {"tohoku": "Tōhoku", "kumamoto": "Kumamoto", "noto": "Noto"}


def load(run):
    r = run["resolved"]
    df = C.load_catalogue(r["catalogue"], r["mag_min"], run["args"].get("max_events", C.MAX_EVENTS), w_mag_jma=r.get("w_mag_jma"))
    t = (df["datetime"] - df["datetime"].min()).dt.total_seconds().values / 86400.0
    return df, t, r


def lambda_etas(run):
    df, t, r = load(run)
    x, y = C.project_km(df["latitude"].values, df["longitude"].values)
    area = C.domain_area_km2(x, y)
    et = E.ETAS(t, x, y, df["mag"].values, mc=r["mag_min"], background="fixed", bg_rate=r["bg_rate"], area=area,
                edge_correction=bool(r.get("edge_correction", False)))
    P = run["final_params"]
    rho = P["K"] * 10.0 ** (P["alpha"] * et.xm)
    F = et._edge_terms(P["d"], P["gamma"], P["q"])[0]
    u = et._dt                                           # t_j - t_i (0 for non-causal pairs)
    s = 1.0 - P["p"]
    om = (np.log((u + P["c"]) / P["c"]) if abs(s) < 1e-9 else ((u + P["c"]) ** s - P["c"] ** s) / s) * et._causal
    return t, r["bg_rate"] * area * t + (rho * F) @ om


def lambda_misd(run):
    df, t, r = load(run)
    a = run["args"]
    mi = M.MISD(t, df["latitude"].values, df["longitude"].values, df["mag"].values, mag_min=r["mag_min"], kernel=r["kernel"],
                productivity=r["productivity"], background="fixed", bg_rate=r["bg_rate"],
                t_bins_per_decade=a.get("t_bins_per_decade", 4), r_bins_per_decade=a.get("r_bins_per_decade", 4),
                mag_bin_width=a.get("mag_bin_width", 1.0), min_parents=a.get("min_parents", 10),
                max_iter=a.get("max_iter", 300)).fit()
    if r["kernel"] != "separable":
        raise SystemExit("GoF for the full-kernel variant is not implemented")
    coef = mi._Kev * (mi.aobs @ mi._f)                   # expected observed offspring per unit of time-kernel mass
    lam = np.zeros(mi.N)
    for j0 in range(0, mi.N, 1000):
        j1 = min(j0 + 1000, mi.N)
        T = t[j0:j1, None] - t[None, :]
        mask = T > 0
        acc = np.zeros_like(T)
        for b in range(mi.nT):
            acc += mi._g[b] * np.clip(T - mi.t_edges[b], 0.0, mi.t_w[b])
        lam[j0:j1] = (acc * mask * coef[None, :]).sum(axis=1)
    return t, r["bg_rate"] * mi.area * t + lam


def summarise(t, Lam):
    d = np.diff(Lam)
    d = np.clip(d, 1e-12, None)
    ks = kstest(d, "expon")
    return d, dict(N=len(t), Lambda_end=float(Lam[-1]), ratio=float(Lam[-1] / len(t)), ks_D=float(ks.statistic), ks_p=float(ks.pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results"); ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seqs", default="tohoku,kumamoto,noto"); ap.add_argument("--models", default="etas,misd")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "tabeli"), exist_ok=True); os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)
    seqs = a.seqs.split(","); res = {}; rows = []
    for s in seqs:
        for m in a.models.split(","):
            run = json.load(open(os.path.join(a.results_dir, f"{s}_allpairs_kg_{m}_run.json"), encoding="utf-8"))
            t, Lam = (lambda_etas if m == "etas" else lambda_misd)(run)
            d, sm = summarise(t, Lam); res[(s, m)] = (t, Lam, d, sm)
            fm = lambda v, k=3: f"{v:.{k}f}".replace(".", "{,}")
            rows.append([LABEL.get(s, s), m.upper(), sm["N"], fm(sm["Lambda_end"], 1), fm(sm["ratio"], 3), fm(sm["ks_D"], 3),
                         ("$<10^{-3}$" if sm["ks_p"] < 1e-3 else fm(sm["ks_p"], 3))])
            print(f"{s:9s} {m:5s} N={sm['N']:5d}  Lambda(T)={sm['Lambda_end']:.1f}  ratio={sm['ratio']:.3f}  KS D={sm['ks_D']:.3f}  p={sm['ks_p']:.2g}")
    L = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Секвенца & Модел & $N$ & $\\Lambda(T)$ & $\\Lambda(T)/N$ & KS растојание $D$ & $p$-вредност \\\\", "\\midrule"]
    L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows] + ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out_dir, "tabeli", "tab_gof.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    fig, ax = plt.subplots(2, len(seqs), figsize=(5.2 * len(seqs), 7), squeeze=False)
    for k, s in enumerate(seqs):
        for m, col in (("etas", "#d81b60"), ("misd", "#1e88e5")):
            if (s, m) not in res: continue
            t, Lam, _, _ = res[(s, m)]
            ax[0, k].plot(t + 1e-2, Lam, color=col, label=f"{m.upper()}  $\\Lambda(t)$")
            ax[1, k].plot(t + 1e-2, (np.arange(1, len(t) + 1) - Lam) / np.sqrt(len(t)), color=col, label=m.upper())
        ax[0, k].plot(t + 1e-2, np.arange(1, len(t) + 1), "k--", lw=1, label="observed $N(t)$")
        ax[0, k].set(xscale="log", title=LABEL.get(s, s), ylabel="cumulative number", xlabel="days since first event")
        ax[1, k].axhline(0, color="k", lw=.6); ax[1, k].set(xscale="log", ylabel=r"$(N-\Lambda)/\sqrt{N_{tot}}$", xlabel="days since first event")
        ax[0, k].legend(fontsize=8); ax[0, k].grid(alpha=.3); ax[1, k].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_gof_kompenzator.pdf")); plt.close()
    fig, ax = plt.subplots(1, len(seqs), figsize=(5.2 * len(seqs), 4.6), squeeze=False)
    for k, s in enumerate(seqs):
        for m, col in (("etas", "#d81b60"), ("misd", "#1e88e5")):
            if (s, m) not in res: continue
            d = np.sort(res[(s, m)][2]); n = len(d)
            q = -np.log(1 - (np.arange(1, n + 1) - 0.5) / n)
            ax[0, k].loglog(q, d, ".", ms=2.5, color=col, label=m.upper())
        lim = [1e-5, 15]; ax[0, k].plot(lim, lim, "k--", lw=.8)
        ax[0, k].set(title=LABEL.get(s, s), xlabel="quantiles of the unit exponential", ylabel=r"rescaled intervals $\Delta\Lambda$")
        ax[0, k].legend(markerscale=4); ax[0, k].grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_gof_qq.pdf")); plt.close()
    json.dump({f"{s}_{m}": v[3] for (s, m), v in res.items()}, open(os.path.join(a.out_dir, "gof_sumarno.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
