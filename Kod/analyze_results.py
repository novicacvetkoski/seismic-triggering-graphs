"""
analyze_results.py  --  Chapter 6 of the thesis (results tables, figures, forest realisations)
================================================================================================
Reads the outputs of seismic_kg_etas.py / seismic_kg_misd.py in `results/` (nodes CSV, pairs NPZ, run JSON,
trace CSV) and writes LaTeX tables (booktabs `tabular` bodies, ready for \\input) and figures:

  tabeli/tab_etas_param.tex      ETAS parameters + diagnostics per sequence                (6.1)
  tabeli/tab_misd_param.tex      MISD kernel read-out + diagnostics per sequence            (6.1)
  tabeli/tab_hubovi_<seq>.tex    top-10 events by weighted out-strength, both models        (6.2)
  tabeli/tab_najgolem.tex        parentage of the largest event of each sequence            (6.2)
  tabeli/tab_grafovi.tex         graph summary (nodes, edges, strength, roots)               (6.2)
  tabeli/tab_sporedba.tex        ETAS vs MISD agreement metrics                              (6.3)
  tabeli/tab_osetlivost.tex      roots / sum P(bg) / agreement vs imposed background         (6.4)
  tabeli/tab_sumi.tex            stochastic forests: family statistics with 95 % intervals   (6.2/6.4)
  tabeli/tab_kaskadi.tex         type A/B share and branching measures from the forests        (6.2)
  tabeli/tab_varijanti.tex       ETAS without edge correction; Tohoku with Mw 9.0 kept          (6.4)
  tabeli/tab_mc.tex              completeness threshold +0.5 (runs from run_mc_sensitivity.py)  (6.4)
  tabeli/tab_historija.tex       roots with pre-sequence events as history (run_history.py)     (6.2)
  tabeli/tab_odbacheni.tex, tab_skratuvanje.tex   discarded probability mass; stability under truncation (6.2)
  figures/fig_graf_<seq>.pdf     readable family skeleton (largest-strength events, ancestry of the largest event)
  figures/fig_jacina_magnituda.pdf, figures/fig_jacina_sporedba.pdf
  rezultati_sumarno.json         all numbers in one place

Usage (from Kod/):   python analyze_results.py --results-dir results --seqs tohoku,kumamoto,noto
Dependencies: numpy pandas scipy matplotlib
"""
import argparse, json, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"tohoku": "Tōhoku", "kumamoto": "Kumamoto", "noto": "Noto"}
VARIANTS = [("_bg_x0.5", r"$\mu\times0{,}5$"), ("", r"$\mu$ (основна)"), ("_bg_x2", r"$\mu\times2$"),
            ("_bg_mean", r"$\mu$ (средна)"), ("_full", "целосен кернел")]
THR = 0.05


def fm(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{d}f}".replace(".", "{,}")


def fsci(v, d=2):
    if v is None or not np.isfinite(v):
        return "--"
    m, e = f"{v:.{d}e}".split("e")
    return f"${m.replace('.', '{,}')}\\cdot10^{{{int(e)}}}$"


class Run:
    def __init__(self, res, prefix, model):
        b = os.path.join(res, f"{prefix}_kg_{model}")
        self.ok = os.path.exists(b + "_run.json")
        if not self.ok:
            return
        self.model, self.prefix = model, prefix
        self.run = json.load(open(b + "_run.json", encoding="utf-8"))
        self.nodes = pd.read_csv(b + "_nodes.csv")
        z = np.load(b + "_pairs.npz")
        self.N = int(z["N"])
        self.par, self.chi, self.w = z["parent"].astype(np.int64), z["child"].astype(np.int64), z["w"].astype(float)
        self.wbg = z["w_bg"].astype(float)
        self.s = self.nodes["out_strength"].values
        self.best = self.nodes["best_parent"].values
        self.mag = self.nodes["mag"].values
        self.date = self.nodes["time"].astype(str).str[:10].values
        self.eid = self.nodes["event_id"].astype(str).values
        self.trace = pd.read_csv(b + "_trace.csv") if os.path.exists(b + "_trace.csv") else None

    def edges(self, thr=THR):
        m = self.w >= thr
        return set((self.par[m] * self.N + self.chi[m]).tolist()), m

    @property
    def roots(self):
        return int((self.wbg > 0.5).sum())


def compare(A, B):
    ea, ma = A.edges(); eb, mb = B.edges()
    inter = ea & eb
    jac = len(inter) / max(len(ea | eb), 1)
    da = dict(zip((A.par[ma] * A.N + A.chi[ma]).tolist(), A.w[ma]))
    db = dict(zip((B.par[mb] * B.N + B.chi[mb]).tolist(), B.w[mb]))
    sh = sorted(inter)
    rho_w = spearmanr([da[k] for k in sh], [db[k] for k in sh])[0] if len(sh) > 2 else np.nan
    both = (A.best >= 0) & (B.best >= 0)
    same = float((A.best[both] == B.best[both]).mean()) if both.any() else np.nan
    rho_s = spearmanr(A.s, B.s)[0]
    ta, tb = set(np.argsort(-A.s)[:10]), set(np.argsort(-B.s)[:10])
    ra, rb = set(np.where(A.wbg > 0.5)[0]), set(np.where(B.wbg > 0.5)[0])
    return dict(edges_a=len(ea), edges_b=len(eb), jaccard=jac, rho_w=rho_w, same_parent=same, rho_s=rho_s,
                top10=len(ta & tb), roots_a=len(ra), roots_b=len(rb), roots_shared=len(ra & rb),
                sbg_a=float(A.wbg.sum()), sbg_b=float(B.wbg.sum()))


# ---------------------------------------------------------------- stochastic forests
class Sampler:
    def __init__(self, R):
        o = np.lexsort((R.par, R.chi))
        self.par, self.chi, self.w = R.par[o], R.chi[o], R.w[o]
        self.N, self.wbg = R.N, R.wbg
        cs = np.cumsum(self.w)
        first = np.searchsorted(self.chi, np.arange(self.N), side="left")
        last = np.searchsorted(self.chi, np.arange(self.N), side="right")
        before = np.where(first > 0, cs[np.maximum(first - 1, 0)], 0.0) if len(cs) else np.zeros(self.N)
        self.cs, self.first, self.last, self.before = cs, first, last, before
        tot = np.zeros(self.N); np.add.at(tot, self.chi, self.w)
        self.total = self.wbg + tot            # pairs below the export cutoff are renormalised away
        self.has = last > first

    def draw(self, rng):
        U = rng.random(self.N) * self.total
        parent = np.full(self.N, -1, dtype=np.int64)
        pick = self.has & (U >= self.wbg)
        j = np.where(pick)[0]
        tgt = self.before[j] + (U[j] - self.wbg[j])
        idx = np.minimum(np.searchsorted(self.cs, tgt, side="left"), self.last[j] - 1)
        idx = np.maximum(idx, self.first[j])
        parent[j] = self.par[idx]
        return parent


def forest_stats(parent, mag, big):
    N = len(parent)
    root = np.arange(N); depth = np.zeros(N, dtype=np.int64)
    for j in range(N):
        p = parent[j]
        if p >= 0:
            root[j] = root[p]; depth[j] = depth[p] + 1
    size = np.bincount(root, minlength=N)
    fam = np.where(size >= 2)[0]
    mmax = np.full(N, -np.inf); np.maximum.at(mmax, root, mag)
    root_is_max = float((mag[fam] >= mmax[fam] - 1e-9).mean()) if len(fam) else np.nan
    nchild = np.bincount(parent[parent >= 0], minlength=N)
    # Gentili-type classification of the families: mainshock = largest event, largest aftershock = largest LATER member
    order = np.lexsort((np.arange(N), root))
    typeA = typeB = bigA = bigB = 0
    rs, rstart = np.unique(root[order], return_index=True)
    rend = np.r_[rstart[1:], N]
    for a_, b_ in zip(rstart, rend):
        if b_ - a_ < 2:
            continue
        mem = order[a_:b_]; mm = mag[mem]
        k = int(np.argmax(mm))                      # first (earliest) among equal magnitudes
        if k == len(mem) - 1:
            continue                                # largest event is last: no aftershock, class undefined
        dM = mm[k] - mm[k + 1:].max()
        isbig = (b_ - a_) >= 20
        if dM < 1.0: typeA += 1; bigA += isbig
        else: typeB += 1; bigB += isbig
    return dict(roots=int((parent < 0).sum()), fam2=len(fam), maxsize=int(size.max()), maxdepth=int(depth.max()),
                root_is_max=root_is_max, mean_branch=float(nchild[nchild > 0].mean()) if (nchild > 0).any() else 0.0,
                big_root=int(root[big]), big_depth=int(depth[big]), big_famsize=int(size[root[big]]),
                big_root_mag=float(mag[root[big]]), typeA=typeA, typeB=typeB, bigA=bigA, bigB=bigB,
                max_children=int(nchild.max()), big_children=int(nchild[big]))


def run_forests(R, n, seed):
    S, rng = Sampler(R), np.random.default_rng(seed)
    big = int(np.argmax(R.mag))
    rows = [forest_stats(S.draw(rng), R.mag, big) for _ in range(n)]
    return pd.DataFrame(rows), big


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan,) * 3


def mci(x, d=1):
    m, lo, hi = ci(x)
    return f"{fm(m, d)} [{fm(lo, d)}; {fm(hi, d)}]"


# ---------------------------------------------------------------- LaTeX helpers
def write_tab(path, header, rows, colspec):
    L = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    L += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    L += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")



# ---------------------------------------------------------------- additional helpers (variants, Mc, history, truncation)
def find_run(res, prefixes, model):
    for p in prefixes:
        R = Run(res, p, model)
        if R.ok:
            return R
    return None


def params_of(R):
    if R.model == "etas":
        p = R.run["final_params"]; return p["alpha"], p["p"], p["q"]
    k = R.run["kernel_readout"]; return k["alpha_log10"], k["p"], k["s"] / 2 if k["s"] else np.nan


def common_compare(A, B):
    """Compare two runs on the events they share (matched by event_id)."""
    ia = {e: i for i, e in enumerate(A.eid)}; ib = {e: i for i, e in enumerate(B.eid)}
    common = [e for e in B.eid if e in ia]
    a_idx = np.array([ia[e] for e in common]); b_idx = np.array([ib[e] for e in common])
    rho = spearmanr(A.s[a_idx], B.s[b_idx])[0]
    pairs = [(A.eid[A.best[i]], B.eid[B.best[j]]) for i, j in zip(a_idx, b_idx) if A.best[i] >= 0 and B.best[j] >= 0]
    same = float(np.mean([x == y for x, y in pairs])) if pairs else np.nan
    return dict(n_common=len(common), rho_s=rho, same_parent=same)


def dropped_mass(R):
    tot = R.wbg.copy(); np.add.at(tot, R.chi, R.w)
    d = np.clip(1.0 - tot, 0.0, None)
    return dict(mean=float(d.mean()), p99=float(np.percentile(d, 99)), max=float(d.max()), share_gt_1pct=float((d > 0.01).mean()))


def with_cutoff(R, cut):
    import copy
    R2 = copy.copy(R); m = R.w >= cut
    R2.par, R2.chi, R2.w = R.par[m], R.chi[m], R.w[m]
    return R2


def plot_skeleton(seq_label, E, M, out_pdf, topk=25, thr=THR):
    """Readable 'family skeleton': the topk events by out-strength (union over both models), the best-parent ancestry of
    the largest event, and the probability edges (rho >= thr) between them. x = chronological order, y = magnitude."""
    nodes = set()
    for R in (E, M):
        nodes |= set(np.argsort(-R.s)[:topk].tolist())
    big = int(np.argmax(E.mag)); nodes.add(big)
    for R in (E, M):
        j = big
        while R.best[j] >= 0:
            j = int(R.best[j]); nodes.add(j)
    nodes = np.array(sorted(nodes)); pos = {int(n): (k, E.mag[n]) for k, n in enumerate(nodes)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), sharey=True)
    for ax, R, name in zip(axes, (E, M), ("ETAS", "MISD")):
        mk = np.isin(R.par, nodes) & np.isin(R.chi, nodes) & (R.w >= thr)
        for p_, c_, w_ in zip(R.par[mk], R.chi[mk], R.w[mk]):
            bp = R.best[c_] == p_
            ax.annotate("", xy=pos[int(c_)], xytext=pos[int(p_)], arrowprops=dict(
                arrowstyle="-|>", color="#1e88e5" if bp else "#ff6d00", alpha=min(1, 0.25 + 0.75 * w_), lw=0.4 + 2.6 * w_,
                connectionstyle="arc3,rad=0.22", shrinkA=3, shrinkB=3))
        xs = [pos[int(n)][0] for n in nodes]; ys = [pos[int(n)][1] for n in nodes]
        sc = ax.scatter(xs, ys, s=[25 + 9 * R.s[n] for n in nodes], c=[R.wbg[n] for n in nodes], cmap="coolwarm", vmin=0, vmax=1,
                        edgecolor="k", lw=.5, zorder=3)
        for n in sorted(nodes, key=lambda n: -R.s[n])[:7]:
            ax.annotate(f"{R.date[n]}\nM{E.mag[n]:.1f}", pos[int(n)], textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.set(title=f"{name}", xlabel="chronological order of the selected events"); ax.grid(alpha=.3)
    axes[0].set_ylabel("magnitude")
    fig.suptitle(f"{seq_label}: events with the largest out-strength and the ancestry of the largest event "
                 f"(blue = most probable parent, orange = other candidate, node size ~ $s^{{out}}$, colour = $P$(background))", fontsize=9)
    plt.tight_layout(); plt.savefig(out_pdf); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--seqs", default="tohoku,kumamoto,noto")
    ap.add_argument("--n-forests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    seqs = a.seqs.split(",")
    os.makedirs(os.path.join(a.out_dir, "tabeli"), exist_ok=True)
    os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)
    T = lambda n: os.path.join(a.out_dir, "tabeli", n)
    out = {}

    P = {(s, m): Run(a.results_dir, s, m) for s in seqs for m in ("etas", "misd")}
    lab = lambda s: LABEL.get(s, s)

    # ---- 6.1 parameters ----
    rows = []
    for s in seqs:
        R = P[(s, "etas")]
        if not R.ok: continue
        p = R.run["final_params"]; ll = R.trace["loglik"].iloc[-1] if R.trace is not None else np.nan
        rows.append([lab(s), fsci(p["K"], 2), fm(p["c"], 4), fm(p["p"], 3), fm(p["alpha"], 3), fm(p["d"], 3),
                     fm(p["gamma"], 3), fm(p["q"], 3), R.run["n_em_iterations"], "да" if R.run["converged"] else "не",
                     f"{ll:,.1f}".replace(",", "\\,").replace(".", "{,}"), fm(R.wbg.sum(), 1), R.roots,
                     "да" if R.run["verification"]["all_passed"] else "НЕ"])
        out[f"etas_{s}"] = dict(params=p, iters=R.run["n_em_iterations"], converged=R.run["converged"], loglik=float(ll),
                                sum_bg=float(R.wbg.sum()), roots=R.roots, checks=R.run["verification"]["all_passed"])
    write_tab(T("tab_etas_param.tex"), ["Секвенца", "$K$", "$c$", "$p$", r"$\alpha$", "$d$", r"$\gamma$", "$q$",
              "Итер.", "Конв.", r"$\ell$", r"$\sum\varphi$", "Корени", "Пров."], rows, "@{}l" + "r" * 13 + "@{}")
    rows = []
    for s in seqs:
        R = P[(s, "misd")]
        if not R.ok: continue
        k = R.run["kernel_readout"]
        rows.append([lab(s), fm(k["alpha_log10"], 3), fm(k["alpha_log10_binned_check"], 3), fm(k["p"], 3), fm(k["s"], 3),
                     fm(k["s"] / 2 if k["s"] else np.nan, 2), R.run["n_em_iterations"], "да" if R.run["converged"] else "не",
                     fm(R.wbg.sum(), 1), R.roots, "да" if R.run["verification"]["all_passed"] else "НЕ"])
        out[f"misd_{s}"] = dict(kernel=k, iters=R.run["n_em_iterations"], converged=R.run["converged"],
                                sum_bg=float(R.wbg.sum()), roots=R.roots, checks=R.run["verification"]["all_passed"])
    write_tab(T("tab_misd_param.tex"), ["Секвенца", r"$\alpha$", r"$\alpha$ (инт.)", "$p$", "$s$", "$s/2$", "Итер.",
              "Конв.", r"$\sum\varphi$", "Корени", "Пров."], rows, "@{}l" + "r" * 10 + "@{}")

    # ---- 6.2 graphs ----
    rows = []
    for s in seqs:
        for m in ("etas", "misd"):
            R = P[(s, m)]
            if not R.ok: continue
            _, mm = R.edges()
            nbest = int((R.best >= 0).sum())
            rows.append([lab(s), m.upper(), R.N, int(mm.sum()), fm(R.s.mean(), 2), fm(R.s.max(), 1),
                         fm(R.wbg.sum(), 1), R.roots])
    write_tab(T("tab_grafovi.tex"), ["Секвенца", "Модел", "Јазли", r"Ребра ($\rho\ge0{,}05$)", r"Средна $s^{\mathrm{out}}$",
              r"Макс. $s^{\mathrm{out}}$", r"$\sum\varphi$", "Корени"], rows, "@{}llrrrrrr@{}")
    big_rows = []
    for s in seqs:
        E, M = P[(s, "etas")], P[(s, "misd")]
        if not (E.ok and M.ok): continue
        o = np.argsort(-E.s)[:10]
        rows = [[k + 1, E.date[i], fm(E.mag[i], 1), fm(E.s[i], 1), fm(M.s[i], 1),
                 int(np.where(np.argsort(-M.s) == i)[0][0] + 1), fm(E.wbg[i], 3)] for k, i in enumerate(o)]
        write_tab(T(f"tab_hubovi_{s}.tex"), ["\\#", "Датум", "$M$", r"$s^{\mathrm{out}}$ ETAS", r"$s^{\mathrm{out}}$ MISD",
                  "Ранг MISD", r"$\varphi$ (ETAS)"], rows, "@{}rlrrrrr@{}")
        i = int(np.argmax(E.mag))
        def top(R):
            m_ = R.chi == i; o2 = np.argsort(-R.w[m_])[:2]
            return [(R.date[R.par[m_][k]], R.mag[R.par[m_][k]], R.w[m_][k]) for k in o2]
        te, tm = top(E), top(M)
        fmt = lambda t: "; ".join(f"{d} (M{fm(mg, 1)}): {fm(w, 2)}" for d, mg, w in t) or "--"
        big_rows.append([lab(s), f"{E.date[i]} (M{fm(E.mag[i], 1)})", fm(E.wbg[i], 3), fm(M.wbg[i], 3), fmt(te), fmt(tm),
                         fm(E.s[i], 1), fm(M.s[i], 1)])
    write_tab(T("tab_najgolem.tex"), ["Секвенца", "Најголем настан", r"$\varphi$ ETAS", r"$\varphi$ MISD",
              "Најверојатни родители ETAS", "Најверојатни родители MISD", r"$s^{\mathrm{out}}$ ETAS", r"$s^{\mathrm{out}}$ MISD"],
              big_rows, "@{}lp{2.6cm}rrp{3.6cm}p{3.6cm}rr@{}")

    # ---- 6.3 comparison ----
    rows = []
    for s in seqs:
        E, M = P[(s, "etas")], P[(s, "misd")]
        if not (E.ok and M.ok): continue
        c = compare(E, M); out[f"cmp_{s}"] = c
        rows.append([lab(s), f"{c['edges_a']:,}".replace(",", "\\,") + " / " + f"{c['edges_b']:,}".replace(",", "\\,"),
                     fm(c["jaccard"], 3), fm(c["rho_w"], 3), fm(100 * c["same_parent"], 1) + "\\,\\%", fm(c["rho_s"], 3),
                     f"{c['top10']}/10", f"{c['roots_a']} / {c['roots_b']} / {c['roots_shared']}",
                     fm(c["sbg_a"], 1) + " / " + fm(c["sbg_b"], 1)])
    write_tab(T("tab_sporedba.tex"), ["Секвенца", "Ребра ETAS / MISD", "Жакард", r"$\rho_S$ тежини", "Ист родител",
              r"$\rho_S$ $s^{\mathrm{out}}$", "Топ-10", "Корени E / M / заед.", r"$\sum\varphi$ E / M"], rows,
              "@{}lrrrrrrrr@{}")
    fig, ax = plt.subplots(1, len(seqs), figsize=(5 * len(seqs), 4.4), squeeze=False)
    for k, s in enumerate(seqs):
        E, M = P[(s, "etas")], P[(s, "misd")]
        if not (E.ok and M.ok): continue
        ax[0, k].loglog(np.maximum(E.s, 1e-4), np.maximum(M.s, 1e-4), ".", ms=3, alpha=.5)
        lim = [1e-4, max(E.s.max(), M.s.max()) * 1.5]; ax[0, k].plot(lim, lim, "k--", lw=.8)
        ax[0, k].set(title=f"{lab(s)}  (Spearman {out[f'cmp_{s}']['rho_s']:.3f})", xlabel=r"$s^{out}$ ETAS", ylabel=r"$s^{out}$ MISD")
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_jacina_sporedba.pdf")); plt.close()
    fig, ax = plt.subplots(1, len(seqs), figsize=(5 * len(seqs), 4.2), squeeze=False)
    for k, s in enumerate(seqs):
        for m, c in (("etas", "#d81b60"), ("misd", "#1e88e5")):
            R = P[(s, m)]
            if R.ok: ax[0, k].semilogy(R.mag, np.maximum(R.s, 1e-4), ".", ms=3, alpha=.4, color=c, label=m.upper())
        ax[0, k].set(title=lab(s), xlabel="магнитуда", ylabel=r"$s^{out}$"); ax[0, k].legend(markerscale=3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_jacina_magnituda.pdf")); plt.close()

    # ---- 6.4 sensitivity ----
    rows = []
    for s in seqs:
        for m in ("etas", "misd"):
            base = P[(s, m)]
            for suf, name in VARIANTS:
                R = base if suf == "" else Run(a.results_dir, s + suf, m)
                if not getattr(R, "ok", False): continue
                both = (R.best >= 0) & (base.best >= 0)
                rows.append([lab(s), m.upper(), name, fm(R.wbg.sum(), 1), R.roots,
                             fm(spearmanr(R.s, base.s)[0], 3), fm(100 * (R.best[both] == base.best[both]).mean(), 1) + "\\,\\%"])
    write_tab(T("tab_osetlivost.tex"), ["Секвенца", "Модел", "Позадина", r"$\sum\varphi$", "Корени",
              r"$\rho_S$ ($s^{\mathrm{out}}$ vs основна)", "Ист родител vs основна"], rows, "@{}llllrrr@{}")

    # ---- forests (tab_sumi) + cascade measures (tab_kaskadi) ----
    rows, rows_k = [], []
    for s in seqs:
        for m in ("etas", "misd"):
            R = P[(s, m)]
            if not R.ok: continue
            df, big = run_forests(R, a.n_forests, a.seed)
            top = df["big_root"].value_counts()
            r0, share = int(top.index[0]), float(top.iloc[0] / len(df))
            rows.append([lab(s), m.upper(), mci(df["roots"], 1), mci(df["fam2"], 1), mci(df["maxsize"], 0),
                         mci(df["maxdepth"], 1), fm(100 * np.nanmean(df["root_is_max"]), 1) + "\\,\\%",
                         f"{R.date[r0]} (M{fm(R.mag[r0], 1)}), {fm(100 * share, 0)}\\,\\%", mci(df["big_depth"], 1)])
            nA, nB, bA, bB = df["typeA"].sum(), df["typeB"].sum(), df["bigA"].sum(), df["bigB"].sum()
            rows_k.append([lab(s), m.upper(), fm(100 * nA / max(nA + nB, 1), 1) + "\\,\\%",
                           fm(100 * bA / max(bA + bB, 1), 1) + "\\,\\%" if bA + bB else "--",
                           mci(df["mean_branch"], 2), mci(df["max_children"], 0), mci(df["big_children"], 1)])
            out[f"forest_{s}_{m}"] = dict(roots=ci(df["roots"]), fam2=ci(df["fam2"]), maxsize=ci(df["maxsize"]),
                                         maxdepth=ci(df["maxdepth"]), root_is_max=float(np.nanmean(df["root_is_max"])),
                                         big_root=dict(idx=r0, date=str(R.date[r0]), mag=float(R.mag[r0]), share=share),
                                         typeA_share=float(nA / max(nA + nB, 1)), n_forests=a.n_forests)
    write_tab(T("tab_sumi.tex"), ["Секвенца", "Модел", "Корени", "Семејства ($\\ge2$)", "Најголемо семејство",
              "Макс. длабочина", "Корен = најголем", "Корен на најголемиот настан", "Длабочина на најголемиот"], rows,
              "@{}llrrrrrp{3.4cm}r@{}")
    write_tab(T("tab_kaskadi.tex"), ["Секвенца", "Модел", "Тип A (сите семејства)", r"Тип A (семејства $\ge20$)",
              "Средна разгранетост", "Макс. деца на еден настан", "Деца на најголемиот настан"], rows_k, "@{}llrrrrr@{}")

    # ---- sensitivity: ETAS without edge correction, Tohoku with Mw 9.0 kept ----
    rows = []
    for s in seqs:
        for m in ("etas", "misd"):
            base = P[(s, m)]
            if not base.ok: continue
            for suf, name in (("", "основна"), ("_noedge", "без корекција на рабовите"), ("_keepW", r"$M_w\,9{,}0$ задржана"), ("_tight", "построг праг на запирање ($10^{-8}$)")):
                R = base if suf == "" else Run(a.results_dir, s + suf, m)
                if not getattr(R, "ok", False): continue
                al, pp, qq = params_of(R)
                both = (R.best >= 0) & (base.best >= 0)
                rows.append([lab(s), m.upper(), name, fm(al, 3), fm(pp, 3), fm(qq, 2), fm(R.wbg.sum(), 1), R.roots, fm(R.s.max(), 1),
                             fm(spearmanr(R.s, base.s)[0], 3), fm(100 * (R.best[both] == base.best[both]).mean(), 1) + "\\,\\%"])
    write_tab(T("tab_varijanti.tex"), ["Секвенца", "Модел", "Варијанта", r"$\alpha$", "$p$", r"$q$ ($s/2$)", r"$\sum\varphi$", "Корени",
              r"Макс. $s^{\mathrm{out}}$", r"$\rho_S$ vs основна", "Ист родител vs основна"], rows, "@{}lllrrrrrrrr@{}")

    # ---- sensitivity: completeness threshold +0.5 ----
    rows = []
    for s in seqs:
        for m in ("etas", "misd"):
            base = P[(s, m)]
            if not base.ok: continue
            mc0 = base.run["resolved"]["mag_min"]; mc1 = round(mc0 + 0.5, 2)
            R = find_run(a.results_dir, [f"{s}_mc{mc1}", f"{s}_mc{str(mc1).replace('.', '_')}"], m)
            for nm, X in ((fm(mc0, 1), base), (fm(mc1, 1), R)):
                if X is None: continue
                al, pp, qq = params_of(X)
                cc = common_compare(base, X) if X is not base else dict(rho_s=np.nan, same_parent=np.nan)
                rows.append([lab(s), m.upper(), nm, X.N, fsci(X.run["resolved"]["bg_rate"], 2), fm(X.wbg.sum(), 1), X.roots,
                             fm(al, 3), fm(pp, 3), fm(qq, 2), fm(cc["rho_s"], 3),
                             "--" if X is base else fm(100 * cc["same_parent"], 1) + "\\,\\%"])
    write_tab(T("tab_mc.tex"), ["Секвенца", "Модел", "$M_c$", "$N$", r"$\mu$", r"$\sum\varphi$", "Корени", r"$\alpha$", "$p$",
              r"$q$ ($s/2$)", r"$\rho_S$ (заеднички)", "Ист родител (заеднички)"], rows, "@{}llrrrrrrrrrr@{}")

    # ---- history: do the roots depend on the observation window? ----
    rows = []
    for s in seqs:
        for m in ("etas", "misd"):
            A_ = P[(s, m)]; H = Run(a.results_dir, f"{s}_hist", m)
            if not (A_.ok and H.ok): continue
            win = np.array([not str(e).startswith("pre_") for e in H.eid])
            roots_h = set(H.eid[win & (H.wbg > 0.5)]); roots_p = set(A_.eid[A_.wbg > 0.5])
            top = int(np.argmax(np.where(A_.wbg > 0.5, A_.mag, -1)))
            hid = {e: i for i, e in enumerate(H.eid)}
            rows.append([lab(s), m.upper(), int((~win).sum()), fm(A_.wbg.sum(), 1) + " / " + fm(H.wbg[win].sum(), 1),
                         f"{len(roots_p)} / {len(roots_h)} / {len(roots_p & roots_h)}",
                         f"{A_.date[top]} (M{fm(A_.mag[top], 1)}): {fm(A_.wbg[top], 3)} $\\to$ {fm(H.wbg[hid[A_.eid[top]]], 3)}",
                         fm(A_.wbg[0], 3) + " $\\to$ " + fm(H.wbg[hid[A_.eid[0]]], 3)])
    write_tab(T("tab_historija.tex"), ["Секвенца", "Модел", "Настани пред прозорецот", r"$\sum\varphi$ (основна / со историја)",
              "Корени (основна / со историја / заед.)", r"Највисок корен: $\varphi$ основна $\to$ со историја",
              r"$\varphi$ на првиот настан"], rows, "@{}llrrrp{4.2cm}r@{}")

    # ---- discarded probability mass and stability of the forest measures under truncation ----
    rows, rows_t = [], []
    for s in seqs:
        for m in ("etas", "misd"):
            R = P[(s, m)]
            if not R.ok: continue
            d = dropped_mass(R)
            rows.append([lab(s), m.upper(), fsci(d["mean"], 1), fsci(d["p99"], 1), fsci(d["max"], 1),
                         fm(100 * d["share_gt_1pct"], 2) + "\\,\\%"])
            out[f"dropped_{s}_{m}"] = d
            R_all = find_run(a.results_dir, [f"{s}_allpairs"], m)
            variants = [("1e-2", with_cutoff(R, 1e-2)), ("1e-3", with_cutoff(R, 1e-3)), ("1e-4 (основна)", R)]
            if R_all is not None: variants.append(("$10^{-8}$ (сите парови)", R_all))
            for nm, X in variants:
                dfx, _ = run_forests(X, max(a.n_forests // 3, 50), a.seed)
                rows_t.append([lab(s), m.upper(), nm, mci(dfx["roots"], 1), mci(dfx["maxsize"], 0), mci(dfx["maxdepth"], 1),
                               fm(100 * float(dfx["big_root"].value_counts().iloc[0] / len(dfx)), 0) + "\\,\\%"])
    write_tab(T("tab_odbacheni.tex"), ["Секвенца", "Модел", "Средно", "99-ти перцентил", "Максимум", "Настани со $>1\\,\\%$ отфрлено"],
              rows, "@{}llrrrr@{}")
    write_tab(T("tab_skratuvanje.tex"), ["Секвенца", "Модел", "Праг", "Корени", "Најголемо семејство", "Макс. длабочина",
              "Удел на најчестиот корен на најголемиот"], rows_t, "@{}lllrrrr@{}")

    # ---- readable graphs ----
    for s in seqs:
        E, M = P[(s, "etas")], P[(s, "misd")]
        if E.ok and M.ok:
            plot_skeleton(lab(s), E, M, os.path.join(a.out_dir, "figures", f"fig_graf_{s}.pdf"))

    # ---- which runs were found? (a missing run silently gives an empty table row) ----
    print("\nRun availability (primary + variants):")
    for s in seqs:
        base = P[(s, "etas")]
        mc1 = round(base.run["resolved"]["mag_min"] + 0.5, 2) if base.ok else None
        names = ["", "_noedge", "_keepW", "_tight", "_hist", "_allpairs", "_bg_x0.5", "_bg_x2", "_bg_mean", "_full"] + ([f"_mc{mc1}"] if mc1 else [])
        found = {}
        for suf in names:
            ok = []
            for m in ("etas", "misd"):
                pref = [s + suf] + ([s + suf.replace(".", "_")] if "." in suf else [])
                ok.append(m if find_run(a.results_dir, pref, m) is not None else None)
            found[suf or "primary"] = "/".join(x for x in ok if x) or "MISSING"
        print(f"  {s:9s} " + "  ".join(f"{k}:{v}" for k, v in found.items()))
    json.dump(out, open(os.path.join(a.out_dir, "rezultati_sumarno.json"), "w"), indent=1, default=float)
    print("done; tables in", T(""))


if __name__ == "__main__":
    main()
