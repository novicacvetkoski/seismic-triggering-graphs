"""
seismic_kg_etas.py   (v2)
=========================
Space-time ETAS (Ogata 1988, 1998) fitted by the stochastic-declustering EM algorithm
(Zhuang, Ogata & Vere-Jones 2002), producing the same seismic "family tree" graph as
seismic_kg_misd.py (shared code in kg_common.py).

    lambda(t,x,y) = mu(x,y) + sum_{i: t_i<t} K 10^(alpha (m_i - mc)) (t - t_i + c)^(-p)
                              * (q-1)/(pi d_i) (1 + r^2/d_i)^(-q),     d_i = d 10^(gamma (m_i - mc))

What changed relative to v1
  * Distances in km (same projection as MISD); d is in km^2. Rates are events/day/km^2.
  * BACKGROUND: by default IMPOSED (--background fixed): mu = --bg-rate, the long-term rate measured
    in the years before the sequence (estimate_background_rate.py) -- identical to the rate given to
    MISD, so the two models differ only in the triggering kernel. --background kde estimates a smooth
    spatial background by adaptive KDE, now in correct units (events/day/km^2).
  * M-step: exact ANALYTIC gradient and log-parametrisation of (K, c, d), so L-BFGS-B converges in
    far fewer iterations (v1 used finite differences: ~8 objective evaluations per gradient).
  * Convergence is monitored with the TRUE log-likelihood sum_j log lambda_j - int(mu) - compensator,
    and every iteration's parameters are saved (trace CSV + plot).
  * After the last iteration a final E-step is run, so the saved probabilities always correspond to
    the saved parameters (in v1 they lagged by one M-step when the iteration cap was reached).
  * Edges are "P(parent)" edges with an `is_best_parent` flag (no "indirect" label); node importance =
    weighted out-strength; probability checks, pairs file and run JSON are written (see kg_common.py).

Known limitations (state them in the thesis)
  * The compensator integrates the spatial kernel over the whole plane (boundary edge effects of the
    study box are neglected); epicentral distances only (no fault geometry for large events).
  * For d the data can only resolve scales above the location error (~1 km); very small fitted d
    values are weakly identified.

Dependencies:  pip install pandas numpy scipy networkx matplotlib
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

import kg_common as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("etas")

LN10 = np.log(10.0)

# ── configuration ───────────────────────────────────────────────────────────────
ETAS_MAX_ITER = 30
ETAS_TOL = 1e-5                  # relative change of the log-likelihood (parameters then stable to ~0.3%)
ETAS_OPTIMIZER_MAXITER = 50      # L-BFGS-B iterations per M-step (cheap now: analytic gradient)
MSTEP_PAIR_CUTOFF = 1e-8         # pairs with w below this are left out of the M-step sums
BG_NN = 15
BG_MIN_BANDWIDTH_KM = 3.0
INIT_PARAMS = dict(K=0.02, c=0.02, p=1.1, alpha=0.5, d=1.0, gamma=0.4, q=1.7)
PARAM_NAMES = ["K", "c", "p", "alpha", "d", "gamma", "q"]
# bounds on z = (log K, log c, p, alpha, log d, gamma, q).  p may be < 1: swarms decay slower than 1/t
# (the Omori integral is taken over the finite observation window, so p <= 1 is well defined)
BOUNDS_Z = [(np.log(1e-8), np.log(1e3)), (np.log(1e-5), np.log(10.0)), (0.5, 5.0),
            (0.0, 3.0), (np.log(1e-3), np.log(1e5)), (0.0, 3.0), (1.001, 5.0)]


def z_from_params(P):
    return np.array([np.log(P["K"]), np.log(P["c"]), P["p"], P["alpha"], np.log(P["d"]), P["gamma"], P["q"]])


def params_from_z(z):
    return dict(K=float(np.exp(z[0])), c=float(np.exp(z[1])), p=float(z[2]), alpha=float(z[3]),
                d=float(np.exp(z[4])), gamma=float(z[5]), q=float(z[6]))


class ETAS:
    def __init__(self, times, x, y, mags, mc, background, bg_rate, area,
                 max_iter=ETAS_MAX_ITER, tol=ETAS_TOL, optimizer_maxiter=ETAS_OPTIMIZER_MAXITER,
                 init_params=None, bg_nn=BG_NN, bg_min_bw_km=BG_MIN_BANDWIDTH_KM):
        self.N = len(times)
        self.times = np.asarray(times, float)
        self.x_km, self.y_km = np.asarray(x, float), np.asarray(y, float)
        self.mags = np.asarray(mags, float)
        self.xm = self.mags - mc                     # magnitude above the cutoff
        self.mc = mc
        self.background, self.bg_rate, self.area = background, bg_rate, area
        self.T_span = max(self.times.max() - self.times.min(), 1e-6)
        self.T_end = self.times.max()
        self.max_iter, self.tol, self.optimizer_maxiter = max_iter, tol, optimizer_maxiter
        self.bg_nn, self.bg_min_bw = bg_nn, bg_min_bw_km
        self.params = dict(INIT_PARAMS)
        if init_params:
            self.params.update(init_params)

        N = self.N
        self._causal = np.triu(np.ones((N, N), dtype=bool), k=1)
        dt = -np.subtract.outer(self.times, self.times)          # t_j - t_i
        dx = np.subtract.outer(self.x_km, self.x_km)
        dr2 = dx * dx
        del dx
        dy = np.subtract.outer(self.y_km, self.y_km)
        dr2 += dy * dy
        del dy
        dt[~self._causal] = 0.0
        dr2[~self._causal] = 0.0
        self._dt, self._dr2 = dt, dr2
        if background == "kde":
            self._build_kde_kernel()
        self.W = None
        self.w_bg = None
        self.trace = []

    # ── background ─────────────────────────────────────────────────────────────

    def _build_kde_kernel(self):
        N = self.N
        xy = np.column_stack([self.x_km, self.y_km])
        k = min(self.bg_nn, N - 1)
        kern = np.empty((N, N), dtype=np.float32)
        for s in range(0, N, 1000):
            D = cdist(xy[s:s + 1000], xy)
            h = np.maximum(np.partition(D, k, axis=1)[:, k], self.bg_min_bw)
            kern[s:s + 1000] = (np.exp(-0.5 * (D / h[:, None]) ** 2) / (2 * np.pi * h[:, None] ** 2)).astype(np.float32)
        self._kern = kern

    def _background_rate(self, w_bg):
        """mu_j [events/day/km^2] and the integral of mu over space and time (expected background count)."""
        if self.background == "fixed":
            return np.full(self.N, self.bg_rate), self.bg_rate * self.T_span * self.area
        mu = (w_bg.astype(np.float32) @ self._kern).astype(float) / self.T_span
        return np.maximum(mu, 1e-15), float(w_bg.sum())

    # ── model pieces ───────────────────────────────────────────────────────────

    def _rate_matrix(self, P=None):
        """R[i,j] = triggering intensity at event j due to event i (i<j), events/day/km^2."""
        P = P or self.params
        rho = P["K"] * 10.0 ** (P["alpha"] * self.xm)
        d_i = P["d"] * 10.0 ** (P["gamma"] * self.xm)
        R = (self._dt + P["c"]) ** (-P["p"])
        R *= (P["q"] - 1.0) / (np.pi * d_i[:, None]) * (1.0 + self._dr2 / d_i[:, None]) ** (-P["q"])
        R *= rho[:, None]
        R[~self._causal] = 0.0
        return R

    def _omori_terms(self, c, p):
        """Omega_i = int_c^{tau_i + c} t^(-p) dt and its derivatives w.r.t. c and p.
        Stable for p < 1, p = 1 and p > 1 (series expansion in s = 1 - p near p = 1)."""
        tau = np.maximum(self.T_end - self.times, 0.0)
        b = tau + c
        A, B = np.log(c), np.log(b)
        s = 1.0 - p
        if abs(s) < 1e-4:
            om = (B - A) + s * (B ** 2 - A ** 2) / 2 + s ** 2 * (B ** 3 - A ** 3) / 6 + s ** 3 * (B ** 4 - A ** 4) / 24
            dom_ds = (B ** 2 - A ** 2) / 2 + s * (B ** 3 - A ** 3) / 3 + s ** 2 * (B ** 4 - A ** 4) / 8
        else:
            eA, eB = np.exp(s * A), np.exp(s * B)                      # c^s, b^s
            om = (eB - eA) / s
            dom_ds = (B * eB - A * eA) / s - (eB - eA) / s ** 2
        return om, -c ** (-p) + b ** (-p), -dom_ds

    def _compensator(self, P):
        om = self._omori_terms(P["c"], P["p"])[0]
        return float((P["K"] * 10.0 ** (P["alpha"] * self.xm) * om).sum())

    def _estep(self, mu):
        R = self._rate_matrix()
        tot = mu + R.sum(axis=0)
        self.w_bg = mu / tot
        self.W = R / tot[None, :]
        return tot

    # ── M-step for (K, c, p, alpha, d, gamma, q): weighted MLE with analytic gradient ──

    def _fg(self, z, data):
        """Negative restricted log-likelihood and its exact gradient w.r.t. z, scaled by 1/sum(w)."""
        w, dt, dr2, xk, Wsum, Wx = data
        K, c, p, a, d, g, q = (np.exp(z[0]), np.exp(z[1]), z[2], z[3], np.exp(z[4]), z[5], z[6])
        dk = d * 10.0 ** (g * xk)
        u = dr2 / dk
        l1p = np.log1p(u)
        ldt = np.log(dt + c)
        L1 = (Wsum * (np.log(K) + np.log(q - 1.0) - np.log(np.pi) - np.log(d))
              + LN10 * (a - g) * Wx - p * (w @ ldt) - q * (w @ l1p))
        r = u / (1.0 + u)
        gK, ga = Wsum / K, LN10 * Wx
        gc = -p * (w @ (1.0 / (dt + c)))
        gp = -(w @ ldt)
        gq = Wsum / (q - 1.0) - (w @ l1p)
        gd = -Wsum / d + (q / d) * (w @ r)
        gg = LN10 * (-Wx + q * ((w * xk) @ r))
        # compensator  C = sum_i rho_i * Omega_i(c, p)
        rho = K * 10.0 ** (a * self.xm)
        om, dom_dc, dom_dp = self._omori_terms(c, p)
        Cc = float(rho @ om)
        gCK, gCa = Cc / K, LN10 * float((self.xm * rho) @ om)
        gCc, gCp = float(rho @ dom_dc), float(rho @ dom_dp)
        f = -(L1 - Cc)
        gnat = -np.array([gK - gCK, gc - gCc, gp - gCp, ga - gCa, gd, gg, gq])
        gz = np.array([gnat[0] * K, gnat[1] * c, gnat[2], gnat[3], gnat[4] * d, gnat[5], gnat[6]])
        return f / Wsum, gz / Wsum

    def _mstep_triggering(self):
        ii, jj = np.where(self.W > MSTEP_PAIR_CUTOFF)
        if len(ii) == 0:
            log.warning("No significant triggering pairs; M-step skipped.")
            return {}
        w = self.W[ii, jj]
        data = (w, self._dt[ii, jj], self._dr2[ii, jj], self.xm[ii], float(w.sum()), float(w @ self.xm[ii]))
        res = minimize(self._fg, z_from_params(self.params), args=(data,), jac=True, method="L-BFGS-B",
                       bounds=BOUNDS_Z, options=dict(maxiter=self.optimizer_maxiter, ftol=1e-12, gtol=1e-8))
        self.params = params_from_z(res.x)
        return dict(opt_nit=int(res.nit), opt_nfev=int(res.nfev), opt_success=bool(res.success))

    # ── EM driver ──────────────────────────────────────────────────────────────

    def fit(self):
        log.info("Starting ETAS EM  (N=%d, background=%s%s, max_iter=%d)", self.N, self.background,
                 f" @ {self.bg_rate:.3e}/day/km^2" if self.background == "fixed" else "", self.max_iter)
        w_bg0 = np.full(self.N, 0.1)
        mu, n0 = self._background_rate(w_bg0)
        ll_prev, self.converged, t_start = None, False, time.time()
        for it in range(self.max_iter):
            tot = self._estep(mu)
            ll = float(np.log(tot).sum() - n0 - self._compensator(self.params))
            rel = np.inf if ll_prev is None else abs(ll - ll_prev) / (abs(ll_prev) + 1e-12)
            row = dict(iteration=it + 1, loglik=ll, rel_change=rel, bg_mass=float(self.w_bg.sum()),
                       seconds=round(time.time() - t_start, 1), **self.params)
            self.trace.append(row)
            log.info("  Iter %3d  loglik=%.3f  rel=%.1e  K=%.4g c=%.4g p=%.3f alpha=%.3f d=%.4g gamma=%.3f q=%.3f  bg_mass=%.1f",
                     it + 1, ll, rel, self.params["K"], self.params["c"], self.params["p"], self.params["alpha"],
                     self.params["d"], self.params["gamma"], self.params["q"], self.w_bg.sum())
            if ll_prev is not None and rel < self.tol:
                self.converged = True
                log.info("Converged at iteration %d.", it + 1)
                break
            ll_prev = ll
            if self.background == "kde":
                mu, n0 = self._background_rate(self.w_bg)
            row.update(self._mstep_triggering())
        if not self.converged:
            # make the saved probabilities consistent with the final parameters
            tot = self._estep(mu)
            ll = float(np.log(tot).sum() - n0 - self._compensator(self.params))
            self.trace.append(dict(iteration=len(self.trace) + 1, loglik=ll, rel_change=np.nan,
                                   bg_mass=float(self.w_bg.sum()), seconds=round(time.time() - t_start, 1),
                                   **self.params))
            log.warning("Reached max_iter=%d without meeting tol=%.0e; final E-step run with the final parameters.",
                        self.max_iter, self.tol)
        self.mu, self.n0 = mu, n0
        log.info("ETAS fit complete.  Final params: %s", {k: float(f"{v:.4g}") for k, v in self.params.items()})
        return self

    # ── outputs ────────────────────────────────────────────────────────────────

    def verify(self):
        """Probability checks (thesis 5.4) + consistency of the stored W with the final parameters."""
        W = self.W
        res = C.verify_probabilities(self.w_bg, W.sum(axis=0), float(W.min()),
                                     int(np.count_nonzero(W[~self._causal])),
                                     int(np.count_nonzero(~np.isfinite(W))))
        R = self._rate_matrix()
        W2 = R / (self.mu + R.sum(axis=0))[None, :]
        res["max_abs_diff_vs_final_params"] = float(np.abs(W2 - W).max())
        res["consistent_with_final_params"] = res["max_abs_diff_vs_final_params"] < 1e-9
        res["all_passed"] = bool(res["all_passed"] and res["consistent_with_final_params"])
        return res

    def pair_arrays(self, cutoff):
        ii, jj = np.where(self.W >= cutoff)
        return ii, jj, self.W[ii, jj]


def plot_trace(etas, out_png):
    tr = pd.DataFrame(etas.trace)
    fig, ax = plt.subplots(2, 4, figsize=(17, 6.5))
    ax = ax.ravel()
    ax[0].plot(tr.iteration, tr.loglik, "o-", color="#2979ff")
    ax[0].set(title="log-likelihood (true, incl. compensator)", xlabel="EM iteration")
    for a, name in zip(ax[1:], PARAM_NAMES):
        a.plot(tr.iteration, tr[name], "o-", color="#2979ff")
        a.set(title=name, xlabel="EM iteration")
        if name in ("K", "c", "d"):
            a.set_yscale("log")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def parse_cli_args():
    p = argparse.ArgumentParser(description="Build an ETAS seismic family-tree graph (v2).")
    p.add_argument("--preset", choices=C.SEQUENCE_PRESETS.keys(), help="case study: catalogue, Mc, prefix and imposed background rate")
    p.add_argument("--catalogue", help="extracted sequence CSV (overrides --preset)")
    p.add_argument("--mag-min", type=float, help="completeness magnitude (overrides --preset)")
    p.add_argument("--max-events", type=int, default=C.MAX_EVENTS)
    p.add_argument("--out-prefix", help="prefix for output files (default: preset name)")
    p.add_argument("--out-dir", default="results", help="output folder (created if missing; default: results)")
    p.add_argument("--background", choices=["fixed", "kde"], help="fixed = impose --bg-rate (default with a preset); kde = estimate")
    p.add_argument("--bg-rate", type=float, help="imposed background rate [events/day/km^2] (overrides the preset's)")
    p.add_argument("--etas-max-iter", type=int, default=ETAS_MAX_ITER)
    p.add_argument("--optimizer-maxiter", type=int, default=ETAS_OPTIMIZER_MAXITER)
    p.add_argument("--tol", type=float, default=ETAS_TOL)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    path, mag_min, prefix, bg, rate = C.resolve_config(args, default_model_bg="kde")
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, f"{prefix}_{name}")

    df = C.load_catalogue(path, mag_min, args.max_events)
    df["t_days"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds() / 86_400.0
    x, y = C.project_km(df["latitude"].values, df["longitude"].values)
    area = C.domain_area_km2(x, y)

    t0 = time.time()
    etas = ETAS(df["t_days"].values, x, y, df["mag"].values, mc=mag_min, background=bg, bg_rate=rate, area=area,
                max_iter=args.etas_max_iter, tol=args.tol, optimizer_maxiter=args.optimizer_maxiter).fit()
    fit_seconds = time.time() - t0

    ver = etas.verify()
    log.info("Probability checks: %s", ver)
    if not ver["all_passed"]:
        log.warning("SOME PROBABILITY CHECKS FAILED - do not use these results; see the run JSON.")

    N = etas.N
    out_strength = etas.W.sum(axis=1)                     # all children, original probabilities
    best = etas.W.argmax(axis=0).astype(np.int64)
    best_w = etas.W.max(axis=0)
    best[best_w <= 0] = -1
    ei, ej, ew = etas.pair_arrays(C.W_THRESHOLD)
    G = C.build_knowledge_graph(df, etas.w_bg, out_strength, best, best_w, ei, ej, ew)
    C.export_graph(G, df, out("kg_etas.graphml"), out("kg_etas_edges.csv"), out("kg_etas_nodes.csv"))
    pi_, pj_, pw_ = etas.pair_arrays(C.PAIR_EXPORT_CUTOFF)
    C.export_pairs(out("kg_etas_pairs.npz"), N, pi_, pj_, pw_, etas.w_bg)

    tr = pd.DataFrame(etas.trace)
    tr.to_csv(out("kg_etas_trace.csv"), index=False)
    final = dict(etas.params, mc=mag_min, n_events=N, background=bg, bg_rate=rate, area_km2=area,
                 final_loglik=float(tr.loglik.iloc[-1]), n_em_iterations=len(tr), converged=etas.converged,
                 bg_mass=float(etas.w_bg.sum()), roots_p_gt_0_5=int((etas.w_bg > 0.5).sum()))
    pd.DataFrame([final]).to_csv(out("kg_etas_params.csv"), index=False)
    C.visualise_subgraph(G, df, "ETAS", out("etas_subgraph.png"))
    plot_trace(etas, out("etas_convergence.png"))
    C.write_run_json(out("kg_etas_run.json"), dict(
        model="ETAS v2", args=vars(args), resolved=dict(catalogue=path, mag_min=mag_min, background=bg, bg_rate=rate,
                                                       area_km2=area, n_events=N),
        final_params=etas.params, converged=etas.converged, n_em_iterations=len(tr),
        fit_seconds=round(fit_seconds, 1), verification=ver, edge_threshold=C.W_THRESHOLD))

    print("\n" + "=" * 64)
    print("ETAS v2 — SUMMARY")
    print("=" * 64)
    print(f"Events             : {N:,}      background: {bg}" + (f" @ {rate:.3e} /day/km^2" if bg == "fixed" else ""))
    print(f"EM iterations      : {len(tr)}  (converged: {etas.converged})   fit time {fit_seconds / 60:.1f} min")
    for k in PARAM_NAMES:
        print(f"  {k:<6} = {etas.params[k]:.5g}")
    print(f"Expected background events (sum P(bg)) : {etas.w_bg.sum():.1f}   roots (P>0.5): {(etas.w_bg > 0.5).sum()}")
    print(f"Probability checks : {'ALL PASSED' if ver['all_passed'] else 'FAILED'}  {ver}")
    print(f"Graph              : {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges (P >= {C.W_THRESHOLD})")
    C.print_top_events(df, out_strength, etas.w_bg)
    print("=" * 64)
