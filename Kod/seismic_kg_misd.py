"""
seismic_kg_misd.py   (v2.1 -- nonparametric MISD)
=================================================
Builds a seismic "family tree" graph from a JMA earthquake sequence using Model-Independent
Stochastic Declustering (MISD; Marsan & Lengline 2008; Marsan & Lengline 2010, Appendix A).
Shares all data handling, graph construction, probability checks and exports with
seismic_kg_etas.py (kg_common.py), so the two models differ ONLY in the triggering kernel
(and optionally the background).

Estimator
  * Triggering kernel = free histograms (no Omori / Gutenberg-Richter / power-law form assumed),
    estimated by EM with an exposure correction for the finite catalogue end and study-box edges.
  * Default kernel "separable": lambda = K(m) g(t) f(r), each factor a free histogram; the productivity
    law K(m) is a smooth log-linear function fitted by Poisson regression on the parent events (the
    binned estimate is always computed and plotted as a check). The paper's full (m,t,r) histogram
    is available as --kernel full; on catalogues of a few thousand events it overfits unless the
    background is imposed (see test_misd_synthetic.py).
  * BACKGROUND: by default IMPOSED (--background fixed): the long-term rate measured in the years
    before the sequence (estimate_background_rate.py), identical to the rate given to ETAS, as in
    Marsan & Lengline (2010). --background kde / homogeneous estimate it by EM (sensitivity checks).
  * Distances: epicentral, km. Known limitations: no fault-plane distances for large events
    (paper, Appendix B), no short-term completeness correction (Appendix C), depth ignored.

Dependencies:  pip install pandas numpy scipy networkx matplotlib
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import kg_common as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("misd")

EARTH_R_KM = C.EARTH_R_KM

# ── MISD discretisation (Marsan & Lengline, Appendix A1) ────────────────────────
MISD_KERNEL          = "separable"
MISD_PRODUCTIVITY    = "loglinear"  # "loglinear" | "bins" (separable kernel only)
MISD_MAX_ITER        = 300
MISD_TOL             = 1e-7         # relative change of log-likelihood
T_BINS_PER_DECADE    = 4
R_BINS_PER_DECADE    = 4
T_MIN_DAYS           = 1e-3         # first log bin edge (lags below go in bin 0)  ~1.4 min
R_MIN_KM             = 0.1
MAG_BIN_WIDTH        = 1.0
MIN_PARENTS_PER_BIN  = 10
BACKGROUND           = "kde"        # model default when no preset / rate is given
BG_NN                = 15
BG_MIN_BANDWIDTH_KM  = 3.0


def parse_cli_args():
    p = argparse.ArgumentParser(description="Build a nonparametric-MISD seismic family-tree graph.")
    p.add_argument("--preset", choices=C.SEQUENCE_PRESETS.keys(), help="case study: catalogue, Mc, prefix and imposed background rate")
    p.add_argument("--catalogue", help="extracted sequence CSV (overrides --preset)")
    p.add_argument("--mag-min", type=float, help="completeness magnitude (overrides --preset)")
    p.add_argument("--max-events", type=int, default=C.MAX_EVENTS)
    p.add_argument("--out-prefix", help="prefix for output files (default: preset name)")
    p.add_argument("--out-dir", default="results", help="output folder (created if missing; default: results)")
    p.add_argument("--kernel", choices=["full", "separable"], default=MISD_KERNEL,
                   help="'separable' (default): K(m)g(t)f(r), free histograms. 'full': the paper's histogram per magnitude bin")
    p.add_argument("--productivity", choices=["loglinear", "bins"], default=MISD_PRODUCTIVITY,
                   help="separable kernel only: smooth log-linear law (default) or free per-bin values")
    p.add_argument("--background", choices=["kde", "homogeneous", "fixed"],
                   help="fixed = impose --bg-rate (default with a preset); kde/homogeneous = estimated by EM")
    p.add_argument("--bg-rate", type=float, help="imposed background rate [events/day/km^2] (overrides the preset's)")
    p.add_argument("--max-iter", type=int, default=MISD_MAX_ITER)
    p.add_argument("--t-bins-per-decade", type=int, default=T_BINS_PER_DECADE)
    p.add_argument("--r-bins-per-decade", type=int, default=R_BINS_PER_DECADE)
    p.add_argument("--mag-bin-width", type=float, default=MAG_BIN_WIDTH)
    p.add_argument("--min-parents", type=int, default=MIN_PARENTS_PER_BIN)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# NONPARAMETRIC MISD
# ═══════════════════════════════════════════════════════════════════════════════

class MISD:
    """
    Parameters
    ----------
    times : (N,) origin times in days, ascending
    lats, lons : (N,) degrees
    mags : (N,)
    mag_min : completeness magnitude (lowest magnitude-bin edge)
    """

    def __init__(self, times, lats, lons, mags, mag_min,
                 kernel=MISD_KERNEL, background=BACKGROUND, productivity=MISD_PRODUCTIVITY,
                 t_bins_per_decade=T_BINS_PER_DECADE, r_bins_per_decade=R_BINS_PER_DECADE,
                 t_min_days=T_MIN_DAYS, r_min_km=R_MIN_KM,
                 mag_bin_width=MAG_BIN_WIDTH, min_parents=MIN_PARENTS_PER_BIN,
                 bg_nn=BG_NN, bg_min_bw_km=BG_MIN_BANDWIDTH_KM, bg_rate=None,
                 max_iter=MISD_MAX_ITER, tol=MISD_TOL):

        self.times = np.asarray(times, dtype=float)
        self.lats = np.asarray(lats, dtype=float)
        self.lons = np.asarray(lons, dtype=float)
        self.mags = np.asarray(mags, dtype=float)
        self.N = len(self.times)
        self.mag_min = mag_min
        self.kernel_mode = kernel
        self.productivity = productivity
        self.background = background
        self.tpd, self.rpd = t_bins_per_decade, r_bins_per_decade
        self.t_min, self.r_min = t_min_days, r_min_km
        self.mag_bin_width, self.min_parents = mag_bin_width, min_parents
        self.bg_nn, self.bg_min_bw = bg_nn, bg_min_bw_km
        self.bg_rate = bg_rate
        if background == "fixed" and not bg_rate:
            raise ValueError("background='fixed' needs bg_rate (events/day/km^2)")
        self.max_iter, self.tol = max_iter, tol

        self._build_geometry()
        self._build_bins()
        self._build_pairs()
        self._build_exposure()
        if self.background == "kde":
            self._build_background_kernel()
        self.loglik_history = []

    # ── geometry ───────────────────────────────────────────────────────────────

    def _build_geometry(self):
        lat0 = np.radians(self.lats.mean())
        self.x = EARTH_R_KM * np.radians(self.lons) * np.cos(lat0)
        self.y = EARTH_R_KM * np.radians(self.lats)
        self.T_end = self.times.max()
        self.T_span = max(self.T_end - self.times.min(), 1e-6)
        self.xmin, self.xmax = self.x.min(), self.x.max()
        self.ymin, self.ymax = self.y.min(), self.y.max()
        self.area = max((self.xmax - self.xmin) * (self.ymax - self.ymin), 1.0)

    # ── discretisation ─────────────────────────────────────────────────────────

    @staticmethod
    def _log_edges(lo, hi, per_decade):
        n = max(int(np.ceil(np.log10(hi / lo) * per_decade)), 1)
        return np.concatenate([[0.0], lo * 10.0 ** (np.arange(n + 1) / per_decade)])

    @staticmethod
    def _bin_index(v, lo, per_decade, nbins):
        idx = np.zeros(v.shape, dtype=np.int32)
        pos = v >= lo
        idx[pos] = 1 + np.floor(np.log10(v[pos] / lo) * per_decade).astype(np.int32)
        return np.minimum(idx, nbins - 1)

    def _build_bins(self):
        r_max = np.hypot(self.xmax - self.xmin, self.ymax - self.ymin) * 1.0001
        self.t_edges = self._log_edges(self.t_min, max(self.T_span, 10 * self.t_min), self.tpd)
        self.r_edges = self._log_edges(self.r_min, max(r_max, 10 * self.r_min), self.rpd)
        self.nT, self.nR = len(self.t_edges) - 1, len(self.r_edges) - 1
        self.t_w = np.diff(self.t_edges)                                    # day
        self.r_area = np.pi * (self.r_edges[1:] ** 2 - self.r_edges[:-1] ** 2)   # km^2
        self.t_mid = 0.5 * (self.t_edges[:-1] + self.t_edges[1:])
        self.r_mid = 0.5 * (self.r_edges[:-1] + self.r_edges[1:])

        # parent magnitude bins: fixed width, sparse top bins merged downwards
        w = self.mag_bin_width
        fine = np.arange(self.mag_min, self.mags.max() + w, w)
        counts, _ = np.histogram(self.mags, bins=fine)
        groups, hi, acc = [], len(counts), 0
        for k in range(len(counts) - 1, -1, -1):
            acc += counts[k]
            if acc >= self.min_parents or k == 0:
                groups.append((k, hi, acc))
                hi, acc = k, 0
        if len(groups) > 1 and groups[-1][2] < self.min_parents:     # sparse bottom stub
            lo_k, _, c_lo = groups.pop()
            k2, hi2, c2 = groups[-1]
            groups[-1] = (lo_k, hi2, c2 + c_lo)
        groups = groups[::-1]
        self.m_edges = np.array([fine[g[0]] for g in groups] + [fine[groups[-1][1]]])
        self.m_edges[-1] = max(self.m_edges[-1], self.mags.max() + 1e-9)
        self.nM = len(groups)
        self.mbin = np.clip(np.searchsorted(self.m_edges, self.mags, side="right") - 1, 0, self.nM - 1)
        self.n_par = np.bincount(self.mbin, minlength=self.nM)
        self.m_mean = np.array([self.mags[self.mbin == i].mean() if self.n_par[i] else np.nan
                                for i in range(self.nM)])
        log.info("Bins: %d magnitude %s | %d time-lag | %d distance",
                 self.nM, np.round(self.m_edges, 2).tolist(), self.nT, self.nR)

    def _build_pairs(self):
        N = self.N
        iu, ju = np.triu_indices(N, k=1)
        self.parent = iu.astype(np.int32)
        self.child = ju.astype(np.int32)
        del iu, ju
        dt = self.times[self.child] - self.times[self.parent]
        tb = self._bin_index(dt, self.t_min, self.tpd, self.nT)
        del dt
        dr = np.hypot(self.x[self.child] - self.x[self.parent],
                      self.y[self.child] - self.y[self.parent])
        rb = self._bin_index(dr, self.r_min, self.rpd, self.nR)
        del dr
        self.tb, self.rb = tb.astype(np.int16), rb.astype(np.int16)
        self.cell = ((self.mbin[self.parent].astype(np.int32) * self.nT + tb) * self.nR + rb).astype(np.int32)
        self.P = len(self.parent)

    def _build_exposure(self):
        """Observed (time x area) exposure of every parent in every lag bin / shell."""
        life = self.T_end - self.times
        self.tau = np.clip(life[:, None] - self.t_edges[None, :-1], 0.0, self.t_w[None, :])   # (N,nT)

        n_ang = 72
        th = 2 * np.pi * (np.arange(n_ang) + 0.5) / n_ang
        rho = np.sqrt(0.5 * (self.r_edges[:-1] ** 2 + self.r_edges[1:] ** 2))   # area-mean radius
        px = self.x[:, None, None] + rho[None, :, None] * np.cos(th)[None, None, :]
        py = self.y[:, None, None] + rho[None, :, None] * np.sin(th)[None, None, :]
        inside = ((px >= self.xmin) & (px <= self.xmax) & (py >= self.ymin) & (py <= self.ymax))
        frac = np.maximum(inside.mean(axis=2), 0.5 / n_ang)                       # (N,nR)
        self.aobs = frac * self.r_area[None, :]

        self.E = np.zeros((self.nM, self.nT, self.nR))
        for i in range(self.nM):
            sel = self.mbin == i
            if sel.any():
                self.E[i] = self.tau[sel].T @ self.aobs[sel]
        self.E_flat = self.E.ravel()

    # ── background ─────────────────────────────────────────────────────────────

    def _build_background_kernel(self):
        """kern[i,j] = Gaussian kernel (1/km^2) centred at event i, evaluated at j,
        with event i's adaptive bandwidth (distance to its k-th neighbour)."""
        N = self.N
        xy = np.column_stack([self.x, self.y])
        k = min(self.bg_nn, N - 1)
        kern = np.empty((N, N), dtype=np.float32)
        h = np.empty(N)
        B = 1000
        for s in range(0, N, B):
            D = cdist(xy[s:s + B], xy)
            hb = np.maximum(np.partition(D, k, axis=1)[:, k], self.bg_min_bw)
            h[s:s + B] = hb
            kern[s:s + B] = (np.exp(-0.5 * (D / hb[:, None]) ** 2) /
                             (2 * np.pi * hb[:, None] ** 2)).astype(np.float32)
        self._kern, self.bg_bandwidth = kern, h

    def _background_rate(self, w_bg):
        """mu_j in events / day / km^2, and the total background mass n0 = sum w_bg."""
        n0 = float(w_bg.sum())
        if self.background == "fixed":
            return np.full(self.N, self.bg_rate), self.bg_rate * self.T_span * self.area
        if self.background == "kde":
            mu = (w_bg.astype(np.float32) @ self._kern).astype(float) / self.T_span
        else:
            mu = np.full(self.N, n0 / (self.T_span * self.area))
        return np.maximum(mu, 1e-15), n0

    # ── kernel (de)composition ─────────────────────────────────────────────────
    # "full":      lam_flat[(i,j,k)] = free histogram per magnitude bin (the paper's)
    # "separable": lambda = Kev[parent] * g[tb] * f[rb] with g, f free histograms
    #              (sum g*dt = 1, sum f*dA = 1) and Kev the productivity of each
    #              parent: per magnitude bin ("bins") or exp(a + b*(m - mmin))
    #              ("loglinear", estimated by Poisson regression on the parents).

    def _pair_rates(self):
        if self.kernel_mode == "full":
            return self.lam_flat[self.cell]
        return self._Kev[self.parent] * self._g[self.tb] * self._f[self.rb]

    def _kernel_integral(self):
        if self.kernel_mode == "full":
            return float((self.lam_flat * self.E_flat).sum())
        return float((self._Kev * (self.tau @ self._g) * (self.aobs @ self._f)).sum())

    def _initial_kernel(self):
        """Smooth Omori-like / power-law / G-R-like start (the EM then moves freely)."""
        g = 1.0 / (self.t_mid + 0.01);  g /= (g * self.t_w).sum()
        f = (self.r_mid ** 2 + 1.0) ** -1.5;  f /= (f * self.r_area).sum()
        x = self.mags - self.mag_min
        if self.kernel_mode == "full":
            K = 10.0 ** (0.8 * (np.nan_to_num(self.m_mean, nan=self.mag_min) - self.mag_min))
            lam = K[:, None, None] * g[None, :, None] * f[None, None, :]
            lam *= 0.5 * self.N / max((lam * self.E).sum(), 1e-300)     # initial branching ~0.5
            self.lam_flat = lam.ravel()
            return
        self._g, self._f = g, f
        self._Kev = 10.0 ** (0.8 * x)
        self._Kev *= 0.5 * self.N / max(self._kernel_integral(), 1e-300)
        self._pa = float(np.log(self._Kev[0]))
        self._pb = 0.8 * np.log(10.0)

    def _update_productivity(self, S, e):
        """S_i: expected number of observed direct offspring of event i;
        e_i: its observed (time x area) exposure under the current g, f."""
        if self.productivity == "bins":
            num = np.bincount(self.mbin, weights=S, minlength=self.nM)
            den = np.bincount(self.mbin, weights=e, minlength=self.nM)
            Kc = np.divide(num, den, out=np.zeros(self.nM), where=den > 0)
            self._Kev = Kc[self.mbin]
            return
        # Poisson regression  S_i ~ Poisson( exp(a + b x_i) * e_i ),  x_i = m_i - mmin
        x = self.mags - self.mag_min
        a, b = self._pa, self._pb
        for _ in range(50):
            mu_i = np.exp(np.clip(a + b * x, -700, 700)) * e
            ga, gb = (S - mu_i).sum(), ((S - mu_i) * x).sum()
            Haa, Hab, Hbb = mu_i.sum(), (mu_i * x).sum(), (mu_i * x * x).sum()
            det = Haa * Hbb - Hab ** 2
            if det <= 1e-300:
                break
            da, db = (Hbb * ga - Hab * gb) / det, (-Hab * ga + Haa * gb) / det
            big = max(abs(da), abs(db))
            if big > 2.0:
                da, db = da * 2.0 / big, db * 2.0 / big
            a, b = a + da, b + db
            if big < 1e-9:
                break
        b = float(np.clip(b, 0.0, 3.0 * np.log(10.0)))          # alpha_log10 in [0, 3]
        denom = (np.exp(b * x) * e).sum()
        a = float(np.log(max(S.sum(), 1e-300) / max(denom, 1e-300)))
        self._pa, self._pb = a, b
        self._Kev = np.exp(a + b * x)

    def _mstep(self, omega):
        if self.kernel_mode == "full":
            n = np.bincount(self.cell, weights=omega, minlength=self.E_flat.size)
            lam = np.zeros_like(n)
            np.divide(n, self.E_flat, out=lam, where=self.E_flat > 0)
            self.lam_flat = lam
            return
        # separable: exact conditional maximisers of the expected complete log-likelihood
        S = np.bincount(self.parent, weights=omega, minlength=self.N)
        Sg = np.bincount(self.tb, weights=omega, minlength=self.nT)
        Sf = np.bincount(self.rb, weights=omega, minlength=self.nR)
        G = self.tau @ self._g
        F = self.aobs @ self._f
        self._update_productivity(S, G * F)
        den = self.tau.T @ (self._Kev * F)
        g = np.divide(Sg, den, out=np.zeros(self.nT), where=den > 0)
        s = (g * self.t_w).sum();  self._g = g / s;  self._Kev = self._Kev * s;  self._pa += np.log(s)
        G = self.tau @ self._g
        den = self.aobs.T @ (self._Kev * G)
        f = np.divide(Sf, den, out=np.zeros(self.nR), where=den > 0)
        s = (f * self.r_area).sum();  self._f = f / s;  self._Kev = self._Kev * s;  self._pa += np.log(s)

    # ── EM ─────────────────────────────────────────────────────────────────────

    def _estep(self, mu):
        phi = self._pair_rates()
        trig = np.bincount(self.child, weights=phi, minlength=self.N)
        lam_tot = mu + trig
        return phi / lam_tot[self.child], mu / lam_tot, lam_tot

    def fit(self):
        log.info("Starting MISD EM  (N=%d, pairs=%d, kernel=%s%s, background=%s)",
                 self.N, self.P, self.kernel_mode,
                 f"/{self.productivity}" if self.kernel_mode == "separable" else "", self.background)
        self._initial_kernel()
        w_bg = np.full(self.N, 0.1)
        mu, n0 = self._background_rate(w_bg)
        ll_prev, converged = None, False

        for it in range(self.max_iter):
            omega, w_bg, lam_tot = self._estep(mu)
            ll = float(np.log(lam_tot).sum() - n0 - self._kernel_integral())
            self.loglik_history.append(ll)
            if it % 10 == 0 or it < 3:
                log.info("  Iter %3d  loglik=%.3f  bg_mass=%.1f  triggered_fraction=%.3f",
                         it + 1, ll, w_bg.sum(), 1.0 - w_bg.mean())
            if ll_prev is not None and abs(ll - ll_prev) <= self.tol * abs(ll_prev):
                converged = True
                log.info("Converged at iteration %d (loglik=%.3f).", it + 1, ll)
                break
            ll_prev = ll
            self._mstep(omega)
            mu, n0 = self._background_rate(w_bg)

        if not converged:
            omega, w_bg, lam_tot = self._estep(mu)
            log.warning("Reached max_iter=%d without meeting tol=%.0e (last loglik=%.3f).",
                        self.max_iter, self.tol, self.loglik_history[-1])
        self.omega, self.w_bg, self.mu = omega, w_bg, mu
        self.converged, self.n_iter = converged, len(self.loglik_history)
        self._summarise_kernel()
        return self

    def edges(self, threshold):
        m = self.omega >= threshold
        return self.parent[m], self.child[m], self.omega[m]

    # ── kernel read-out (the point of a nonparametric method) ─────────────────

    def _summarise_kernel(self):
        # pair mass actually observed in each bin (how much data supports it)
        self.mass_t = np.bincount(self.tb, weights=self.omega, minlength=self.nT)
        self.mass_r = np.bincount(self.rb, weights=self.omega, minlength=self.nR)

        if self.kernel_mode == "full":
            lam = self.lam_flat.reshape(self.nM, self.nT, self.nR)
            self.K_mag = np.einsum("ijk,j,k->i", lam, self.t_w, self.r_area)
            gt = np.einsum("ijk,i,k->j", lam, self.n_par, self.r_area)
            fr = np.einsum("ijk,i,j->k", lam, self.n_par, self.t_w)
            self.g_t = gt / max((gt * self.t_w).sum(), 1e-300)
            self.f_r = fr / max((fr * self.r_area).sum(), 1e-300)
            self.K_mag_fit = None
            self.K_event = self.K_mag[self.mbin]
            self.alpha_fit = None
        else:
            # binned maximum-likelihood productivity given the final g, f (a check on the law)
            S = np.bincount(self.parent, weights=self.omega, minlength=self.N)
            e = (self.tau @ self._g) * (self.aobs @ self._f)
            num = np.bincount(self.mbin, weights=S, minlength=self.nM)
            den = np.bincount(self.mbin, weights=e, minlength=self.nM)
            self.K_mag = np.divide(num, den, out=np.zeros(self.nM), where=den > 0)
            self.g_t, self.f_r = self._g.copy(), self._f.copy()
            self.K_event = self._Kev.copy()
            self.K_mag_fit = np.array([self._Kev[self.mbin == i].mean() if self.n_par[i] else np.nan
                                       for i in range(self.nM)])
            self.alpha_fit = (self._pb / np.log(10.0)) if self.productivity == "loglinear" else None

        self.alpha_bins = self._slope(self.m_mean, np.log10(np.maximum(self.K_mag, 1e-300)),
                                      self.n_par.astype(float), mask=self.K_mag > 0)
        self.alpha_hat = self.alpha_fit if self.alpha_fit is not None else self.alpha_bins
        t_ok = (self.t_mid > 0.05) & (self.t_mid < self.T_span / 5) & (self.mass_t >= 3) & (self.g_t > 0)
        self.p_hat = -self._slope(np.log10(self.t_mid), np.log10(np.maximum(self.g_t, 1e-300)),
                                  self.mass_t, mask=t_ok)
        r_ok = (self.r_mid > 1.0) & (self.r_mid < 50.0) & (self.mass_r >= 3) & (self.f_r > 0)
        self.s_hat = -self._slope(np.log10(self.r_mid), np.log10(np.maximum(self.f_r, 1e-300)),
                                  self.mass_r, mask=r_ok)
        self.triggered_fraction = float(1.0 - self.w_bg.mean())
        # expected direct offspring per event (whole plane, all lags supported by the data)
        self.branching = float(self.K_event.mean())
        log.info("Kernel read-out: productivity alpha(log10)=%s (binned check %s) | Omori p=%s | "
                 "spatial decay f~r^-s, s=%s (ETAS-equivalent q=s/2) | triggered fraction=%.3f | "
                 "branching ratio n=%.3f (indicative only: biased high in synthetic tests)",
                 self._fmt(self.alpha_hat), self._fmt(self.alpha_bins), self._fmt(self.p_hat),
                 self._fmt(self.s_hat), self.triggered_fraction, self.branching)

    @staticmethod
    def _slope(x, y, w, mask):
        m = mask & np.isfinite(x) & np.isfinite(y) & (w > 0)
        if m.sum() < 2:
            return np.nan
        return float(np.polyfit(x[m], y[m], 1, w=np.sqrt(w[m]))[0])

    @staticmethod
    def _fmt(v):
        return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"


def export_kernels(misd, out_csv):
    """Fitted nonparametric kernel in long format (for plots / comparison with ETAS)."""
    t = pd.DataFrame({"kind": "time", "lo": misd.t_edges[:-1], "hi": misd.t_edges[1:],
                      "mid": misd.t_mid, "value": misd.g_t, "pair_mass": misd.mass_t})
    r = pd.DataFrame({"kind": "distance", "lo": misd.r_edges[:-1], "hi": misd.r_edges[1:],
                      "mid": misd.r_mid, "value": misd.f_r, "pair_mass": misd.mass_r})
    m = pd.DataFrame({"kind": "magnitude", "lo": misd.m_edges[:-1], "hi": misd.m_edges[1:],
                      "mid": misd.m_mean, "value": misd.K_mag, "pair_mass": misd.n_par})
    parts = [t, r, m]
    if misd.K_mag_fit is not None:
        parts.append(pd.DataFrame({"kind": "magnitude_fit", "lo": misd.m_edges[:-1], "hi": misd.m_edges[1:],
                                   "mid": misd.m_mean, "value": misd.K_mag_fit, "pair_mass": misd.n_par}))
    summ = {"summary:alpha_log10": misd.alpha_hat, "summary:alpha_log10_binned_check": misd.alpha_bins,
            "summary:p": misd.p_hat, "summary:s": misd.s_hat,
            "summary:triggered_fraction": misd.triggered_fraction,
            "summary:branching_ratio_indicative": misd.branching,
            "summary:n_iter": float(misd.n_iter), "summary:converged": float(misd.converged)}
    parts.append(pd.DataFrame({"kind": list(summ), "lo": np.nan, "hi": np.nan, "mid": np.nan,
                               "value": list(summ.values()), "pair_mass": np.nan}))
    pd.concat(parts).to_csv(out_csv, index=False)
    log.info("Kernel table saved -> %s", out_csv)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISE KERNELS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_kernels(misd: MISD, out_png):
    """The fitted nonparametric kernel: productivity vs magnitude, Omori-type
    decay, spatial decay. Marker size/opacity reflects how much data supports a bin."""
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    ok = misd.K_mag > 0
    ax[0].semilogy(misd.m_mean[ok], misd.K_mag[ok], "o", ms=8, color="#2979ff", label="binned estimate (check)")
    if misd.K_event is not None and misd.kernel_mode == "separable":
        o = np.argsort(misd.mags)
        ax[0].semilogy(misd.mags[o], misd.K_event[o], "-", color="#ff6d00", label="fitted law")
    ax[0].legend(fontsize=8)
    ax[0].set(xlabel="parent magnitude", ylabel="direct offspring per event",
              title=f"Productivity   (alpha_log10 ≈ {MISD._fmt(misd.alpha_hat)})")

    for a, x, y, mass, xl, yl, ttl in [
        (ax[1], misd.t_mid, misd.g_t, misd.mass_t, "lag t  [day]", "density  [1/day]",
         f"Temporal decay   (p ≈ {MISD._fmt(misd.p_hat)})"),
        (ax[2], misd.r_mid, misd.f_r, misd.mass_r, "distance r  [km]", "density  [1/km²]",
         f"Spatial decay   (f ~ r^-s, s ≈ {MISD._fmt(misd.s_hat)})")]:
        good = (y > 0) & (mass >= 1.0)      # hide bins with < 1 unit of pair mass (no data support)
        a.loglog(x[good], y[good], "-", color="#999999", lw=1)
        sc = a.scatter(x[good], y[good], c=np.log10(mass[good] + 1), cmap="viridis", s=36, zorder=3)
        a.set(xlabel=xl, ylabel=yl, title=ttl)
        a.grid(alpha=0.3, which="both")
        plt.colorbar(sc, ax=a, label="log10(pair mass + 1)")
    ax[0].grid(alpha=0.3)

    ax[3].plot(range(1, len(misd.loglik_history) + 1), misd.loglik_history, "o-", ms=3, color="#2979ff")
    ax[3].set(xlabel="EM iteration", ylabel="log-likelihood", title="MISD convergence")
    ax[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    log.info("Kernel diagnostics saved -> %s", out_png)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_cli_args()
    path, mag_min, prefix, bg, rate = C.resolve_config(args, default_model_bg=BACKGROUND)
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda name: os.path.join(args.out_dir, f"{prefix}_{name}")

    df = C.load_catalogue(path, mag_min, args.max_events)
    df["t_days"] = (df["datetime"] - df["datetime"].min()).dt.total_seconds() / 86_400.0

    t0 = time.time()
    misd = MISD(df["t_days"].values, df["latitude"].values, df["longitude"].values, df["mag"].values,
                mag_min=mag_min, kernel=args.kernel, productivity=args.productivity, background=bg, bg_rate=rate,
                t_bins_per_decade=args.t_bins_per_decade, r_bins_per_decade=args.r_bins_per_decade,
                mag_bin_width=args.mag_bin_width, min_parents=args.min_parents, max_iter=args.max_iter).fit()
    fit_seconds = time.time() - t0

    N = misd.N
    ver = C.verify_probabilities(misd.w_bg, np.bincount(misd.child, weights=misd.omega, minlength=N),
                                 float(misd.omega.min()), int((misd.parent >= misd.child).sum()),
                                 int(np.count_nonzero(~np.isfinite(misd.omega))))
    log.info("Probability checks: %s", ver)
    if not ver["all_passed"]:
        log.warning("SOME PROBABILITY CHECKS FAILED - do not use these results; see the run JSON.")

    out_strength = np.bincount(misd.parent, weights=misd.omega, minlength=N)   # all children, original probabilities
    best, best_w = C.best_parent_from_pairs(N, misd.parent, misd.child, misd.omega)
    ei, ej, ew = misd.edges(C.W_THRESHOLD)
    G = C.build_knowledge_graph(df, misd.w_bg, out_strength, best, best_w, ei, ej, ew)
    C.export_graph(G, df, out("kg_misd.graphml"), out("kg_misd_edges.csv"), out("kg_misd_nodes.csv"))
    C.export_pairs(out("kg_misd_pairs.npz"), N, misd.parent, misd.child, misd.omega, misd.w_bg)
    export_kernels(misd, out("kg_misd_kernels.csv"))
    pd.DataFrame({"iteration": np.arange(1, len(misd.loglik_history) + 1), "loglik": misd.loglik_history}
                 ).to_csv(out("kg_misd_trace.csv"), index=False)
    C.visualise_subgraph(G, df, "MISD", out("misd_subgraph.png"))
    plot_kernels(misd, out("misd_kernels.png"))
    C.write_run_json(out("kg_misd_run.json"), dict(
        model="MISD v2.1", args=vars(args),
        resolved=dict(catalogue=path, mag_min=mag_min, background=bg, bg_rate=rate, n_events=N,
                      kernel=misd.kernel_mode, productivity=misd.productivity, area_km2=misd.area),
        kernel_readout=dict(alpha_log10=misd.alpha_hat, alpha_log10_binned_check=misd.alpha_bins, p=misd.p_hat,
                            s=misd.s_hat, triggered_fraction=misd.triggered_fraction),
        converged=misd.converged, n_em_iterations=misd.n_iter, fit_seconds=round(fit_seconds, 1),
        verification=ver, edge_threshold=C.W_THRESHOLD, roots_p_gt_0_5=int((misd.w_bg > 0.5).sum())))

    print("\n" + "=" * 64)
    print("MISD (nonparametric) — SUMMARY")
    print("=" * 64)
    print(f"Events / pairs     : {N:,} / {misd.P:,}")
    print(f"Kernel / background: {misd.kernel_mode}{'/' + misd.productivity if misd.kernel_mode == 'separable' else ''} / "
          f"{bg}" + (f" @ {rate:.3e} /day/km^2" if bg == "fixed" else ""))
    print(f"EM iterations      : {misd.n_iter}  (converged: {misd.converged})   fit time {fit_seconds:.0f} s")
    print(f"Productivity alpha : {MISD._fmt(misd.alpha_hat)}  (log10 units; binned check {MISD._fmt(misd.alpha_bins)})")
    print(f"Omori p (fitted)   : {MISD._fmt(misd.p_hat)}")
    print(f"Spatial s / q=s/2  : {MISD._fmt(misd.s_hat)} / {MISD._fmt(misd.s_hat / 2 if np.isfinite(misd.s_hat) else np.nan)}")
    print(f"Expected background events (sum P(bg)) : {misd.w_bg.sum():.1f}   roots (P>0.5): {(misd.w_bg > 0.5).sum()}")
    print(f"Probability checks : {'ALL PASSED' if ver['all_passed'] else 'FAILED'}  {ver}")
    print(f"Graph              : {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges (P >= {C.W_THRESHOLD})")
    C.print_top_events(df, out_strength, misd.w_bg)
    print("=" * 64)
