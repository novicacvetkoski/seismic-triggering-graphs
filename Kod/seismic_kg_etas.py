"""
seismic_kg_etas.py
===================
Builds a seismic "family tree" Knowledge Graph (KG) from the merged Japan
earthquake catalogue using the space-time ETAS model (Epidemic-Type
Aftershock Sequence; Ogata, 1988, 1998) fitted by the stochastic
declustering EM algorithm of Zhuang, Ogata & Vere-Jones (2002, 2004).

This is the parametric counterpart to seismic_kg_misd.py. Where MISD
estimates the triggering density g(Δt, Δr) *nonparametrically* (as a
histogram / kernel, re-estimated at every EM iteration with no fixed
functional form), ETAS assumes a specific parametric shape for the
triggering kernel from the outset:

    Temporal :  Omori-Utsu law           ~ (t - t_i + c)^(-p)
    Magnitude:  Gutenberg-Richter scaling ~ 10^(alpha * (m_i - m_cut))
    Spatial  :  long-tailed isotropic     ~ (1 + r^2 / d(m_i))^(-q)

and estimates the small parameter vector (mu, K, c, p, alpha, d, gamma, q)
by maximum likelihood instead of estimating a full N x N nonparametric
kernel. The two models are complementary: MISD makes no assumption about
the triggering kernel's shape (useful for checking whether Omori/GR/ETAS
assumptions are reasonable for the catalogue), while ETAS gives a compact,
interpretable, well-studied parametrisation that is directly comparable to
the seismology literature (and to JMA-catalogue ETAS studies).

Everything downstream of the triggering-probability matrix w[i,j] --
knowledge-graph construction, direct/indirect edge labelling, GraphML /
CSV export, and ego-network visualisation -- is intentionally kept
IDENTICAL to seismic_kg_misd.py so that the two graphs are directly
comparable node-for-node and edge-for-edge.

Pipeline
--------
  1. Load & filter the merged catalogue               (identical to MISD)
  2. Fit ETAS via stochastic-declustering EM:
       E-step : w[i,j] = P(j triggered by i),  w_bg[j] = P(j background)
       M-step : (a) re-estimate spatially-variable background mu(x,y) via
                    adaptive kernel smoothing of background-weighted events
                (b) re-estimate ETAS params (K,c,p,alpha,d,gamma,q) by
                    weighted MLE (numerical optimisation) on the current
                    E-step weights
  3. Build a directed NetworkX graph from the converged w[i,j] matrix
  4. Identify direct vs. indirect (cascade) edges
  5. Export the graph (GraphML + edge-list CSV) for downstream use
  6. Print a summary and plot a small subgraph + parameter estimates

Dependencies
------------
  pip install pandas numpy scipy networkx matplotlib tqdm

References
----------
  Ogata, Y. (1988). Statistical models for earthquake occurrences and
    residual analysis for point processes. JASA, 83(401), 9-27.
  Ogata, Y. (1998). Space-time point-process models for earthquake
    occurrences. Ann. Inst. Statist. Math., 50(2), 379-402.
  Zhuang, J., Ogata, Y., & Vere-Jones, D. (2002). Stochastic declustering
    of space-time earthquake occurrences. JASA, 97(458), 369-380.
  Zhuang, J. (2011). Next-day earthquake forecasts by using the ETAS model.
    Earth, Planets and Space, 63, 207-216. (tutorial on the EM recipe used
    here for the M-step split into temporal / spatial / background parts)
"""

import argparse
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  (edit these to taste — kept aligned with seismic_kg_misd.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-sequence presets, tuned so each sequence yields a comparable-sized graph
# despite very different regional completeness magnitudes. Kept identical to
# the presets in seismic_kg_misd.py so the two graphs are directly comparable.
SEQUENCE_PRESETS = {
    "tohoku":   dict(catalogue="jma_tohoku.csv",   mag_min=4.0, out_prefix="tohoku"),
    "kumamoto": dict(catalogue="jma_kumamoto.csv", mag_min=2.0, out_prefix="kumamoto"),
    "noto":     dict(catalogue="jma_noto.csv",     mag_min=2.0, out_prefix="noto"),
}

CATALOGUE_PATH   = "merged_japan_earthquakes.csv"
OUT_GRAPHML      = "seismic_kg_etas.graphml"
OUT_EDGE_CSV     = "seismic_kg_etas_edges.csv"
OUT_NODE_CSV     = "seismic_kg_etas_nodes.csv"
OUT_PARAMS_CSV   = "seismic_kg_etas_params.csv"

# -- Catalogue filter (identical defaults to the MISD script) ------------------
MAG_MIN          = 4.5          # only keep M >= this (magnitude of completeness proxy)
DATE_START       = "1900-01-01"
DATE_END         = "2030-12-31"
MAX_EVENTS       = 8000         # safety cap; dense NxN matrix at 8000 events ~ 512MB

# -- ETAS EM hyper-parameters ---------------------------------------------------
ETAS_MAX_ITER    = 30           # outer EM iterations
ETAS_TOL         = 1e-4         # convergence threshold (relative change in log-lik)
BG_BANDWIDTH_NN  = 15           # adaptive background KDE: bandwidth = dist to k-th NN
ETAS_OPTIMIZER_MAXITER = 20     # L-BFGS-B iterations per M-step. The M-step likelihood
                                 # is evaluated over up to O(N^2) pairs and scipy's
                                 # finite-difference gradient needs ~8 evals/iteration
                                 # (7 free params), so this bounds worst-case per-M-step
                                 # cost. Raise this (e.g. 50-100) for a more precise fit
                                 # once you're running outside an interactive session
                                 # with no wall-clock constraint.

# -- ETAS parameter initial values ----------------------------------------------
# lambda(t,x,y) = mu(x,y)
#   + sum_{i: t_i<t} K * 10^(alpha*(m_i - mc)) * (t-t_i+c)^(-p)
#                       * (q-1)/(pi*d_i) * (1 + r_ij^2/d_i)^(-q)
#   with d_i = d * 10^(gamma*(m_i - mc))
INIT_PARAMS = dict(
    K      = 0.05,     # productivity scale
    c      = 0.01,     # Omori time offset [days]
    p      = 1.1,      # Omori decay exponent
    alpha  = 1.0,      # magnitude-productivity exponent
    d      = 0.02,      # spatial scale at m = mc [deg^2]
    gamma  = 0.5,       # magnitude scaling of spatial extent
    q      = 1.5,       # spatial decay exponent (q>1 required)
)
MC = MAG_MIN            # magnitude cutoff used inside the productivity/spatial terms

# -- Graph edge threshold (identical semantics to the MISD script) -------------
W_THRESHOLD      = 0.05

# -- Visualisation ---------------------------------------------------------------
VIZ_SEED_NODE    = None
VIZ_HOP_RADIUS   = 2
VIZ_MIN_MAG      = 5.5
VIZ_MAX_NODES    = 150          # cap ego-network size for readability


def parse_cli_args():
    p = argparse.ArgumentParser(description="Build an ETAS seismic family-tree graph.")
    p.add_argument("--preset", choices=SEQUENCE_PRESETS.keys(),
                    help="Use a built-in (catalogue, mag_min, out_prefix) combo for one of the case studies")
    p.add_argument("--catalogue", help="Path to the extracted sequence CSV (overrides --preset)")
    p.add_argument("--mag-min", type=float, help="Completeness magnitude cutoff (overrides --preset)")
    p.add_argument("--max-events", type=int, default=MAX_EVENTS)
    p.add_argument("--out-prefix", help="Prefix for output files, e.g. 'kumamoto' -> kumamoto_kg.graphml")
    p.add_argument("--etas-max-iter", type=int, default=ETAS_MAX_ITER,
                    help="Outer EM iterations (default: %(default)s). This is the single "
                         "biggest runtime lever -- each iteration includes an O(N^2) "
                         "numerical MLE M-step.")
    p.add_argument("--optimizer-maxiter", type=int, default=ETAS_OPTIMIZER_MAXITER,
                    help="L-BFGS-B iterations per M-step (default: %(default)s). Raise "
                         "for a more precise fit once running with no time constraint "
                         "(e.g. 50-100); lower for a fast smoke test (e.g. 3-5).")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & FILTER CATALOGUE   (identical logic to seismic_kg_misd.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogue(path: str, mag_min: float = MAG_MIN,
                    max_events: int = MAX_EVENTS) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df["datetime"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude", "depth", "mag"])

    # -- Magnitude determination method ---------------------------------------
    # JMA extracts carry a magType column: 'V' (velocity-type) vs 'D'
    # (displacement-type) are BOTH native JMA-scale (Mj) magnitudes, not
    # different scales -- JMA switches determination method as magnitude
    # grows (velocity-type saturates for large ruptures), so V and D are
    # strongly correlated with event size. Filtering to 'V' only silently
    # discards most of the largest events in a sequence -- do NOT do that.
    # Only genuinely missing/unparseable magType rows are dropped here.
    if "magType" in df.columns:
        n_before = len(df)
        df = df.dropna(subset=["magType"])
        n_dropped = n_before - len(df)
        if n_dropped:
            log.info("Dropped %d events with missing magType.", n_dropped)

    df = df[df["mag"] >= mag_min]
    df = df[df["datetime"] >= pd.Timestamp(DATE_START, tz="UTC")]
    df = df[df["datetime"] <= pd.Timestamp(DATE_END,   tz="UTC")]
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) > max_events:
        log.warning("Catalogue has %d events above M%.1f; capping at %d for ETAS. "
                    "Consider raising mag_min instead of truncating in time.",
                    len(df), mag_min, max_events)
        df = df.iloc[:max_events].copy()

    log.info("Catalogue loaded: %d events  (M%.1f+, %s - %s)",
             len(df), mag_min, DATE_START, DATE_END)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  ETAS MODEL + STOCHASTIC-DECLUSTERING EM
# ═══════════════════════════════════════════════════════════════════════════════

class ETAS:
    """
    Space-time ETAS fitted by the Zhuang/Ogata/Vere-Jones stochastic
    declustering EM algorithm.

    Parameters
    ----------
    times, lats, lons, mags : arrays of shape (N,)
    mc      : magnitude cutoff used in productivity & spatial-scale terms
    T_span  : total observation duration in days (for background-rate MLE)
    domain_area : study-area size in deg^2 (for background-rate MLE)
    """

    def __init__(self, times, lats, lons, mags, mc=MC,
                 init_params=None, max_iter=ETAS_MAX_ITER, tol=ETAS_TOL,
                 bg_bandwidth_nn=BG_BANDWIDTH_NN, optimizer_maxiter=ETAS_OPTIMIZER_MAXITER):

        self.N = len(times)
        self.times = np.asarray(times, dtype=float)
        self.lats  = np.asarray(lats,  dtype=float)
        self.lons  = np.asarray(lons,  dtype=float)
        self.mags  = np.asarray(mags,  dtype=float)
        self.mc    = mc

        p0 = dict(INIT_PARAMS)
        if init_params:
            p0.update(init_params)
        self.params = p0          # K, c, p, alpha, d, gamma, q
        self.max_iter = max_iter
        self.tol = tol
        self.bg_bandwidth_nn = bg_bandwidth_nn
        self.optimizer_maxiter = optimizer_maxiter

        # pairwise (i<j) time & space separations, precomputed once
        self._dt = np.zeros((self.N, self.N))
        self._dr2 = np.zeros((self.N, self.N))
        for j in range(1, self.N):
            self._dt[:j, j]  = self.times[j] - self.times[:j]
            self._dr2[:j, j] = ((self.lats[j]  - self.lats[:j])**2 +
                                 (self.lons[j]  - self.lons[:j])**2)

        self.W     = np.zeros((self.N, self.N))   # triggering probabilities
        self.w_bg  = np.ones(self.N) / self.N      # background probabilities
        self.mu    = np.full(self.N, 1.0 / self.N)  # background intensity at each event's location
        self._bg_bandwidth = np.full(self.N, 1.0)   # adaptive KDE bandwidths (deg)

        self.loglik_history = []

    # ── ETAS kernel pieces ──────────────────────────────────────────────────────

    def _productivity(self, i_idx):
        K, alpha = self.params["K"], self.params["alpha"]
        return K * 10.0 ** (alpha * (self.mags[i_idx] - self.mc))

    def _spatial_scale(self, i_idx):
        d, gamma = self.params["d"], self.params["gamma"]
        return d * 10.0 ** (gamma * (self.mags[i_idx] - self.mc))

    def _triggering_density(self, dt, dr2, d_i, m_i_scalar_idx=None):
        """
        Raw (unnormalised-in-i) space-time triggering intensity contributed
        by a parent at temporal lag dt >= 0 and squared spatial lag dr2,
        given that parent's spatial scale d_i.
          Omori-Utsu(dt) x GR-productivity x long-tailed spatial kernel
        """
        c, p, q = self.params["c"], self.params["p"], self.params["q"]
        temporal = (dt + c) ** (-p)
        spatial  = (q - 1.0) / (np.pi * d_i) * (1.0 + dr2 / d_i) ** (-q)
        return temporal * spatial

    def _rate_matrix(self):
        """
        Vectorised computation of the full (N, N) triggering-rate matrix.
        R[i,j] = productivity(i) * triggering_density(dt_ij, dr_ij^2 ; d_i)
        for i < j (causal), 0 otherwise. Shape (N, N).

        Replaces a Python-level loop over j (with per-j slicing) with
        NumPy broadcasting -- self._dt / self._dr2 are precomputed once in
        __init__, so each EM iteration's rate-matrix rebuild (needed
        because rho/d_i change every M-step) is now a handful of array
        ops instead of N Python-level iterations.
        """
        rho = self._productivity(np.arange(self.N))          # (N,)
        d_i = self._spatial_scale(np.arange(self.N))          # (N,)
        c, p, q = self.params["c"], self.params["p"], self.params["q"]

        temporal = (self._dt + c) ** (-p)                              # (N,N)
        spatial  = (q - 1.0) / (np.pi * d_i[:, None]) * (1.0 + self._dr2 / d_i[:, None]) ** (-q)
        R = rho[:, None] * temporal * spatial

        # exact causal mask: i < j (matches the original loop's range(j) semantics)
        causal = np.triu(np.ones((self.N, self.N), dtype=bool), k=1)
        R[~causal] = 0.0
        return R

    # ── background intensity: adaptive Gaussian KDE, weighted by w_bg ──────────

    def _update_background(self):
        """
        Re-estimate the spatially-variable background rate mu(x,y) via a
        variable-bandwidth Gaussian KDE over event locations, each event
        weighted by its current background probability w_bg[j] (Zhuang et
        al. 2002, sec. 3). Bandwidth at each point = distance to its
        bg_bandwidth_nn-th nearest neighbour (adaptive smoothing: sparser
        regions get wider kernels).

        Vectorised: the original Python loop over j (each iteration doing
        an O(N) kernel evaluation) is replaced by one (N,N) kernel matrix
        and a single weighted sum (matrix-vector product).
        """
        coords = np.column_stack([self.lats, self.lons])
        D = cdist(coords, coords)
        k = min(self.bg_bandwidth_nn, self.N - 1)
        # k-th nearest-neighbour distance (excluding self) as local bandwidth
        bw = np.sort(D, axis=1)[:, k]
        bw = np.maximum(bw, 1e-3)   # floor to avoid degenerate zero-bandwidth
        self._bg_bandwidth = bw

        # kern[i, j] = Gaussian kernel centred at event i, evaluated at j,
        # using event i's own adaptive bandwidth bw[i]
        kern = np.exp(-0.5 * D**2 / bw[:, None]**2) / (2 * np.pi * bw[:, None]**2)
        mu_new = self.w_bg @ kern   # mu_new[j] = sum_i w_bg[i] * kern[i,j]

        # normalise so that sum(mu) matches expected number of background events
        target = self.w_bg.sum()
        cur = mu_new.sum()
        if cur > 0:
            mu_new *= target / cur
        self.mu = mu_new

    # ── weighted MLE for (K, c, p, alpha, d, gamma, q) given current W ─────────

    def _negloglik_triggering(self, theta, w, dt, dr2, m_i):
        """
        Weighted negative log-likelihood of the Omori/GR/spatial triggering
        kernel, treating each (i,j) pair with i<j as a weighted observation
        with weight W[i,j] = current E-step probability that j was
        triggered by i. This is the standard "restricted likelihood" used
        for the ETAS M-step (Zhuang, 2011): it is proportional to the true
        space-time ETAS log-likelihood's triggering term when W already
        reflects the correct posterior, which holds at the EM fixed point.

        Takes pre-extracted sparse pair arrays (w, dt, dr2, m_i) rather
        than re-filtering the full (N,N) W matrix on every call -- the
        optimizer calls this hundreds of times per M-step (one per
        finite-difference gradient component), and re-scanning millions
        of matrix entries every single call was the dominant cost.
        """
        K, c, p, alpha, d, gamma, q = theta
        if K <= 0 or c <= 0 or p <= 1.0 or d <= 0 or q <= 1.0:
            return 1e12
        if len(w) == 0:
            return 1e12

        rho = K * 10.0 ** (alpha * (m_i - self.mc))
        d_i = d * 10.0 ** (gamma * (m_i - self.mc))

        log_rho      = np.log(rho + 1e-300)
        log_temporal = -p * np.log(dt + c)
        log_spatial  = (np.log(q - 1.0) - np.log(np.pi * d_i)
                        - q * np.log(1.0 + dr2 / d_i))

        ll = np.sum(w * (log_rho + log_temporal + log_spatial))

        # compensator (integral) term -- O(N), not O(pairs), always cheap
        T_max = self.times.max()
        rho_all = K * 10.0 ** (alpha * (self.mags - self.mc))
        time_left = np.maximum(T_max - self.times, 0.0)
        if abs(p - 1.0) > 1e-6:
            omori_integral = (c ** (1 - p) - (time_left + c) ** (1 - p)) / (p - 1)
        else:
            omori_integral = np.log((time_left + c) / c)
        compensator = np.sum(rho_all * omori_integral)

        return -(ll - compensator)

    def _update_triggering_params(self, optimizer_maxiter=20):
        # Extract the sparse (i,j) pairs ONCE per M-step call, not once per
        # objective-function evaluation (see _negloglik_triggering docstring).
        idx_i, idx_j = np.where(self.W > 1e-12)
        if len(idx_i) == 0:
            log.warning("  No significant triggering pairs (W all ~0); skipping M-step (b).")
            return
        w   = self.W[idx_i, idx_j]
        dt  = self._dt[idx_i, idx_j]
        dr2 = self._dr2[idx_i, idx_j]
        m_i = self.mags[idx_i]

        theta0 = np.array([self.params["K"], self.params["c"], self.params["p"],
                            self.params["alpha"], self.params["d"],
                            self.params["gamma"], self.params["q"]])
        bounds = [(1e-6, None), (1e-5, None), (1.001, 5.0),
                  (0.01, 5.0), (1e-6, None), (0.0, 3.0), (1.001, 5.0)]
        res = minimize(self._negloglik_triggering, theta0, args=(w, dt, dr2, m_i),
                        method="L-BFGS-B", bounds=bounds,
                        options=dict(maxiter=optimizer_maxiter))
        if res.success or res.status in (0, 1, 2):
            K, c, p, alpha, d, gamma, q = res.x
            self.params.update(K=K, c=c, p=p, alpha=alpha, d=d, gamma=gamma, q=q)
        else:
            log.warning("  M-step optimiser did not fully converge (%s); keeping previous params.",
                        res.message)

    # ── total observed-data log-likelihood (for convergence tracking) ──────────

    def _total_loglik(self, R):
        lam = self.mu + R.sum(axis=0)
        lam = np.maximum(lam, 1e-300)
        # point-process log-likelihood ~ sum(log lambda_j) - integral(lambda)
        # (integral term omitted from the tracked value since it is ~constant
        #  across iterations at convergence; used only to monitor progress)
        return np.sum(np.log(lam))

    # ── EM driver ────────────────────────────────────────────────────────────

    def fit(self):
        log.info("Starting ETAS stochastic-declustering EM  (N=%d, max_iter=%d)",
                 self.N, self.max_iter)
        if self.N > 8000:
            log.warning("N=%d is large for a dense (N,N) matrix (%.1f GB as float64). "
                         "Consider raising mc to shrink the catalogue.",
                         self.N, 8 * self.N**2 / 1e9)
        log.info("Initial params: %s", self.params)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):

            # ── E-step (vectorised): rate matrix + normalised weights ───────
            R = self._rate_matrix()
            beta_sum = R.sum(axis=0)               # shape (N,), sum over i<j
            total = self.mu + beta_sum
            safe_total = np.where(total <= 0, 1.0, total)

            self.w_bg = np.where(total <= 0, 1.0, self.mu / safe_total)
            self.W = R / safe_total[np.newaxis, :]
            self.W[:, total <= 0] = 0.0

            ll = self._total_loglik(R)
            self.loglik_history.append(ll)
            rel_change = abs(ll - prev_ll) / (abs(prev_ll) + 1e-12) if iteration > 0 else np.inf

            log.info("  Iter %3d  loglik=%.3f  Δ=%.2e  K=%.4f c=%.4f p=%.3f "
                     "alpha=%.3f d=%.4f gamma=%.3f q=%.3f  bg_frac=%.3f",
                     iteration + 1, ll, rel_change,
                     self.params["K"], self.params["c"], self.params["p"],
                     self.params["alpha"], self.params["d"], self.params["gamma"],
                     self.params["q"], self.w_bg.mean())

            if iteration > 0 and rel_change < self.tol:
                log.info("Converged at iteration %d.", iteration + 1)
                break

            # ── M-step (a): background intensity ─────────────────────────────
            self._update_background()

            # ── M-step (b): triggering kernel parameters ─────────────────────
            self._update_triggering_params(optimizer_maxiter=self.optimizer_maxiter)

            prev_ll = ll

        self.R = self._rate_matrix()
        log.info("ETAS fit complete.  Final params: %s", self.params)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD THE KNOWLEDGE GRAPH   (identical logic/semantics to seismic_kg_misd.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(df: pd.DataFrame, etas: ETAS,
                          threshold: float = W_THRESHOLD) -> nx.DiGraph:
    """
    Construct a directed graph where:
      - Nodes = earthquake events (attributes: time, lat, lon, depth, mag, id)
      - Edges = triggering relationships with weight w[i,j] >= threshold
      - Edge type: 'direct' or 'indirect' (cascade)

    An edge i->j is labelled 'indirect' if there exists at least one
    intermediate node k such that w[i,k] >= threshold AND w[k,j] >= threshold.
    """
    G = nx.DiGraph()

    for idx, row in df.iterrows():
        G.add_node(idx,
                   time      = str(row["datetime"]),
                   latitude  = float(row["latitude"]),
                   longitude = float(row["longitude"]),
                   depth     = float(row["depth"]),
                   mag       = float(row["mag"]),
                   event_id  = str(row.get("id", f"evt_{idx}")),
                   is_background = float(etas.w_bg[idx]))

    rows_i, cols_j = np.where(etas.W >= threshold)
    for i, j in zip(rows_i, cols_j):
        if i >= j:
            continue
        G.add_edge(int(i), int(j),
                   weight    = float(etas.W[i, j]),
                   edge_type = "direct")

    log.info("Labelling cascade (indirect) edges ...")
    for i, j in list(G.edges()):
        intermediates = set(G.successors(i)) & set(G.predecessors(j))
        if intermediates:
            G[i][j]["edge_type"] = "indirect"
            # GraphML doesn't support list-valued attributes -- store as a
            # comma-separated string instead.
            G[i][j]["via"] = ",".join(str(k) for k in list(intermediates)[:5])

    n_direct   = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "direct")
    n_indirect = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "indirect")
    log.info("Graph built:  %d nodes  |  %d edges  (%d direct, %d indirect)",
             G.number_of_nodes(), G.number_of_edges(), n_direct, n_indirect)
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_graph(G: nx.DiGraph, df: pd.DataFrame, etas: ETAS,
                 out_graphml=None, out_edge_csv=None, out_node_csv=None, out_params_csv=None):
    out_graphml    = out_graphml or OUT_GRAPHML
    out_edge_csv   = out_edge_csv or OUT_EDGE_CSV
    out_node_csv   = out_node_csv or OUT_NODE_CSV
    out_params_csv = out_params_csv or OUT_PARAMS_CSV

    nx.write_graphml(G, out_graphml)
    log.info("GraphML saved -> %s", out_graphml)

    edge_rows = []
    for i, j, data in G.edges(data=True):
        edge_rows.append({
            "source_idx":   i,
            "target_idx":   j,
            "weight":       data["weight"],
            "edge_type":    data["edge_type"],
            "source_time":  df.loc[i, "datetime"],
            "target_time":  df.loc[j, "datetime"],
            "source_mag":   df.loc[i, "mag"],
            "target_mag":   df.loc[j, "mag"],
        })
    pd.DataFrame(edge_rows).to_csv(out_edge_csv, index=False)
    log.info("Edge CSV saved -> %s", out_edge_csv)

    node_rows = []
    for node, data in G.nodes(data=True):
        node_rows.append({"idx": node, **data})
    pd.DataFrame(node_rows).to_csv(out_node_csv, index=False)
    log.info("Node CSV saved -> %s", out_node_csv)

    # Fitted ETAS parameters + convergence trace (useful for the thesis
    # write-up / appendix and for comparison against literature values)
    params_row = dict(etas.params)
    params_row["mc"] = etas.mc
    params_row["n_events"] = etas.N
    params_row["final_loglik"] = etas.loglik_history[-1] if etas.loglik_history else None
    params_row["n_em_iterations"] = len(etas.loglik_history)
    pd.DataFrame([params_row]).to_csv(out_params_csv, index=False)
    log.info("Fitted parameters saved -> %s", out_params_csv)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISE A SUBGRAPH   (identical styling to seismic_kg_misd.py)
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_subgraph(G: nx.DiGraph, df: pd.DataFrame,
                       seed_idx=None, hops=VIZ_HOP_RADIUS, out_prefix="seismic_kg_etas",
                       max_nodes=VIZ_MAX_NODES):
    """
    Plot the ego-network around the highest-magnitude event (or seed_idx).
    Node size ∝ magnitude.  Edge colour: blue=direct, orange=indirect.

    Layout: aftershock zones and swarms pack many events into a tiny
    geographic footprint, so plotting raw (longitude, latitude) as node
    positions causes heavy overlap. We seed a force-directed layout with
    the geographic positions (so the plot still roughly reflects real
    spatial structure) and let a short relaxation pull apart nodes that
    would otherwise stack on top of each other. If the ego-network is
    still very large, it's trimmed to the highest-magnitude nodes so the
    plot stays legible. (Identical approach to seismic_kg_misd.py.)
    """
    if seed_idx is None:
        candidates = [n for n in G.nodes() if G.out_degree(n) > 0
                      and df.loc[n, "mag"] >= VIZ_MIN_MAG]
        if not candidates:
            candidates = list(G.nodes())
        seed_idx = max(candidates, key=lambda n: df.loc[n, "mag"])

    log.info("Visualising ego-network around node %d  (M%.1f, %s)",
             seed_idx, df.loc[seed_idx, "mag"], df.loc[seed_idx, "datetime"])

    ego_nodes = {seed_idx}
    frontier  = {seed_idx}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier |= set(G.successors(n)) | set(G.predecessors(n))
        ego_nodes |= next_frontier
        frontier = next_frontier
        if len(ego_nodes) > max_nodes * 4:
            log.info("Ego network growing very large (%d nodes); stopping hop "
                      "expansion early.", len(ego_nodes))
            break

    if len(ego_nodes) > max_nodes:
        log.info("Ego network has %d nodes; trimming to the top %d by magnitude "
                  "(plus seed) for readability. Full graph is still in the "
                  "exported GraphML/CSV -- this only affects this one plot.",
                  len(ego_nodes), max_nodes)
        ranked = sorted(ego_nodes, key=lambda n: df.loc[n, "mag"], reverse=True)
        ego_nodes = set(ranked[:max_nodes]) | {seed_idx}

    sub = G.subgraph(ego_nodes).copy()
    n_nodes = sub.number_of_nodes()

    geo_pos = {n: (df.loc[n, "longitude"], df.loc[n, "latitude"]) for n in sub.nodes()}
    k_spacing = 0.5 / max(n_nodes, 1) ** 0.5
    pos = nx.spring_layout(sub, pos=geo_pos, k=k_spacing, iterations=150, seed=42)

    sizes = [22 * (df.loc[n, "mag"] ** 2) for n in sub.nodes()]

    edge_colours = ["#2979ff" if d["edge_type"] == "direct" else "#ff6d00"
                    for _, _, d in sub.edges(data=True)]
    edge_widths  = [1.8 * d["weight"] + 0.2 for _, _, d in sub.edges(data=True)]

    bg_probs   = [G.nodes[n].get("is_background", 0.5) for n in sub.nodes()]
    node_cmap  = cm.get_cmap("coolwarm")   # low (triggered) -> blue, high (background) -> red
    node_colors = [node_cmap(p) for p in bg_probs]

    fig_w = max(12, min(26, 8 + n_nodes / 15))
    fig_h = max(10, min(22, 7 + n_nodes / 18))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    nx.draw_networkx_nodes(sub, pos, node_size=sizes,
                           node_color=node_colors, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(sub, pos, edge_color=edge_colours,
                           width=edge_widths, ax=ax, alpha=0.55,
                           arrows=True, arrowsize=9,
                           connectionstyle="arc3,rad=0.1")

    labels = {n: f"M{df.loc[n,'mag']:.1f}"
              for n in sub.nodes() if df.loc[n, "mag"] >= VIZ_MIN_MAG}
    nx.draw_networkx_labels(sub, pos, labels=labels,
                            font_color="white", font_size=8, ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="#2979ff", lw=2, label="Direct trigger"),
        Line2D([0],[0], color="#ff6d00", lw=2, label="Indirect (cascade)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

    ax.set_title(
        f"ETAS Seismic Family Tree — ego-network around event {seed_idx}  "
        f"(M{df.loc[seed_idx,'mag']:.1f}, {str(df.loc[seed_idx,'datetime'])[:10]})\n"
        f"{n_nodes} nodes shown  |  Node colour: background probability (red=background, blue=triggered)",
        color="white", fontsize=10)
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_axis_off()

    plt.tight_layout()
    out_png = f"{out_prefix}_etas_subgraph.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info("Subgraph plot saved -> %s", out_png)
    plt.close(fig)


def plot_convergence(etas: ETAS, out_prefix="seismic_kg_etas"):
    """Quick diagnostic: log-likelihood trace across EM iterations."""
    if not etas.loglik_history:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(etas.loglik_history) + 1), etas.loglik_history,
            marker="o", color="#2979ff")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Observed-data log-likelihood (approx.)")
    ax.set_title("ETAS EM convergence")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = f"{out_prefix}_etas_convergence.png"
    plt.savefig(out_png, dpi=150)
    log.info("Convergence plot saved -> %s", out_png)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    args = parse_cli_args()

    if args.preset:
        preset = SEQUENCE_PRESETS[args.preset]
        catalogue_path = args.catalogue or preset["catalogue"]
        mag_min        = args.mag_min if args.mag_min is not None else preset["mag_min"]
        out_prefix     = args.out_prefix or preset["out_prefix"]
    else:
        catalogue_path = args.catalogue or CATALOGUE_PATH
        mag_min        = args.mag_min if args.mag_min is not None else MAG_MIN
        out_prefix     = args.out_prefix or "seismic_kg_etas"

    out_graphml    = f"{out_prefix}_kg_etas.graphml"
    out_edge_csv   = f"{out_prefix}_kg_etas_edges.csv"
    out_node_csv   = f"{out_prefix}_kg_etas_nodes.csv"
    out_params_csv = f"{out_prefix}_kg_etas_params.csv"

    # 1. Load
    df = load_catalogue(catalogue_path, mag_min=mag_min, max_events=args.max_events)

    t0 = df["datetime"].min()
    df["t_days"] = (df["datetime"] - t0).dt.total_seconds() / 86_400.0

    # 2. Fit ETAS
    etas = ETAS(
        times = df["t_days"].values,
        lats  = df["latitude"].values,
        lons  = df["longitude"].values,
        mags  = df["mag"].values,
        mc    = mag_min,
        max_iter = args.etas_max_iter,
        optimizer_maxiter = args.optimizer_maxiter,
    ).fit()

    # 3. Build KG
    G = build_knowledge_graph(df, etas, threshold=W_THRESHOLD)

    # 4. Export
    export_graph(G, df, etas, out_graphml, out_edge_csv, out_node_csv, out_params_csv)

    # 5. Visualise
    visualise_subgraph(G, df, seed_idx=VIZ_SEED_NODE, out_prefix=out_prefix)
    plot_convergence(etas, out_prefix=out_prefix)

    # 6. Print graph + parameter statistics
    print("\n" + "="*60)
    print("ETAS FITTED PARAMETERS")
    print("="*60)
    for k, v in etas.params.items():
        print(f"  {k:<8} = {v:.5f}")
    print(f"  mc       = {etas.mc}")
    print(f"  mean background fraction = {etas.w_bg.mean():.3f}")

    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Nodes              : {G.number_of_nodes():,}")
    print(f"Edges              : {G.number_of_edges():,}")
    print(f"Direct edges       : {sum(1 for _,_,d in G.edges(data=True) if d['edge_type']=='direct'):,}")
    print(f"Indirect edges     : {sum(1 for _,_,d in G.edges(data=True) if d['edge_type']=='indirect'):,}")

    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most influential events (by out-degree):")
    print(f"  {'Idx':>5}  {'Mag':>5}  {'Date':^22}  {'Out-deg':>7}  {'Lat':>7}  {'Lon':>8}")
    for idx, deg in out_deg:
        row = df.loc[idx]
        print(f"  {idx:>5}  {row['mag']:>5.1f}  {str(row['datetime'])[:19]:^22}  "
              f"{deg:>7}  {row['latitude']:>7.3f}  {row['longitude']:>8.3f}")
    print("="*60)
