"""
seismic_kg_misd.py
==================
Builds a seismic "family tree" Knowledge Graph (KG) from the merged Japan
earthquake catalogue using the Model-Independent Stochastic Declustering
(MISD) algorithm (Marsan & Lengliné, 2008).

Pipeline
--------
  1. Load & filter the merged catalogue
  2. Run MISD (EM algorithm) → triggering probabilities w[i,j]
  3. Build a directed NetworkX graph from the w[i,j] matrix
  4. Identify direct vs. indirect (cascade) edges
  5. Export the graph (GraphML + edge-list CSV) for downstream use
  6. Print a summary and plot a small subgraph

Dependencies
------------
  pip install pandas numpy scipy networkx matplotlib tqdm
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION  (edit these to taste)
# ═══════════════════════════════════════════════════════════════════════════════

CATALOGUE_PATH   = "merged_japan_earthquakes.csv"
OUT_GRAPHML      = "seismic_kg.graphml"
OUT_EDGE_CSV     = "seismic_kg_edges.csv"
OUT_NODE_CSV     = "seismic_kg_nodes.csv"

# -- Catalogue filter ----------------------------------------------------------
MAG_MIN          = 4.5          # only keep M ≥ this (keeps the matrix tractable)
DATE_START       = "2000-01-01"
DATE_END         = "2023-12-31"
MAX_EVENTS       = 2000         # safety cap; increase if you have RAM to spare

# -- MISD hyper-parameters -----------------------------------------------------
MISD_MAX_ITER    = 50           # EM iterations
MISD_TOL         = 1e-4         # convergence threshold (relative change in μ)

# Temporal kernel bandwidth σ_t  [days]  – starting point, learned by EM
SIGMA_T_INIT     = 10.0

# Spatial kernel bandwidth σ_s  [degrees] – starting point, learned by EM
SIGMA_S_INIT     = 0.5

# b-value for the productivity term  n(m) ∝ 10^(b*(m-mmin))
B_VALUE          = 1.0

# -- Graph edge threshold ------------------------------------------------------
# Only draw an edge i→j when w[i,j] ≥ this (avoids a fully-dense graph)
W_THRESHOLD      = 0.05

# -- Visualisation -------------------------------------------------------------
VIZ_SEED_NODE    = None         # set to an integer index to pick a seed, or None
VIZ_HOP_RADIUS   = 2            # neighbourhood hops shown in the subgraph plot
VIZ_MIN_MAG      = 5.5          # only seed nodes with mag ≥ this


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & FILTER CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogue(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Parse time
    df["datetime"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude", "depth", "mag"])

    # Filter
    df = df[df["mag"] >= MAG_MIN]
    df = df[df["datetime"] >= pd.Timestamp(DATE_START, tz="UTC")]
    df = df[df["datetime"] <= pd.Timestamp(DATE_END,   tz="UTC")]
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) > MAX_EVENTS:
        log.warning("Catalogue has %d events above M%.1f; capping at %d for MISD.",
                    len(df), MAG_MIN, MAX_EVENTS)
        df = df.iloc[:MAX_EVENTS].copy()

    log.info("Catalogue loaded: %d events  (M%.1f+, %s – %s)",
             len(df), MAG_MIN, DATE_START, DATE_END)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  MISD ALGORITHM  (Marsan & Lengliné, 2008)
# ═══════════════════════════════════════════════════════════════════════════════
#
# We model the conditional intensity as:
#
#   λ(t,x,y) = μ(x,y)  +  Σ_{i: t_i < t}  ρ(m_i) · g(t-t_i, r_i)
#
# where
#   μ(x,y)     = background rate (estimated from the data)
#   ρ(m_i)     = productivity  = 10^(b*(m_i - m_min))
#   g(Δt, Δr)  = triggering kernel (Gaussian in both time and space)
#
# The EM algorithm alternates between:
#   E-step: compute w[i,j] = probability that event j was triggered by event i
#           (and w[j,background] = probability j is a background event)
#   M-step: re-estimate μ, σ_t, σ_s from the w weights
# ═══════════════════════════════════════════════════════════════════════════════

class MISD:
    """
    Simplified MISD for a finite catalogue.

    Parameters
    ----------
    times  : array of shape (N,)  – origin times in days (float)
    lats   : array of shape (N,)  – latitudes  [degrees]
    lons   : array of shape (N,)  – longitudes [degrees]
    mags   : array of shape (N,)  – magnitudes
    """

    def __init__(self, times, lats, lons, mags,
                 b=B_VALUE, mag_min=MAG_MIN,
                 sigma_t=SIGMA_T_INIT, sigma_s=SIGMA_S_INIT,
                 max_iter=MISD_MAX_ITER, tol=MISD_TOL):

        self.N       = len(times)
        self.times   = np.asarray(times, dtype=float)
        self.lats    = np.asarray(lats,  dtype=float)
        self.lons    = np.asarray(lons,  dtype=float)
        self.mags    = np.asarray(mags,  dtype=float)

        self.b       = b
        self.mag_min = mag_min
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.max_iter = max_iter
        self.tol      = tol

        # productivity  ρ(m) = 10^(b*(m - m_min))
        self.rho = 10.0 ** (self.b * (self.mags - self.mag_min))

        # w[i,j]  = P(j triggered by i),   shape (N, N),  only lower-triangle used
        # w_bg[j] = P(j is background)
        self.W    = np.zeros((self.N, self.N))
        self.w_bg = np.ones(self.N) / self.N

        # background rate (uniform initialisation)
        self.mu = np.full(self.N, 1.0 / self.N)

    # ── kernel ─────────────────────────────────────────────────────────────────

    def _temporal_kernel(self, dt):
        """Gaussian temporal kernel  g_t(Δt), Δt > 0."""
        return np.exp(-0.5 * (dt / self.sigma_t) ** 2) / (
            self.sigma_t * np.sqrt(2 * np.pi))

    def _spatial_kernel(self, dr):
        """Isotropic Gaussian spatial kernel  g_s(Δr)."""
        return np.exp(-0.5 * (dr / self.sigma_s) ** 2) / (
            2 * np.pi * self.sigma_s ** 2)

    def _triggering_rate(self, i, j):
        """
        Raw triggering rate from event i to event j (j must be later).
        ρ(m_i) · g_t(t_j - t_i) · g_s(r_ij)
        """
        dt = self.times[j] - self.times[i]
        if dt <= 0:
            return 0.0
        dr = np.sqrt((self.lats[j] - self.lats[i])**2 +
                     (self.lons[j] - self.lons[i])**2)
        return self.rho[i] * self._temporal_kernel(dt) * self._spatial_kernel(dr)

    # ── EM ─────────────────────────────────────────────────────────────────────

    def fit(self):
        log.info("Starting MISD EM  (N=%d, max_iter=%d)", self.N, self.max_iter)

        # Pre-compute pairwise triggering rates (lower-triangle only)
        log.info("Pre-computing triggering rate matrix …")
        R = np.zeros((self.N, self.N))
        for j in tqdm(range(1, self.N), desc="  Rate matrix"):
            for i in range(j):
                R[i, j] = self._triggering_rate(i, j)

        mu_old = self.mu.copy()

        for iteration in range(self.max_iter):

            # ── E-step ────────────────────────────────────────────────────────
            # For each event j, compute unnormalised weights
            #   α_j  = μ_j                         (background)
            #   β_ij = R[i,j]  for i < j           (triggered by i)
            # then normalise so they sum to 1.

            for j in range(self.N):
                alpha_j = self.mu[j]
                beta_j  = R[:j, j]           # shape (j,)
                total   = alpha_j + beta_j.sum()
                if total == 0:
                    self.w_bg[j]  = 1.0
                    self.W[:j, j] = 0.0
                else:
                    self.w_bg[j]  = alpha_j / total
                    self.W[:j, j] = beta_j  / total

            # ── M-step ────────────────────────────────────────────────────────
            # Re-estimate background rates μ_j  (proportional to w_bg[j])
            self.mu = self.w_bg / self.w_bg.sum()

            # Re-estimate σ_t and σ_s as weighted RMS of Δt and Δr
            w_trig = self.W.copy()   # only lower-triangle matters
            total_w = w_trig.sum()

            if total_w > 0:
                # weighted variance of Δt
                dt_mat = np.zeros((self.N, self.N))
                dr_mat = np.zeros((self.N, self.N))
                for j in range(1, self.N):
                    dt_mat[:j, j] = self.times[j] - self.times[:j]
                    dr_mat[:j, j] = np.sqrt(
                        (self.lats[j]  - self.lats[:j])**2 +
                        (self.lons[j]  - self.lons[:j])**2)

                self.sigma_t = np.sqrt((w_trig * dt_mat**2).sum() / total_w)
                self.sigma_s = np.sqrt((w_trig * dr_mat**2).sum() / total_w)
                self.sigma_t = max(self.sigma_t, 0.1)   # numerical floor
                self.sigma_s = max(self.sigma_s, 0.01)

            # ── convergence check ─────────────────────────────────────────────
            rel_change = np.abs(self.mu - mu_old).sum() / (np.abs(mu_old).sum() + 1e-12)
            log.info("  Iter %3d  |Δμ|/|μ| = %.2e  σ_t=%.2f d  σ_s=%.3f°",
                     iteration + 1, rel_change, self.sigma_t, self.sigma_s)

            if rel_change < self.tol:
                log.info("Converged at iteration %d.", iteration + 1)
                break

            # Recompute R with updated kernels
            for j in tqdm(range(1, self.N), desc=f"  Iter {iteration+1} rate matrix", leave=False):
                for i in range(j):
                    R[i, j] = self._triggering_rate(i, j)

            mu_old = self.mu.copy()

        self.R = R
        log.info("MISD complete.  σ_t=%.2f d  σ_s=%.3f°", self.sigma_t, self.sigma_s)
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD THE KNOWLEDGE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(df: pd.DataFrame, misd: MISD,
                          threshold: float = W_THRESHOLD) -> nx.DiGraph:
    """
    Construct a directed graph where:
      - Nodes  = earthquake events  (attributes: time, lat, lon, depth, mag, id)
      - Edges  = triggering relationships with weight w[i,j] ≥ threshold
      - Edge type: 'direct' or 'indirect' (cascade)

    An edge i→j is labelled 'indirect' if there exists at least one intermediate
    node k such that  w[i,k] ≥ threshold  AND  w[k,j] ≥ threshold.
    """
    G = nx.DiGraph()

    # ── nodes ─────────────────────────────────────────────────────────────────
    for idx, row in df.iterrows():
        G.add_node(idx,
                   time      = str(row["datetime"]),
                   latitude  = float(row["latitude"]),
                   longitude = float(row["longitude"]),
                   depth     = float(row["depth"]),
                   mag       = float(row["mag"]),
                   event_id  = str(row.get("id", f"evt_{idx}")),
                   is_background = float(misd.w_bg[idx]))

    # ── edges (direct candidates) ─────────────────────────────────────────────
    rows_i, cols_j = np.where(misd.W >= threshold)
    for i, j in zip(rows_i, cols_j):
        if i >= j:          # only causal (i before j)
            continue
        G.add_edge(int(i), int(j),
                   weight     = float(misd.W[i, j]),
                   edge_type  = "direct")   # will be refined below

    # ── label indirect (cascade) edges ────────────────────────────────────────
    # For each edge i→j, check if ∃k such that i→k and k→j both exist.
    log.info("Labelling cascade (indirect) edges …")
    for i, j in list(G.edges()):
        intermediates = set(G.successors(i)) & set(G.predecessors(j))
        if intermediates:
            G[i][j]["edge_type"] = "indirect"
            G[i][j]["via"]       = list(intermediates)[:5]   # store up to 5

    n_direct   = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "direct")
    n_indirect = sum(1 for _, _, d in G.edges(data=True) if d["edge_type"] == "indirect")
    log.info("Graph built:  %d nodes  |  %d edges  (%d direct, %d indirect)",
             G.number_of_nodes(), G.number_of_edges(), n_direct, n_indirect)
    return G


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_graph(G: nx.DiGraph, df: pd.DataFrame):
    # GraphML (preserves all attributes, loadable by Gephi / Neo4j / PyVis)
    nx.write_graphml(G, OUT_GRAPHML)
    log.info("GraphML saved → %s", OUT_GRAPHML)

    # Edge list CSV
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
    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(OUT_EDGE_CSV, index=False)
    log.info("Edge CSV saved → %s", OUT_EDGE_CSV)

    # Node list CSV
    node_rows = []
    for node, data in G.nodes(data=True):
        node_rows.append({"idx": node, **data})
    node_df = pd.DataFrame(node_rows)
    node_df.to_csv(OUT_NODE_CSV, index=False)
    log.info("Node CSV saved → %s", OUT_NODE_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  VISUALISE A SUBGRAPH
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_subgraph(G: nx.DiGraph, df: pd.DataFrame,
                       seed_idx=None, hops=VIZ_HOP_RADIUS):
    """
    Plot the ego-network around the highest-magnitude event (or seed_idx).
    Node size ∝ magnitude.  Edge colour: blue=direct, orange=indirect.
    """
    if seed_idx is None:
        # Pick the highest-magnitude node that has at least one out-edge
        candidates = [n for n in G.nodes() if G.out_degree(n) > 0
                      and df.loc[n, "mag"] >= VIZ_MIN_MAG]
        if not candidates:
            candidates = list(G.nodes())
        seed_idx = max(candidates, key=lambda n: df.loc[n, "mag"])

    log.info("Visualising ego-network around node %d  (M%.1f, %s)",
             seed_idx, df.loc[seed_idx, "mag"], df.loc[seed_idx, "datetime"])

    # Build ego subgraph
    ego_nodes = {seed_idx}
    frontier  = {seed_idx}
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier |= set(G.successors(n)) | set(G.predecessors(n))
        ego_nodes |= next_frontier
        frontier = next_frontier

    sub = G.subgraph(ego_nodes).copy()

    # Layout: longitude / latitude as (x, y)
    pos = {n: (df.loc[n, "longitude"], df.loc[n, "latitude"]) for n in sub.nodes()}

    # Node sizes ∝ magnitude
    sizes = [30 * (df.loc[n, "mag"] ** 2) for n in sub.nodes()]

    # Edge colours
    edge_colours = ["#2979ff" if d["edge_type"] == "direct" else "#ff6d00"
                    for _, _, d in sub.edges(data=True)]
    edge_widths  = [2.5 * d["weight"] + 0.3 for _, _, d in sub.edges(data=True)]

    # Node colours: background probability
    bg_probs   = [G.nodes[n].get("is_background", 0.5) for n in sub.nodes()]
    node_cmap  = cm.get_cmap("coolwarm_r")
    node_colors = [node_cmap(p) for p in bg_probs]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor("#0d0d0d")
    fig.patch.set_facecolor("#0d0d0d")

    nx.draw_networkx_nodes(sub, pos, node_size=sizes,
                           node_color=node_colors, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(sub, pos, edge_color=edge_colours,
                           width=edge_widths, ax=ax, alpha=0.75,
                           arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.1")

    # Label only high-magnitude nodes
    labels = {n: f"M{df.loc[n,'mag']:.1f}"
              for n in sub.nodes() if df.loc[n, "mag"] >= VIZ_MIN_MAG}
    nx.draw_networkx_labels(sub, pos, labels=labels,
                            font_color="white", font_size=7, ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color="#2979ff", lw=2, label="Direct trigger"),
        Line2D([0],[0], color="#ff6d00", lw=2, label="Indirect (cascade)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)

    ax.set_title(
        f"Seismic Family Tree — ego-network around event {seed_idx}  "
        f"(M{df.loc[seed_idx,'mag']:.1f}, {str(df.loc[seed_idx,'datetime'])[:10]})\n"
        f"Node colour: background probability (red=background, blue=triggered)",
        color="white", fontsize=10)
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()
    plt.savefig("seismic_kg_subgraph.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info("Subgraph plot saved → seismic_kg_subgraph.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Load
    df = load_catalogue(CATALOGUE_PATH)

    # Convert datetime to float days since first event (MISD needs numeric time)
    t0 = df["datetime"].min()
    df["t_days"] = (df["datetime"] - t0).dt.total_seconds() / 86_400.0

    # 2. Run MISD
    misd = MISD(
        times  = df["t_days"].values,
        lats   = df["latitude"].values,
        lons   = df["longitude"].values,
        mags   = df["mag"].values,
    ).fit()

    # 3. Build KG
    G = build_knowledge_graph(df, misd, threshold=W_THRESHOLD)

    # 4. Export
    export_graph(G, df)

    # 5. Visualise
    visualise_subgraph(G, df, seed_idx=VIZ_SEED_NODE)

    # 6. Print graph statistics
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Nodes              : {G.number_of_nodes():,}")
    print(f"Edges              : {G.number_of_edges():,}")
    print(f"Direct edges       : {sum(1 for _,_,d in G.edges(data=True) if d['edge_type']=='direct'):,}")
    print(f"Indirect edges     : {sum(1 for _,_,d in G.edges(data=True) if d['edge_type']=='indirect'):,}")

    # Top 10 most influential earthquakes (highest out-degree = most triggers)
    out_deg = sorted(G.out_degree(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most influential events (by out-degree):")
    print(f"  {'Idx':>5}  {'Mag':>5}  {'Date':^22}  {'Out-deg':>7}  {'Lat':>7}  {'Lon':>8}")
    for idx, deg in out_deg:
        row = df.loc[idx]
        print(f"  {idx:>5}  {row['mag']:>5.1f}  {str(row['datetime'])[:19]:^22}  "
              f"{deg:>7}  {row['latitude']:>7.3f}  {row['longitude']:>8.3f}")
    print("="*60)
