"""
seismic_kg_misd_gaussov.py
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

import argparse
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
# 0.  CONFIGURATION  (edit these to taste, or override via CLI flags — see --help)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-sequence presets, tuned so each sequence yields a comparable-sized graph
# (roughly 1,500-6,000 events) despite very different regional completeness
# magnitudes. Run with e.g.  python seismic_kg_misd_gaussov.py --preset kumamoto
# Folder holding the extracted sequence CSVs, relative to wherever you run
# this script from (e.g. "../earthquake series" if you run from Kod/), or
# an absolute path. Leave as "" if the CSVs sit next to the script.
DATA_DIR = "../earthquake series"

SEQUENCE_PRESETS = {
    "tohoku":   dict(catalogue=f"{DATA_DIR}/jma_tohoku.csv",   mag_min=4.0, out_prefix="tohoku"),
    "kumamoto": dict(catalogue=f"{DATA_DIR}/jma_kumamoto.csv", mag_min=2.0, out_prefix="kumamoto"),
    "noto":     dict(catalogue=f"{DATA_DIR}/jma_noto.csv",     mag_min=2.0, out_prefix="noto"),
}

CATALOGUE_PATH   = "merged_japan_earthquakes.csv"
OUT_GRAPHML      = "seismic_kg.graphml"
OUT_EDGE_CSV     = "seismic_kg_edges.csv"
OUT_NODE_CSV     = "seismic_kg_nodes.csv"

# -- Catalogue filter ----------------------------------------------------------
MAG_MIN          = 4.5          # only keep M ≥ this (keeps the matrix tractable)
DATE_START       = "1900-01-01"
DATE_END         = "2030-12-31"
MAX_EVENTS       = 8000         # safety cap; dense NxN matrix at 8000 events ≈ 512MB


def parse_cli_args():
    p = argparse.ArgumentParser(description="Build a MISD seismic family-tree graph.")
    p.add_argument("--preset", choices=SEQUENCE_PRESETS.keys(),
                    help="Use a built-in (catalogue, mag_min, out_prefix) combo for one of the case studies")
    p.add_argument("--catalogue", help="Path to the extracted sequence CSV (overrides --preset)")
    p.add_argument("--mag-min", type=float, help="Completeness magnitude cutoff (overrides --preset)")
    p.add_argument("--max-events", type=int, default=MAX_EVENTS)
    p.add_argument("--out-prefix", help="Prefix for output files, e.g. 'kumamoto' -> kumamoto_kg.graphml")
    return p.parse_args()

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
VIZ_MAX_NODES    = 150          # cap ego-network size for readability (see visualise_subgraph)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & FILTER CATALOGUE
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogue(path: str, mag_min: float = MAG_MIN,
                    max_events: int = MAX_EVENTS) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Parse time
    df["datetime"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude", "depth", "mag"])

    # -- Magnitude determination method ---------------------------------------
    # JMA extracts carry a magType column: 'V' (velocity-type) vs 'D'
    # (displacement-type) are BOTH native JMA-scale (Mj) magnitudes, not
    # different scales -- JMA switches determination method as magnitude
    # grows (velocity-type saturates for large ruptures), so V and D are
    # strongly correlated with event size. Filtering to 'V' only, as an
    # earlier version of this script did, silently discards most of the
    # largest events in a sequence -- do NOT do that. Only genuinely
    # missing/unparseable magType rows are dropped here.
    if "magType" in df.columns:
        n_before = len(df)
        df = df.dropna(subset=["magType"])
        n_dropped = n_before - len(df)
        if n_dropped:
            log.info("Dropped %d events with missing magType.", n_dropped)

    # Filter
    df = df[df["mag"] >= mag_min]
    df = df[df["datetime"] >= pd.Timestamp(DATE_START, tz="UTC")]
    df = df[df["datetime"] <= pd.Timestamp(DATE_END,   tz="UTC")]
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) > max_events:
        log.warning("Catalogue has %d events above M%.1f; capping at %d for MISD. "
                    "Consider raising mag_min instead of truncating in time.",
                    len(df), mag_min, max_events)
        df = df.iloc[:max_events].copy()

    log.info("Catalogue loaded: %d events  (M%.1f+, %s – %s)",
             len(df), mag_min, DATE_START, DATE_END)
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

    def _rate_matrix(self):
        """
        Vectorised computation of the full (N, N) triggering-rate matrix.

        R[i, j] = ρ(m_i) · g_t(t_j - t_i) · g_s(r_ij)   for t_j > t_i, else 0.

        This replaces a pure-Python O(N^2) double loop with NumPy
        broadcasting -- same math, but the whole matrix is built with a
        handful of array ops instead of N^2 Python-level iterations. At
        N ~ a few thousand this is the difference between minutes/hours
        and well under a second; the dense (N, N) float64 matrix costs
        8*N^2 bytes (e.g. ~320MB at N=6000), so keep N in that range
        (control this via mag_min / MAX_EVENTS -- see load_catalogue()).
        """
        dt = self.times[np.newaxis, :] - self.times[:, np.newaxis]   # dt[i,j] = t_j - t_i
        dlat = self.lats[np.newaxis, :] - self.lats[:, np.newaxis]
        dlon = self.lons[np.newaxis, :] - self.lons[:, np.newaxis]
        dr = np.sqrt(dlat**2 + dlon**2)

        causal = dt > 0

        g_t = np.exp(-0.5 * (dt / self.sigma_t) ** 2) / (self.sigma_t * np.sqrt(2 * np.pi))
        g_s = np.exp(-0.5 * (dr / self.sigma_s) ** 2) / (2 * np.pi * self.sigma_s ** 2)

        R = self.rho[:, np.newaxis] * g_t * g_s
        R[~causal] = 0.0
        return R, dt, dr

    # ── EM ─────────────────────────────────────────────────────────────────────

    def fit(self):
        log.info("Starting MISD EM  (N=%d, max_iter=%d)", self.N, self.max_iter)
        if self.N > 8000:
            log.warning("N=%d is large for a dense (N,N) matrix (%.1f GB as float64). "
                         "Consider raising mag_min to shrink the catalogue.",
                         self.N, 8 * self.N**2 / 1e9)

        log.info("Pre-computing triggering rate matrix (vectorised) …")
        R, dt, dr = self._rate_matrix()

        mu_old = self.mu.copy()

        for iteration in range(self.max_iter):

            # ── E-step (vectorised) ──────────────────────────────────────────
            # For each event j:  α_j = μ_j  (background),  β_ij = R[i,j] for i<j
            # (R is already zero for non-causal pairs, so column sums are safe)
            beta_sum = R.sum(axis=0)                    # shape (N,)
            total = self.mu + beta_sum
            safe_total = np.where(total == 0, 1.0, total)

            self.w_bg = np.where(total == 0, 1.0, self.mu / safe_total)
            self.W = R / safe_total[np.newaxis, :]
            self.W[:, total == 0] = 0.0

            # ── M-step (vectorised) ──────────────────────────────────────────
            self.mu = self.w_bg / self.w_bg.sum()

            total_w = self.W.sum()
            if total_w > 0:
                self.sigma_t = np.sqrt((self.W * dt**2).sum() / total_w)
                self.sigma_s = np.sqrt((self.W * dr**2).sum() / total_w)
                self.sigma_t = max(self.sigma_t, 0.1)   # numerical floor
                self.sigma_s = max(self.sigma_s, 0.01)

            # ── convergence check ─────────────────────────────────────────────
            rel_change = np.abs(self.mu - mu_old).sum() / (np.abs(mu_old).sum() + 1e-12)
            log.info("  Iter %3d  |Δμ|/|μ| = %.2e  σ_t=%.2f d  σ_s=%.3f°",
                     iteration + 1, rel_change, self.sigma_t, self.sigma_s)

            if rel_change < self.tol:
                log.info("Converged at iteration %d.", iteration + 1)
                break

            # Recompute R with updated kernels (dt, dr are geometry-only, unchanged)
            R, dt, dr = self._rate_matrix()
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
            # GraphML doesn't support list-valued attributes -- store as a
            # comma-separated string instead (still fully readable/parseable).
            G[i][j]["via"] = ",".join(str(k) for k in list(intermediates)[:5])

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
                       seed_idx=None, hops=VIZ_HOP_RADIUS, out_prefix="seismic_kg",
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
    plot stays legible.
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

    # Layout: geographic positions as a starting point, force-directed
    # relaxation to resolve overlap.
    geo_pos = {n: (df.loc[n, "longitude"], df.loc[n, "latitude"]) for n in sub.nodes()}
    k_spacing = 0.5 / max(n_nodes, 1) ** 0.5
    pos = nx.spring_layout(sub, pos=geo_pos, k=k_spacing, iterations=150, seed=42)

    # Node sizes ∝ magnitude
    sizes = [22 * (df.loc[n, "mag"] ** 2) for n in sub.nodes()]

    # Edge colours
    edge_colours = ["#2979ff" if d["edge_type"] == "direct" else "#ff6d00"
                    for _, _, d in sub.edges(data=True)]
    edge_widths  = [1.8 * d["weight"] + 0.2 for _, _, d in sub.edges(data=True)]

    # Node colours: background probability
    bg_probs   = [G.nodes[n].get("is_background", 0.5) for n in sub.nodes()]
    node_cmap  = cm.get_cmap("coolwarm")
    node_colors = [node_cmap(p) for p in bg_probs]

    # Scale figure size with node count so dense sequences get more room
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

    # Label only high-magnitude nodes
    labels = {n: f"M{df.loc[n,'mag']:.1f}"
              for n in sub.nodes() if df.loc[n, "mag"] >= VIZ_MIN_MAG}
    nx.draw_networkx_labels(sub, pos, labels=labels,
                            font_color="white", font_size=8, ax=ax)

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
        f"{n_nodes} nodes shown  |  Node colour: background probability (red=background, blue=triggered)",
        color="white", fontsize=10)
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_axis_off()

    plt.tight_layout()
    out_png = f"{out_prefix}_subgraph.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info("Subgraph plot saved → %s", out_png)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    args = parse_cli_args()

    # Resolve config: --preset provides defaults, explicit flags override it
    if args.preset:
        preset = SEQUENCE_PRESETS[args.preset]
        catalogue_path = args.catalogue or preset["catalogue"]
        mag_min        = args.mag_min if args.mag_min is not None else preset["mag_min"]
        out_prefix     = args.out_prefix or preset["out_prefix"]
    else:
        catalogue_path = args.catalogue or CATALOGUE_PATH
        mag_min        = args.mag_min if args.mag_min is not None else MAG_MIN
        out_prefix     = args.out_prefix or "seismic_kg"

    OUT_GRAPHML  = f"{out_prefix}_kg.graphml"
    OUT_EDGE_CSV = f"{out_prefix}_kg_edges.csv"
    OUT_NODE_CSV = f"{out_prefix}_kg_nodes.csv"

    # 1. Load
    df = load_catalogue(catalogue_path, mag_min=mag_min, max_events=args.max_events)

    # Convert datetime to float days since first event (MISD needs numeric time)
    t0 = df["datetime"].min()
    df["t_days"] = (df["datetime"] - t0).dt.total_seconds() / 86_400.0

    # 2. Run MISD
    misd = MISD(
        times   = df["t_days"].values,
        lats    = df["latitude"].values,
        lons    = df["longitude"].values,
        mags    = df["mag"].values,
        mag_min = mag_min,
    ).fit()

    # 3. Build KG
    G = build_knowledge_graph(df, misd, threshold=W_THRESHOLD)

    # 4. Export
    export_graph(G, df)

    # 5. Visualise
    visualise_subgraph(G, df, seed_idx=VIZ_SEED_NODE, out_prefix=out_prefix)

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
