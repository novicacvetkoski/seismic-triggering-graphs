"""
kg_common.py
============
Code shared by seismic_kg_misd.py and seismic_kg_etas.py, so that the two models are run
on IDENTICAL data and produce IDENTICAL downstream objects (graph, node/edge tables,
verification, exports). Keep this file in the same folder as the two scripts.

Design decisions (from the supervisor's guidelines)
  * An edge i -> j means "P(i is the parent of j)". Every edge is a direct probability edge.
    An edge is NOT called "indirect" because an alternative path exists; instead each edge
    carries `is_best_parent` (1 if i is the single most probable parent of j).
  * Importance of an event = weighted OUT-STRENGTH = sum_j w[i,j] over ALL children (the
    original estimated probabilities, not only edges above the visualisation threshold).
  * Every run writes probability checks (finite, non-negative, sum to 1, causal order),
    a pairs file with the estimated probabilities, and a JSON with configuration and diagnostics.
"""
import json
import logging
import platform
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

log = logging.getLogger("kg")

EARTH_R_KM = 6371.0

# Folder holding the extracted sequence CSVs, relative to where you run the scripts
# (e.g. from Kod/), or an absolute path.
DATA_DIR = "../Earthquake-series"

# bg_rate = imposed background rate [events / day / km^2] estimated from the years BEFORE each
# sequence (estimate_background_rate.py, 'robust' = median monthly count). Used by default
# for both models so that they share exactly the same background.
SEQUENCE_PRESETS = {
    "tohoku":   dict(catalogue=f"{DATA_DIR}/jma_tohoku.csv",   mag_min=4.0, out_prefix="tohoku",   bg_rate=1.2364e-06),
    "kumamoto": dict(catalogue=f"{DATA_DIR}/jma_kumamoto.csv", mag_min=2.0, out_prefix="kumamoto", bg_rate=3.9515e-05),
    "noto":     dict(catalogue=f"{DATA_DIR}/jma_noto.csv",     mag_min=2.0, out_prefix="noto",     bg_rate=8.3677e-06),
}

DATE_START = "1900-01-01"
DATE_END = "2030-12-31"
MAX_EVENTS = 8000
W_THRESHOLD = 0.05          # visualisation / graph-edge threshold ONLY (analysis uses all probabilities)
PAIR_EXPORT_CUTOFF = 1e-4   # pairs with probability below this are not written to the pairs file

VIZ_HOP_RADIUS = 2
VIZ_MIN_MAG = 5.5
VIZ_MAX_NODES = 150


# ═══════════════════════════════════════════════════════════════════════════════
# data
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalogue(path, mag_min, max_events=MAX_EVENTS):
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude", "depth", "mag"])

    # JMA magType: 'V' (velocity-type) and 'D' (displacement-type) are both native JMA-scale
    # magnitudes; JMA switches V -> D as events get large. Never filter to 'V' only.
    if "magType" in df.columns:
        n0 = len(df)
        df = df.dropna(subset=["magType"])
        if n0 - len(df):
            log.info("Dropped %d events with missing magType.", n0 - len(df))

    df = df[df["mag"] >= mag_min]
    df = df[(df["datetime"] >= pd.Timestamp(DATE_START, tz="UTC")) &
            (df["datetime"] <= pd.Timestamp(DATE_END, tz="UTC"))]
    df = df.sort_values("datetime").reset_index(drop=True)
    if len(df) > max_events:
        log.warning("Catalogue has %d events above M%.1f; capping at %d (raise mag_min instead of truncating).",
                    len(df), mag_min, max_events)
        df = df.iloc[:max_events].copy()
    log.info("Catalogue loaded: %d events  (M%.1f+)", len(df), mag_min)
    return df


def project_km(lats, lons):
    """Equirectangular projection to km around the mean latitude (same for both models)."""
    lat0 = np.radians(np.mean(lats))
    return (EARTH_R_KM * np.radians(lons) * np.cos(lat0), EARTH_R_KM * np.radians(lats))


def domain_area_km2(x, y):
    return max((x.max() - x.min()) * (y.max() - y.min()), 1.0)


def resolve_config(args, default_model_bg):
    """Resolve catalogue path, Mc, output prefix, background mode and rate from preset + CLI."""
    pr = SEQUENCE_PRESETS.get(args.preset) if args.preset else None
    path = args.catalogue or (pr["catalogue"] if pr else "merged_japan_earthquakes.csv")
    mag_min = args.mag_min if args.mag_min is not None else (pr["mag_min"] if pr else 4.5)
    prefix = args.out_prefix or (pr["out_prefix"] if pr else "seismic_kg")
    bg = args.background or ("fixed" if (pr or args.bg_rate) else default_model_bg)
    rate = args.bg_rate if args.bg_rate is not None else (pr["bg_rate"] if (pr and bg == "fixed") else None)
    if bg == "fixed" and not rate:
        raise SystemExit("--background fixed needs --bg-rate (events/day/km^2) or a --preset")
    return path, mag_min, prefix, bg, rate


# ═══════════════════════════════════════════════════════════════════════════════
# probabilities -> node / edge quantities
# ═══════════════════════════════════════════════════════════════════════════════

def best_parent_from_pairs(N, parent, child, w):
    """Most probable parent of every event (-1 if it has no candidate) and its probability."""
    best = np.full(N, -1, dtype=np.int64)
    bw = np.zeros(N)
    if len(w):
        order = np.lexsort((-w, child))
        ch = child[order]
        first = np.r_[True, ch[1:] != ch[:-1]]
        best[ch[first]] = parent[order][first]
        bw[ch[first]] = w[order][first]
    return best, bw


def verify_probabilities(w_bg, child_sums, min_w, n_acausal, n_nonfinite):
    """Checks required by the thesis methodology (5.4)."""
    s = w_bg + child_sums
    res = {
        "all_finite": bool(n_nonfinite == 0 and np.isfinite(w_bg).all()),
        "non_negative": bool(min_w >= 0 and (w_bg >= 0).all()),
        "max_abs_sum_minus_1": float(np.abs(s - 1.0).max()),
        "acausal_pairs": int(n_acausal),
    }
    res["sum_to_one"] = res["max_abs_sum_minus_1"] < 1e-8
    res["all_passed"] = bool(res["all_finite"] and res["non_negative"] and res["sum_to_one"] and n_acausal == 0)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# graph
# ═══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph(df, w_bg, out_strength, best_parent, best_w, edge_i, edge_j, edge_w):
    """Nodes = events. Edges = parent probabilities >= W_THRESHOLD (for graph analysis/plots);
    out_strength is computed from ALL probabilities, not only these edges."""
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_node(idx, time=str(row["datetime"]), latitude=float(row["latitude"]),
                   longitude=float(row["longitude"]), depth=float(row["depth"]), mag=float(row["mag"]),
                   event_id=str(row.get("id", f"evt_{idx}")),
                   is_background=float(w_bg[idx]), out_strength=float(out_strength[idx]),
                   best_parent=int(best_parent[idx]), best_parent_prob=float(best_w[idx]))
    for i, j, w in zip(edge_i, edge_j, edge_w):
        G.add_edge(int(i), int(j), weight=float(w), is_best_parent=int(best_parent[j] == i))
    log.info("Graph built: %d nodes | %d edges (threshold %.2f) | %d edges are a best-parent link",
             G.number_of_nodes(), G.number_of_edges(), W_THRESHOLD,
             sum(d["is_best_parent"] for _, _, d in G.edges(data=True)))
    return G


def export_graph(G, df, out_graphml, out_edge_csv, out_node_csv):
    nx.write_graphml(G, out_graphml)
    rows = [{"source_idx": i, "target_idx": j, "weight": d["weight"], "is_best_parent": d["is_best_parent"],
             "source_time": df.loc[i, "datetime"], "target_time": df.loc[j, "datetime"],
             "source_mag": df.loc[i, "mag"], "target_mag": df.loc[j, "mag"]} for i, j, d in G.edges(data=True)]
    pd.DataFrame(rows).to_csv(out_edge_csv, index=False)
    pd.DataFrame([{"idx": n, **d} for n, d in G.nodes(data=True)]).to_csv(out_node_csv, index=False)
    log.info("Saved %s, %s, %s", out_graphml, out_edge_csv, out_node_csv)


def export_pairs(path, N, parent, child, w, w_bg, cutoff=PAIR_EXPORT_CUTOFF):
    """Estimated probabilities for later analysis (forest realisations, sensitivity)."""
    m = w >= cutoff
    np.savez_compressed(path, N=N, cutoff=cutoff, parent=parent[m].astype(np.int32),
                        child=child[m].astype(np.int32), w=w[m].astype(np.float32),
                        w_bg=np.asarray(w_bg, dtype=np.float32))
    log.info("Pairs saved -> %s (%d pairs with w >= %g)", path, int(m.sum()), cutoff)


def write_run_json(path, info):
    info = dict(info)
    info["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                           "pandas": pd.__version__, "networkx": nx.__version__}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log.info("Run configuration and diagnostics saved -> %s", path)


def print_top_events(df, out_strength, w_bg, n=10):
    order = np.argsort(-out_strength)[:n]
    print(f"\nTop {n} events by weighted out-strength (sum of outgoing parent probabilities):")
    print(f"  {'Idx':>5}  {'Mag':>5}  {'Date':^22}  {'Out-strength':>12}  {'P(background)':>13}")
    for i in order:
        print(f"  {i:>5}  {df['mag'].iloc[i]:>5.1f}  {str(df['datetime'].iloc[i])[:19]:^22}  "
              f"{out_strength[i]:>12.2f}  {w_bg[i]:>13.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_subgraph(G, df, title_prefix, out_png, seed_idx=None, hops=VIZ_HOP_RADIUS, max_nodes=VIZ_MAX_NODES):
    """Ego-network around the event with the largest out-strength among the largest-magnitude events.
    Blue edges = most probable parent of the target; orange = other candidate parent.
    Node colour = P(background). Layout = force-directed, seeded from geography."""
    if seed_idx is None:
        cand = [n for n in G.nodes() if G.out_degree(n) > 0 and df.loc[n, "mag"] >= VIZ_MIN_MAG] or \
               [n for n in G.nodes() if G.out_degree(n) > 0] or list(G.nodes())
        seed_idx = max(cand, key=lambda n: df.loc[n, "mag"])
    ego, frontier = {seed_idx}, {seed_idx}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt |= set(G.successors(n)) | set(G.predecessors(n))
        ego |= nxt
        frontier = nxt
        if len(ego) > max_nodes * 4:
            break
    if len(ego) > max_nodes:
        ranked = sorted(ego, key=lambda n: df.loc[n, "mag"], reverse=True)
        ego = set(ranked[:max_nodes]) | {seed_idx}
    sub = G.subgraph(ego).copy()
    n = sub.number_of_nodes()
    geo = {k: (df.loc[k, "longitude"], df.loc[k, "latitude"]) for k in sub.nodes()}
    pos = nx.spring_layout(sub, pos=geo, k=0.5 / max(n, 1) ** 0.5, iterations=150, seed=42)

    sizes = [22 * (df.loc[k, "mag"] ** 2) for k in sub.nodes()]
    ecol = ["#2979ff" if d["is_best_parent"] else "#ff6d00" for _, _, d in sub.edges(data=True)]
    ewid = [1.8 * d["weight"] + 0.2 for _, _, d in sub.edges(data=True)]
    cmap = plt.get_cmap("coolwarm")
    ncol = [cmap(min(max(G.nodes[k]["is_background"], 0.0), 1.0)) for k in sub.nodes()]

    fig, ax = plt.subplots(figsize=(max(12, min(26, 8 + n / 15)), max(10, min(22, 7 + n / 18))))
    ax.set_facecolor("#0d0d0d"); fig.patch.set_facecolor("#0d0d0d")
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=ncol, ax=ax, alpha=0.85)
    nx.draw_networkx_edges(sub, pos, edge_color=ecol, width=ewid, ax=ax, alpha=0.55, arrows=True,
                           arrowsize=9, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_labels(sub, pos, labels={k: f"M{df.loc[k, 'mag']:.1f}" for k in sub.nodes()
                                              if df.loc[k, "mag"] >= VIZ_MIN_MAG},
                            font_color="white", font_size=8, ax=ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="#2979ff", lw=2, label="Most probable parent of the target"),
                       Line2D([0], [0], color="#ff6d00", lw=2, label="Other candidate parent")],
              loc="upper left", facecolor="#1a1a1a", labelcolor="white", framealpha=0.8)
    ax.set_title(f"{title_prefix} family tree — ego-network around event {seed_idx} "
                 f"(M{df.loc[seed_idx, 'mag']:.1f}, {str(df.loc[seed_idx, 'datetime'])[:10]})\n"
                 f"{n} nodes shown | edges: P(parent) >= {W_THRESHOLD} | node colour: P(background) "
                 f"(red = background, blue = triggered)", color="white", fontsize=10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Subgraph plot saved -> %s", out_png)
