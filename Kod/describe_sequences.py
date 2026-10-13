"""
describe_sequences.py  --  Chapter 3 of the thesis (data description, completeness, figures)
=============================================================================================
Reads the three extracted JMA sequence CSVs (columns: time, latitude, longitude, depth, mag,
magType, id, region, event_type) and produces, for the thesis:

  figures/fig_karta.pdf       epicentre maps (one panel per sequence)
  figures/fig_vreme.pdf       cumulative number of events + magnitude-time plot
  figures/fig_magnituda.pdf   frequency-magnitude distribution with Mc and b-value
  figures/fig_mc_vreme.pdf    Mc(t) in time windows (short-term incompleteness check)
  podatoci_makroa.tex         LaTeX macros + table rows consumed by poglavje3_podatoci.tex
  podatoci_sumarno.csv        the same numbers as a plain table

Completeness magnitude Mc (decision for the thesis):
  * MAXC (maximum curvature of the non-cumulative frequency-magnitude distribution)
    + 0.2 correction (Woessner & Wiemer 2005), as the main estimate;
  * goodness-of-fit test GFT (Wiemer & Wyss 2000): smallest Mc with 90 % / 95 % fit;
  * b-value by Aki-Utsu maximum likelihood with the Bender (1983) bin correction;
  * the Mc actually used in the pipeline (kg_common.SEQUENCE_PRESETS) is compared with these.

Usage (from the folder that holds the CSVs, or pass --data-dir):
  python describe_sequences.py --data-dir "../Earthquake-series"
Dependencies: numpy pandas matplotlib
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEQ = {   # name -> (file, Mc used in the pipeline, label)
    "tohoku":   ("jma_tohoku.csv",   4.0, "Tōhoku 2011"),
    "kumamoto": ("jma_kumamoto.csv", 2.0, "Kumamoto 2016"),
    "noto":     ("jma_noto.csv",     2.0, "Noto 2020–2023"),
}
DM = 0.1          # JMA magnitudes are reported to 0.1


def load(path):
    df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)
    df["datetime"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime", "latitude", "longitude", "depth", "mag"])
    n_core = len(df)
    n_nomt = int(df["magType"].isna().sum()) if "magType" in df.columns else 0
    if "magType" in df.columns:
        df = df.dropna(subset=["magType"])
    df = df.sort_values("datetime").reset_index(drop=True)
    dup = int(df.duplicated(subset=["time", "latitude", "longitude", "mag"]).sum())
    return df, dict(n_raw=n_raw, n_dropped_core=n_raw - n_core, n_dropped_magtype=n_nomt, n_dup=dup)


def fmd(mags, mmin=None):
    m = np.round(np.asarray(mags) / DM) * DM
    lo = np.floor(m.min() / DM) * DM if mmin is None else mmin
    edges = np.arange(lo, m.max() + 1.5 * DM, DM)
    centres = np.round(edges[:-1], 2)
    counts = np.array([(np.abs(m - c) < DM / 2).sum() for c in centres])
    return centres, counts


def b_value(mags, mc):
    """Aki-Utsu MLE with Bender correction for DM binning; returns b, its standard error, n."""
    x = np.asarray(mags)[np.asarray(mags) >= mc - 1e-9]
    n = len(x)
    if n < 20:
        return np.nan, np.nan, n
    b = np.log10(np.e) / (x.mean() - (mc - DM / 2))
    return b, b / np.sqrt(n), n


def maxc(mags):
    c, k = fmd(mags)
    return c[np.argmax(k)]


def gft(mags, thresholds=(0.90, 0.95)):
    """Wiemer & Wyss (2000): smallest Mc for which the fitted G-R law explains >= 90/95 % of the data."""
    c, k = fmd(mags)
    res = {t: np.nan for t in thresholds}
    for mc in c:
        sel = np.asarray(mags) >= mc - 1e-9
        if sel.sum() < 50:
            break
        b, _, _ = b_value(mags, mc)
        a = np.log10(sel.sum()) + b * (mc - DM / 2)
        cc = c[c >= mc - 1e-9]
        obs = k[c >= mc - 1e-9].astype(float)
        # fitted non-cumulative counts from a Gutenberg-Richter law with this b and a
        pred = 10 ** (a - b * (cc - DM / 2)) - 10 ** (a - b * (cc + DM / 2))
        R = 100 - 100 * np.abs(obs - pred).sum() / obs.sum()
        for t in thresholds:
            if np.isnan(res[t]) and R >= 100 * t:
                res[t] = mc
    return res


def mc_time(df, nwin=6):
    """Mc (MAXC+0.2) in log-spaced windows after the start of the sequence (short-term incompleteness)."""
    t = (df["datetime"] - df["datetime"].min()).dt.total_seconds().values / 86400.0
    edges = np.unique(np.r_[0, np.logspace(-1, np.log10(t.max() + 1e-9), nwin)])
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (t >= a) & (t < b + (1e-9 if b == edges[-1] else 0))
        if s.sum() >= 100:
            out.append((np.sqrt(a * b) if a > 0 else b / 2, maxc(df["mag"].values[s]) + 0.2, int(s.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "figures"), exist_ok=True)

    D, M, rows = {}, {}, []
    for k, (f, mc_used, lab) in SEQ.items():
        df, info = load(os.path.join(a.data_dir, f))
        D[k] = df
        sub = df[df["mag"] >= mc_used]
        mx = maxc(df["mag"].values)
        g = gft(df["mag"].values)
        b, bse, nb = b_value(df["mag"].values, mc_used)
        mtc = df["magType"].astype(str).str.upper().value_counts()
        M[k] = dict(mx=mx, g=g, b=b, bse=bse, mcused=mc_used)
        rows.append(dict(
            sequence=k, label=lab,
            t_start=str(df["datetime"].min())[:10], t_end=str(df["datetime"].max())[:10],
            lat_min=df.latitude.min(), lat_max=df.latitude.max(), lon_min=df.longitude.min(), lon_max=df.longitude.max(),
            n_all=len(df), n_raw_rows=info["n_raw"], n_dropped_core=info["n_dropped_core"],
            n_dropped_magtype=info["n_dropped_magtype"], n_duplicates=info["n_dup"],
            mc_used=mc_used, n_used=len(sub), mmax=df["mag"].max(),
            mc_maxc=mx, mc_maxc_plus=mx + 0.2, mc_gft90=g[0.90], mc_gft95=g[0.95],
            b=b, b_se=bse, depth_median=df.depth.median(), depth_p95=df.depth.quantile(.95),
            magtype_V=int(mtc.get("V", 0)), magtype_D=int(mtc.get("D", 0)),
            magtype_other=int(len(df) - mtc.get("V", 0) - mtc.get("D", 0))))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out_dir, "podatoci_sumarno.csv"), index=False)

    # ---- figures ----------------------------------------------------------------------------
    cols = {"tohoku": "#d81b60", "kumamoto": "#1e88e5", "noto": "#43a047"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for i, (k, df) in enumerate(D.items()):
        s = df[df["mag"] >= SEQ[k][1]]
        ax[i].scatter(s.longitude, s.latitude, s=np.maximum(2, 3 * (s.mag - SEQ[k][1] + 1) ** 2.2) / 3,
                      c=s.depth, cmap="viridis_r", alpha=.6, lw=0)
        big = s[s.mag >= 6.0]
        ax[i].scatter(big.longitude, big.latitude, s=140, facecolors="none", edgecolors="red", lw=1.4)
        ax[i].set(title=f"{SEQ[k][2]}  (N = {len(s)}, M ≥ {SEQ[k][1]})", xlabel="longitude [°]", ylabel="latitude [°]")
        ax[i].set_aspect(1 / np.cos(np.radians(s.latitude.mean())))
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_karta.pdf")); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(15, 6.5))
    for i, (k, df) in enumerate(D.items()):
        s = df[df["mag"] >= SEQ[k][1]]
        t = (s.datetime - s.datetime.min()).dt.total_seconds() / 86400
        ax[0, i].plot(t, np.arange(1, len(s) + 1), color=cols[k])
        ax[0, i].set(title=SEQ[k][2], ylabel="cumulative number", xlabel="days since first event")
        ax[1, i].vlines(t, SEQ[k][1], s.mag, lw=.4, color="k", alpha=.5)
        ax[1, i].set(ylabel="magnitude", xlabel="days since first event", ylim=(SEQ[k][1], s.mag.max() + .3))
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_vreme.pdf")); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for i, (k, df) in enumerate(D.items()):
        c, n = fmd(df["mag"].values)
        cum = np.cumsum(n[::-1])[::-1]
        ax[i].semilogy(c, cum, "o", ms=3, color=cols[k], label="cumulative N(≥M)")
        ax[i].semilogy(c, np.maximum(n, .5), "s", ms=3, mfc="none", color="gray", label="non-cumulative")
        ax[i].axvline(SEQ[k][1], color="k", ls="--", label=f"Mc used = {SEQ[k][1]}")
        ax[i].axvline(M[k]["mx"] + .2, color="red", ls=":", label=f"MAXC+0.2 = {M[k]['mx'] + .2:.1f}")
        ax[i].set(title=f"{SEQ[k][2]}:  b = {M[k]['b']:.2f} ± {M[k]['bse']:.2f}", xlabel="magnitude", ylabel="number of events")
        ax[i].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_magnituda.pdf")); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for i, (k, df) in enumerate(D.items()):
        r = mc_time(df)
        if r:
            ax[i].semilogx([x[0] for x in r], [x[1] for x in r], "o-", color=cols[k])
        ax[i].axhline(SEQ[k][1], color="k", ls="--")
        ax[i].set(title=SEQ[k][2], xlabel="days since first event (window centre)", ylabel="Mc (MAXC + 0.2)")
    plt.tight_layout(); plt.savefig(os.path.join(a.out_dir, "figures", "fig_mc_vreme.pdf")); plt.close()

    # ---- LaTeX macros -----------------------------------------------------------------------
    def f(v, d=1):
        return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}".replace(".", "{,}")
    names = {"tohoku": "Tohoku", "kumamoto": "Kumamoto", "noto": "Noto"}
    L = ["% generated by describe_sequences.py -- do not edit by hand"]
    for _, r in S.iterrows():
        n = names[r.sequence]
        for key, val in [("TStart", r.t_start), ("TEnd", r.t_end), ("NAll", f"{r.n_all:,}".replace(",", "\\,")),
                         ("NRaw", f"{r.n_raw_rows:,}".replace(",", "\\,")), ("NUsed", f"{r.n_used:,}".replace(",", "\\,")),
                         ("McUsed", f(r.mc_used)), ("McMaxc", f(r.mc_maxc)), ("McMaxcPlus", f(r.mc_maxc_plus)),
                         ("McGftNinety", f(r.mc_gft90)), ("McGftNinetyFive", f(r.mc_gft95)),
                         ("Bval", f(r.b, 2)), ("BvalSE", f(r.b_se, 2)), ("Mmax", f(r.mmax)),
                         ("NDup", str(r.n_duplicates)), ("NDropMagType", str(r.n_dropped_magtype)),
                         ("NDropCore", str(r.n_dropped_core)), ("DepthMed", f(r.depth_median, 0)),
                         ("DepthPNinetyFive", f(r.depth_p95, 0)), ("NMagV", str(r.magtype_V)), ("NMagD", str(r.magtype_D)),
                         ("NMagOther", str(r.magtype_other))]:
            L.append(f"\\newcommand{{\\{key}{n}}}{{{val}}}")
    rowsT = []
    for _, r in S.iterrows():
        rowsT.append(f"{r.label} & {r.t_start}\\,--\\,{r.t_end} & "
                     f"{f(r.lat_min,1)}--{f(r.lat_max,1)} / {f(r.lon_min,1)}--{f(r.lon_max,1)} & "
                     f"{f(r.mc_used)} & {r.n_used:,} & {f(r.mmax)} \\\\".replace(",", "\\,", 1) if False else
                     f"{r.label} & {r.t_start}\\,--\\,{r.t_end} & {f(r.lat_min,1)}--{f(r.lat_max,1)} / "
                     f"{f(r.lon_min,1)}--{f(r.lon_max,1)} & {f(r.mc_used)} & {r.n_used} & {f(r.mmax)} \\\\")
    L.append("\\newcommand{\\TabelaSekvenciRedovi}{%\n" + "\n".join(rowsT) + "}")
    open(os.path.join(a.out_dir, "podatoci_makroa.tex"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(S.T.to_string())


if __name__ == "__main__":
    main()
