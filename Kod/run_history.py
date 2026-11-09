"""
run_history.py - does the identity of the 'roots' depend on the observation window?  (mentor's point on 6.2)
Without earlier events as history the first retained event necessarily has P(background) = 1, and events early in the
window cannot have observed parents. This script prepends the PRE-SEQUENCE events (same box, same Mc) to the sequence
catalogue and runs both models on the longer catalogue; the imposed background rate is unchanged (it was estimated from
exactly that period). analyze_results.py then compares P(background) of the sequence events with the primary runs.

  python run_history.py --pre-dir ../Earthquake-series [--seqs noto,kumamoto,tohoku] [--dry-run]

Inputs : <data-dir>/jma_<seq>.csv (sequence) and <pre-dir>/jma_<seq>_pre.csv (pre-sequence, same schema).
Outputs: <data-dir>/jma_<seq>_hist.csv and results/<seq>_hist_kg_{etas,misd}_* (event ids: pre_<i> / original jma_<i>).
"""
import argparse, os, subprocess, sys, time
import pandas as pd
import kg_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", default="noto,kumamoto,tohoku")
ap.add_argument("--pre-dir", default=C.DATA_DIR)
ap.add_argument("--data-dir", default=C.DATA_DIR)
ap.add_argument("--out-dir", default="results")
ap.add_argument("--models", default="misd,etas")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

for seq in a.seqs.split(","):
    seq_csv = os.path.join(a.data_dir, f"jma_{seq}.csv")
    pre_csv = os.path.join(a.pre_dir, f"jma_{seq}_pre.csv")
    out_csv = os.path.join(a.data_dir, f"jma_{seq}_hist.csv")
    s, p = pd.read_csv(seq_csv, low_memory=False), pd.read_csv(pre_csv, low_memory=False)
    p["id"] = [f"pre_{i}" for i in range(len(p))]
    h = pd.concat([p, s], ignore_index=True)
    h["_t"] = pd.to_datetime(h["time"], utc=True, errors="coerce")
    h = h.sort_values("_t").drop(columns="_t")
    print(f"{seq}: {len(p)} pre-sequence rows + {len(s)} sequence rows -> {out_csv}")
    if not a.dry_run:
        h.to_csv(out_csv, index=False)
    for model in a.models.split(","):
        cmd = [sys.executable, f"seismic_kg_{model}.py", "--preset", seq, "--catalogue", out_csv, "--out-dir", a.out_dir,
               "--out-prefix", f"{seq}_hist"]
        print("  " + " ".join(cmd[1:]))
        if not a.dry_run:
            t0 = time.time(); rc = subprocess.run(cmd).returncode; print(f"    -> exit {rc}, {time.time() - t0:.0f} s")
