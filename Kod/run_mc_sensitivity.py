"""
run_mc_sensitivity.py - sensitivity of the results to the completeness threshold (mentor's point on 3.3 / 6.4).
For every sequence the threshold is raised (default +0.5 magnitude units), the background rate is RE-ESTIMATED from the
pre-sequence catalogue at the new threshold (robust = median monthly count, as in estimate_background_rate.py; if that
median is 0 the mean rate N/(T A) is used and a warning is printed), and both models are run on the thinner catalogue.

  python run_mc_sensitivity.py --pre-dir ../Earthquake-series --dry-run      # print rates and commands only
  python run_mc_sensitivity.py --pre-dir ../Earthquake-series                # run (ETAS Tohoku is the slow one)

Pre-sequence files: <pre-dir>/jma_<seq>_pre.csv  (output of extract_jma_sequence.py, see estimate_background_rate.py).
Run prefixes: <seq>_mc<Mc>  (e.g. kumamoto_mc2.5).  Event ids are the same as in the primary runs (jma_<row>), so
the events can be matched across runs.
"""
import argparse, os, subprocess, sys, time
import numpy as np
import pandas as pd
import estimate_background_rate as B
import kg_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--seqs", default="noto,kumamoto,tohoku")
ap.add_argument("--delta", type=float, default=0.5, help="threshold increase relative to the primary Mc")
ap.add_argument("--pre-dir", default=C.DATA_DIR)
ap.add_argument("--out-dir", default="results")
ap.add_argument("--models", default="misd,etas")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()


def rate(seq, mc):
    P = B.PRESETS[seq]
    df = pd.read_csv(os.path.join(a.pre_dir, f"jma_{seq}_pre.csv"), usecols=["time", "latitude", "longitude", "mag"], low_memory=False)
    df["t"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    start, end = pd.Timestamp(P["start"], tz="UTC"), pd.Timestamp(P["end"], tz="UTC")
    b = P["box"]
    df = df.dropna(subset=["t", "mag"])
    df = df[(df.t >= start) & (df.t <= end) & (df.mag >= mc) & df.latitude.between(b[0], b[1]) & df.longitude.between(b[2], b[3])]
    T = (end - start).total_seconds() / 86400.0
    A = B.box_area_km2(b)
    n_m = int(round(T / B.MONTH_DAYS))
    counts = np.histogram((df.t - start).dt.total_seconds().values / 86400.0, bins=np.linspace(0, T, n_m + 1))[0]
    med = float(np.median(counts))
    robust, mean = med / (B.MONTH_DAYS * A), len(df) / (T * A)
    return (robust if robust > 0 else mean), robust, mean, len(df), med


jobs = []
for seq in a.seqs.split(","):
    mc0 = C.SEQUENCE_PRESETS[seq]["mag_min"]; mc = round(mc0 + a.delta, 2)
    r, robust, mean, n, med = rate(seq, mc)
    print(f"{seq}: Mc {mc0} -> {mc}: pre-sequence events {n}, median monthly count {med:.0f}, robust {robust:.3e}, mean {mean:.3e}"
          + ("   [robust = 0 -> mean rate used]" if robust <= 0 else ""))
    for model in a.models.split(","):
        jobs.append([sys.executable, f"seismic_kg_{model}.py", "--preset", seq, "--mag-min", str(mc), "--bg-rate", f"{r:.4e}",
                     "--out-dir", a.out_dir, "--out-prefix", f"{seq}_mc{mc}"])
for k, cmd in enumerate(jobs, 1):
    print(f"[{k}/{len(jobs)}] " + " ".join(cmd[1:]))
    if not a.dry_run:
        t0 = time.time(); rc = subprocess.run(cmd).returncode; print(f"    -> exit {rc}, {time.time() - t0:.0f} s")
