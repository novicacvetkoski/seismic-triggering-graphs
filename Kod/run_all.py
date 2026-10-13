"""
run_all.py - runs the complete set of model runs for the thesis, one after another.

  python run_all.py                       # everything: MISD + ETAS x 3 sequences x 4 background rates (+ MISD full-kernel check)
  python run_all.py --only-primary        # just the 6 main runs (robust background rate)
  python run_all.py --models etas --sequences tohoku --only-primary     # e.g. only the long ETAS Tohoku run
  python run_all.py --dry-run             # print the commands without running them

Run from the folder that contains the scripts (Kod/). Results go to ./results (see --out-dir).
Primary run prefix = sequence name; sensitivity runs: <seq>_bg_x0.5, <seq>_bg_x2, <seq>_bg_mean; MISD full-kernel check: <seq>_full.
Rates [events/day/km^2] come from estimate_background_rate.py (robust = median monthly count; mean = upper bound).
Approximate run times: MISD < 1 min each; ETAS ~2 min (Noto), ~3 min (Kumamoto), ~7-10 min (Tohoku).
"""
import argparse, subprocess, sys, time

RATES = {   # robust, mean
    "tohoku":   (1.2364e-06, 1.3169e-06),
    "kumamoto": (3.9515e-05, 4.8833e-05),
    "noto":     (8.3677e-06, 1.7196e-05),
}

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="misd,etas")
ap.add_argument("--sequences", default="noto,kumamoto,tohoku")
ap.add_argument("--only-primary", action="store_true")
ap.add_argument("--out-dir", default="results")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

jobs = []
for model in a.models.split(","):
    for seq in a.sequences.split(","):
        robust, mean = RATES[seq]
        variants = [("", robust)] if a.only_primary else [("", robust), ("_bg_x0.5", 0.5 * robust),
                                                          ("_bg_x2", 2 * robust), ("_bg_mean", mean)]
        for suffix, rate in variants:
            jobs.append([sys.executable, f"seismic_kg_{model}.py", "--preset", seq, "--out-dir", a.out_dir,
                         "--out-prefix", seq + suffix, "--bg-rate", f"{rate:.4e}"])
        if model == "misd" and not a.only_primary:
            jobs.append([sys.executable, "seismic_kg_misd.py", "--preset", seq, "--out-dir", a.out_dir,
                         "--out-prefix", seq + "_full", "--bg-rate", f"{robust:.4e}", "--kernel", "full"])

print(f"{len(jobs)} runs")
failed = []
for k, cmd in enumerate(jobs, 1):
    print(f"\n[{k}/{len(jobs)}] " + " ".join(cmd[1:]))
    if a.dry_run:
        continue
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    print(f"    -> exit {rc}, {time.time() - t0:.0f} s")
    if rc:
        failed.append(" ".join(cmd[1:]))
print("\nDONE." if not failed else "\nFAILED RUNS:\n  " + "\n  ".join(failed))
