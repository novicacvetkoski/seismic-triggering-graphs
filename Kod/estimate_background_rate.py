"""
estimate_background_rate.py
---------------------------
Estimates the long-term background rate of a sequence's study box from a
PRE-sequence catalogue extract, in the units MISD needs
(events / day / km^2), and prints ready-to-run MISD commands.

Why: Marsan & Lengline (2010) impose the background rate rather than estimate
it, because for a flexible triggering kernel the background and the kernel's
long-lag / long-distance tails are not separately identifiable. A rate measured
in the quiet years BEFORE the sequence, in the same box and above the same
completeness magnitude, is an estimate that does not depend on the kernel
model. This is what decides which events are roots (background) of the
family trees.

Step 1 - extract the pre-sequence window (same box, same Mc) from the full JMA file:

  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv \\
      --start 2009-03-01 --end 2011-03-01 \\
      --minlat 35.0 --maxlat 41.0 --minlon 140.0 --maxlon 145.0 --minmag 4.0 --out jma_tohoku_pre.csv
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv \\
      --start 2013-04-01 --end 2016-04-01 \\
      --minlat 32.6 --maxlat 33.0 --minlon 130.6 --maxlon 131.0 --minmag 2.0 --out jma_kumamoto_pre.csv
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv \\
      --start 2017-11-01 --end 2020-11-01 \\
      --minlat 37.2 --maxlat 37.7 --minlon 136.6 --maxlon 137.4 --minmag 2.0 --out jma_noto_pre.csv

Step 2 - estimate the rate:

  python estimate_background_rate.py --preset tohoku   --pre-csv jma_tohoku_pre.csv
  python estimate_background_rate.py --preset kumamoto --pre-csv jma_kumamoto_pre.csv
  python estimate_background_rate.py --preset noto     --pre-csv jma_noto_pre.csv

Two estimates are reported:
  * mean rate   = N / (T * A)                      -- includes any aftershock bursts in
                                                      the window, so it is an UPPER bound;
  * robust rate = median monthly count / (30.44 d * A) -- insensitive to a few burst months.
Run MISD at the robust rate and use mean x1 / robust x0.5 / robust x2 as the
sensitivity analysis (as in the paper).
"""
import argparse
import numpy as np
import pandas as pd

KM_PER_DEG = 6371.0 * np.pi / 180.0          # same Earth radius as seismic_kg_misd.py

# box = (minlat, maxlat, minlon, maxlon) exactly as used for the sequence extraction;
# window = the sequence window (for the implied-background sanity check)
PRESETS = {
    "tohoku":   dict(box=(35.0, 41.0, 140.0, 145.0), mag_min=4.0,
                     start="2009-03-01", end="2011-03-01", seq=("2011-03-01", "2011-12-31")),
    "kumamoto": dict(box=(32.6, 33.0, 130.6, 131.0), mag_min=2.0,
                     start="2013-04-01", end="2016-04-01", seq=("2016-04-01", "2016-08-31")),
    "noto":     dict(box=(37.2, 37.7, 136.6, 137.4), mag_min=2.0,
                     start="2017-11-01", end="2020-11-01", seq=("2020-11-01", "2023-12-31")),
}
MONTH_DAYS = 30.4375


def box_area_km2(box):
    la0, la1, lo0, lo1 = box
    return (la1 - la0) * KM_PER_DEG * (lo1 - lo0) * KM_PER_DEG * np.cos(np.radians(0.5 * (la0 + la1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS.keys(), required=True)
    ap.add_argument("--pre-csv", required=True, help="pre-sequence extract (output of extract_jma_sequence.py)")
    ap.add_argument("--start"), ap.add_argument("--end")
    ap.add_argument("--box", nargs=4, type=float, metavar=("MINLAT", "MAXLAT", "MINLON", "MAXLON"))
    ap.add_argument("--mag-min", type=float)
    a = ap.parse_args()

    P = PRESETS[a.preset]
    box = tuple(a.box) if a.box else P["box"]
    mc = a.mag_min if a.mag_min is not None else P["mag_min"]
    start = pd.Timestamp(a.start or P["start"], tz="UTC")
    end = pd.Timestamp(a.end or P["end"], tz="UTC")
    T = (end - start).total_seconds() / 86400.0
    A = box_area_km2(box)

    df = pd.read_csv(a.pre_csv, usecols=["time", "latitude", "longitude", "mag"], low_memory=False)
    df["t"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["t", "mag"])
    df = df[(df.t >= start) & (df.t <= end) & (df.mag >= mc) &
            df.latitude.between(box[0], box[1]) & df.longitude.between(box[2], box[3])]
    N = len(df)
    if N == 0:
        raise SystemExit("No events in the pre-sequence window - check the file, window and box.")

    # monthly counts over the window
    n_m = int(round(T / MONTH_DAYS))
    days = (df.t - start).dt.total_seconds().values / 86400.0
    counts = np.histogram(days, bins=np.linspace(0, T, n_m + 1))[0]
    med = float(np.median(counts))
    bursts = [(i, int(c)) for i, c in enumerate(counts) if c > max(3 * med, med + 5)]

    rate_mean = N / (T * A)
    rate_robust = med / (MONTH_DAYS * A)

    # completeness sanity checks
    b = np.log10(np.e) / (df.mag.mean() - (mc - 0.05))          # Aki / Utsu, 0.1-unit bins
    yearly = df.groupby(df.t.dt.year).size()

    t0, t1 = pd.Timestamp(P["seq"][0], tz="UTC"), pd.Timestamp(P["seq"][1], tz="UTC")
    T_seq = (t1 - t0).total_seconds() / 86400.0

    print("=" * 66)
    print(f"Background rate for {a.preset}   (box {box}, M >= {mc})")
    print("=" * 66)
    print(f"Pre-sequence window : {start.date()} -> {end.date()}  ({T:.0f} d, {n_m} months)")
    print(f"Box area            : {A:,.0f} km^2")
    print(f"Events (M>={mc})    : {N:,}    b-value (Aki) = {b:.2f}   [typical 0.8-1.2; much lower => incomplete near Mc]")
    print(f"Events per year     : " + ", ".join(f"{y}: {c}" for y, c in yearly.items())
          + "   [a sudden jump/drop => catalogue change or a sequence in the window]")
    print(f"Monthly count       : median {med:.0f}, min {counts.min()}, max {counts.max()}"
          + (f";  burst months (index from start): {bursts}" if bursts else ";  no burst months"))
    print()
    print(f"mean rate   = {rate_mean:.3e} events/day/km^2   (upper bound: includes aftershocks)")
    print(f"robust rate = {rate_robust:.3e} events/day/km^2   (median month; recommended)")
    print()
    print(f"Implied background events during the sequence window ({T_seq:.0f} d, whole box):")
    print(f"   robust: {rate_robust * T_seq * A:,.0f}    mean: {rate_mean * T_seq * A:,.0f}")
    print()
    print("Run MISD with the imposed background (run from Kod/):")
    for tag, r in [("robust", rate_robust), ("mean", rate_mean),
                   ("robust_x0.5", 0.5 * rate_robust), ("robust_x2", 2 * rate_robust)]:
        print(f"  python seismic_kg_misd.py --preset {a.preset} --background fixed "
              f"--bg-rate {r:.4e} --out-prefix {a.preset}_bg_{tag}")
    print()
    print("Paper-faithful full-histogram sensitivity check (same imposed background):")
    print(f"  python seismic_kg_misd.py --preset {a.preset} --kernel full --background fixed "
          f"--bg-rate {rate_robust:.4e} --out-prefix {a.preset}_full_bg_robust")
    print("=" * 66)


if __name__ == "__main__":
    main()
