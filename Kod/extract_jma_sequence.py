"""
extract_jma_sequence.py
------------------------
Extracts a spatiotemporal subset (one earthquake sequence) from the large
merged JMA hypocenter catalogue CSV and writes it out in the column schema
that seismic_kg_misd.py / seismic_kg_etas.py already expect:

    time, latitude, longitude, depth, mag, magType, id, source_file
    (plus region / event_type kept for reference / QA)

This means you generally do NOT need to touch the loader in
seismic_kg_misd.py -- just point CATALOGUE_PATH at the file this script
produces for each sequence.

Why chunked reading
--------------------
Your merged JMA CSV is ~700MB. Rather than pd.read_csv() the whole thing
into memory, this script streams it in chunks (default 500k rows/chunk),
filters each chunk, and only accumulates the rows that match.

Usage
-----
  # Built-in windows for the three thesis case studies
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv --preset tohoku
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv --preset kumamoto
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv --preset noto

  # Fully custom window
  python extract_jma_sequence.py --source jma_earthquakes_enriched.csv \\
      --start 2016-04-01 --end 2016-08-31 \\
      --minlat 32.6 --maxlat 33.0 --minlon 130.6 --maxlon 131.0 \\
      --out kumamoto_2016.csv

Notes
-----
* Filters on timestamp_utc (UTC), matching how the old pipeline's 'time'
  column was interpreted.
* By default keeps only event_type == 'normal' and drops poor_solution /
  reference / template rows. Use --keep-flagged to disable.
* magType is carried through as-is from magnitude_type ('V' = velocity-
  type, 'D' = displacement-type JMA magnitude, etc.) -- these are native
  JMA-scale magnitudes, not Mw. The summary printed at the end breaks
  down magnitude_type counts for the extracted window so you can confirm
  each sequence is on a single homogeneous scale before running
  MISD/ETAS (the triggering kernel assumes one scale throughout).
* MAX_EVENTS in seismic_kg_misd.py / seismic_kg_etas.py may need
  attention after extraction -- see the printed event count. The MISD
  script's rate-matrix computation is O(N^2) in a Python loop; a raw
  swarm extract (e.g. Noto, tens of thousands of events at low Mc) will
  be impractically slow unless you raise the completeness magnitude
  (MAG_MIN) for that run or filter further before feeding it in.
"""

import argparse
import sys
import pandas as pd

CHUNK_SIZE = 500_000

# Convenience presets for the three thesis case studies.
# Dates are UTC. Bounding boxes are generous rectangles around each sequence
# -- tighten them once you've eyeballed the extracted event map.
JMA_SEQUENCES = {
    "tohoku": dict(
        start="2011-03-01", end="2011-12-31",
        minlat=35.0, maxlat=41.0, minlon=140.0, maxlon=145.0,
    ),
    "kumamoto": dict(
        start="2016-04-01", end="2016-08-31",
        minlat=32.6, maxlat=33.0, minlon=130.6, maxlon=131.0,
    ),
    "noto": dict(
        start="2020-11-01", end="2023-12-31",
        minlat=37.2, maxlat=37.7, minlon=136.6, maxlon=137.4,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Extract a JMA earthquake sequence subset.")
    p.add_argument("--source", default="jma_earthquakes_enriched.csv",
                    help="Path to the full merged JMA CSV (default: %(default)s)")
    p.add_argument("--preset", choices=JMA_SEQUENCES.keys(),
                    help="Use a built-in date/bbox window for one of the thesis case studies")
    p.add_argument("--start", help="Start of window, e.g. 2016-04-01 (UTC). Overrides preset if given.")
    p.add_argument("--end", help="End of window, e.g. 2016-08-31 (UTC). Overrides preset if given.")
    p.add_argument("--minlat", type=float)
    p.add_argument("--maxlat", type=float)
    p.add_argument("--minlon", type=float)
    p.add_argument("--maxlon", type=float)
    p.add_argument("--minmag", type=float, default=None,
                    help="Optional minimum magnitude filter (applied on top of date/bbox)")
    p.add_argument("--keep-flagged", action="store_true",
                    help="Keep poor_solution / non-'normal' event_type rows (default: drop them)")
    p.add_argument("--out", help="Output CSV path (default derived from preset name)")
    p.add_argument("--chunksize", type=int, default=CHUNK_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    if args.preset:
        cfg = JMA_SEQUENCES[args.preset].copy()
        # allow CLI overrides on top of a preset
        for key in ("start", "end", "minlat", "maxlat", "minlon", "maxlon"):
            val = getattr(args, key)
            if val is not None:
                cfg[key] = val
        out_path = args.out or f"jma_{args.preset}.csv"
    else:
        required = ["start", "end", "minlat", "maxlat", "minlon", "maxlon"]
        missing = [r for r in required if getattr(args, r) is None]
        if missing:
            sys.exit(f"Missing required arguments (no --preset given): {missing}")
        cfg = {k: getattr(args, k) for k in required}
        out_path = args.out or "jma_custom_extract.csv"

    start_ts = pd.Timestamp(cfg["start"], tz="UTC")
    end_ts   = pd.Timestamp(cfg["end"],   tz="UTC")

    print(f"Source             : {args.source}")
    print(f"Window             : {start_ts} -> {end_ts}")
    print(f"Bounding box       : lat[{cfg['minlat']}, {cfg['maxlat']}]  lon[{cfg['minlon']}, {cfg['maxlon']}]")
    if args.minmag is not None:
        print(f"Min magnitude      : {args.minmag}")
    print(f"Drop flagged rows  : {not args.keep_flagged}")
    print()

    kept_frames = []
    total_rows_seen = 0

    reader = pd.read_csv(args.source, chunksize=args.chunksize, low_memory=False)
    for i, chunk in enumerate(reader):
        total_rows_seen += len(chunk)

        chunk["_ts_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
        mask = (
            chunk["_ts_utc"].between(start_ts, end_ts)
            & chunk["latitude"].between(cfg["minlat"], cfg["maxlat"])
            & chunk["longitude"].between(cfg["minlon"], cfg["maxlon"])
        )
        if args.minmag is not None:
            mask &= chunk["magnitude"] >= args.minmag
        if not args.keep_flagged:
            mask &= chunk["event_type"].astype(str).str.lower() == "normal"

        sub = chunk.loc[mask].copy()
        if len(sub):
            kept_frames.append(sub)

        if (i + 1) % 20 == 0:
            kept_so_far = sum(len(f) for f in kept_frames)
            print(f"  ...scanned {total_rows_seen:,} rows, kept {kept_so_far:,} so far")

    if not kept_frames:
        sys.exit("No events matched the given window -- check dates/bbox against the source file.")

    result = pd.concat(kept_frames, ignore_index=True)
    result.sort_values("_ts_utc", inplace=True)
    result.reset_index(drop=True, inplace=True)

    # ── remap to the schema seismic_kg_misd.py / seismic_kg_etas.py expect ──
    out = pd.DataFrame()
    out["time"]        = result["_ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out["latitude"]    = result["latitude"]
    out["longitude"]   = result["longitude"]
    out["depth"]       = result["depth_km"]
    out["mag"]         = result["magnitude"]
    out["magType"]     = result["magnitude_type"]
    out["id"]          = [f"jma_{idx}" for idx in out.index]
    out["region"]      = result["region"]
    out["event_type"]  = result["event_type"]
    out["source_file"] = result["source_file"]

    out.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total rows scanned : {total_rows_seen:,}")
    print(f"Events extracted   : {len(out):,}")
    print(f"Output file        : {out_path}")
    print(f"Time range         : {out['time'].min()} -> {out['time'].max()}")
    print(f"Mag range          : {out['mag'].min():.1f} - {out['mag'].max():.1f}")
    print(f"\nmagnitude_type breakdown (check for scale homogeneity):")
    for mt, cnt in result["magnitude_type"].value_counts(dropna=False).items():
        print(f"  {str(mt):>6}: {cnt:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
