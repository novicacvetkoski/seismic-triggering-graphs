"""
parse_jma.py
------------
Parses ALL JMA unified hypocenter "deck" format .txt files (the labeled
LAT=/LONG=/DEPTH=/MAG1= variant, e.g. d202301a.txt / d202301b.txt / d202301c.txt)
found in a folder into ONE merged CSV, matching (as closely as reasonable) the
schema used by merge_earthquakes.py so it can eventually be folded into the
USGS-based merged catalogue.

Usage
-----
  python parse_jma.py                              # reads ./jma_data/*.txt -> ./jma_earthquakes_enriched.csv
  python parse_jma.py --input-dir /path/to/folder
  python parse_jma.py --input-dir data --output combined.csv --pattern "*.txt"

Just dump every JMA .txt file you have into one folder (any filenames,
any number of files, any months/years) and point --input-dir at it. Every
event from every file is parsed, tagged with its source file, concatenated,
de-duplicated (in case adjacent monthly files overlap at the boundary),
sorted by time, and written out as a single CSV.

Handles format quirks NOT present in a plain "every event looks the same" deck:

  * Normal events        "              2023Y  1M  1D ..."
  * Mechanism events      "MF            2023Y  1M  1D ..."   -> real event, kept
  * Poor-quality events   "POOR SOLUTION 2023Y  1M  1D ..."   -> real event, kept
                                                                   (often missing MAG1)
  * External reference    "AFTER USGS    2023Y  1M  2D ..."   -> a USGS-reported
                            teleseism shown for context only, NOT a JMA-determined
                            location -> EXCLUDED
  * TEMPLATE blocks        a line literally reading "TEMPLATE" marks that the very
                            next header+LAT block is a *historical* reference event
                            (used for waveform-correlation matching against MF
                            mechanism solutions), not a new event for this day
                            -> EXCLUDED

STATION phase-pick lines are intentionally not parsed into event fields.
"""

import os
import re
import glob
import argparse
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = "jma_data"
DEFAULT_OUTPUT     = "jma_earthquakes_enriched.csv"
DEFAULT_PATTERN    = "*.txt"

# Rounding used to detect duplicate events across overlapping files
# (e.g. a month-boundary event appearing in both the previous and next file)
DEDUP_TIME_ROUND = "1min"
DEDUP_LATLON_ROUND = 3

HEADER_RE = re.compile(
    r"(\d{4})Y\s+(\d+)M\s+(\d+)D\s+(\d+)H\s+(\d+)M\s+(\d+\.\d+)S"
    r"(?:\s*\+/-([\d.]+))?"
    r"\s+(.*?)\s+R=\((\d+),\s*(\d*)\)"
)

LAT_RE   = re.compile(r"LAT=\s*(\d+)\s+(\d+\.\d+)([NS])(?:\s*\+/-([\d.]+))?")
LON_RE   = re.compile(r"LONG=\s*(\d+)\s+(\d+\.\d+)([EW])(?:\s*\+/-([\d.]+))?")
DEPTH_RE = re.compile(r"DEPTH=\s*(\d+)KM(?:\s*\+/-([\d.]+))?")
MAG_RE   = re.compile(r"MAG1=([-\d.]+)([A-Za-z]?)")


def classify_header(line: str):
    """Return (event_type, header_text) based on the line's leading tag."""
    stripped = line.lstrip()
    if stripped.startswith("MF"):
        return "mechanism", stripped[2:]
    if stripped.startswith("POOR SOLUTION"):
        return "poor_solution", stripped[len("POOR SOLUTION"):]
    if stripped.startswith("AFTER USGS"):
        return "after_usgs_reference", stripped[len("AFTER USGS"):]
    return "normal", stripped


def parse_file(path: str):
    events = []
    n_lines_total = 0
    n_header_lines = 0
    n_lat_lines = 0
    n_skipped_template = 0
    n_skipped_after_usgs = 0
    n_header_parse_fail = 0
    n_lat_parse_fail = 0

    current_event = None
    template_pending = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            n_lines_total += 1
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            # ---------------------------------------------------------
            # TEMPLATE marker: the next header+LAT block is a historical
            # reference event, not a new event for this file.
            # ---------------------------------------------------------
            if stripped == "TEMPLATE":
                template_pending = True
                continue

            # ---------------------------------------------------------
            # HEADER LINE
            # ---------------------------------------------------------
            m = HEADER_RE.search(line)
            if m and "LAT=" not in line:
                n_header_lines += 1
                event_type, _ = classify_header(line)

                skip = False
                if event_type == "after_usgs_reference":
                    skip = True
                    n_skipped_after_usgs += 1
                if template_pending:
                    skip = True
                    event_type = "template_reference"
                    n_skipped_template += 1
                template_pending = False  # consumed by this header

                (year, month, day, hour, minute, second,
                 ot_unc, region, n_p, n_tot) = m.groups()

                current_event = {
                    "_skip": skip,
                    "event_type": event_type,
                    "year": int(year), "month": int(month), "day": int(day),
                    "hour": int(hour), "minute": int(minute), "second": float(second),
                    "origin_time_uncertainty_sec": float(ot_unc) if ot_unc else None,
                    "region": region.strip() if region else None,
                    "num_p_stations": int(n_p) if n_p else None,
                    "num_total_stations": int(n_tot) if n_tot.strip() else None,
                }
                continue

            # ---------------------------------------------------------
            # LOCATION / MAGNITUDE LINE
            # ---------------------------------------------------------
            if "LAT=" in line and "LONG=" in line:
                n_lat_lines += 1

                if current_event is None:
                    # LAT line with no preceding header we recognised -> drop
                    n_lat_parse_fail += 1
                    continue

                try:
                    lat_m = LAT_RE.search(line)
                    lon_m = LON_RE.search(line)
                    dep_m = DEPTH_RE.search(line)
                    mag_m = MAG_RE.search(line)

                    if not (lat_m and lon_m):
                        raise ValueError("LAT/LON not matched")

                    lat_deg, lat_min, lat_dir, lat_unc = lat_m.groups()
                    latitude = int(lat_deg) + float(lat_min) / 60
                    if lat_dir == "S":
                        latitude *= -1

                    lon_deg, lon_min, lon_dir, lon_unc = lon_m.groups()
                    longitude = int(lon_deg) + float(lon_min) / 60
                    if lon_dir == "W":
                        longitude *= -1

                    current_event["latitude"] = latitude
                    current_event["longitude"] = longitude
                    current_event["latitude_uncertainty_km"] = float(lat_unc) if lat_unc else None
                    current_event["longitude_uncertainty_km"] = float(lon_unc) if lon_unc else None

                    if dep_m:
                        current_event["depth_km"] = int(dep_m.group(1))
                        current_event["depth_uncertainty_km"] = (
                            float(dep_m.group(2)) if dep_m.group(2) else None
                        )
                    else:
                        current_event["depth_km"] = None
                        current_event["depth_uncertainty_km"] = None

                    if mag_m:
                        current_event["magnitude"] = float(mag_m.group(1))
                        current_event["magnitude_type"] = mag_m.group(2) or None
                    else:
                        current_event["magnitude"] = None
                        current_event["magnitude_type"] = None

                    ts = pd.Timestamp(
                        year=current_event["year"], month=current_event["month"],
                        day=current_event["day"], hour=current_event["hour"],
                        minute=current_event["minute"],
                        second=int(current_event["second"]),
                        microsecond=int(round((current_event["second"] % 1) * 1_000_000)),
                    )
                    current_event["timestamp_jst"] = ts
                    current_event["timestamp_utc"] = ts - pd.Timedelta(hours=9)

                    if not current_event["_skip"]:
                        current_event["source_file"] = os.path.basename(path)
                        events.append({k: v for k, v in current_event.items() if k != "_skip"})

                except Exception:
                    n_lat_parse_fail += 1
                finally:
                    current_event = None

    stats = dict(
        file=path, n_lines_total=n_lines_total, n_header_lines=n_header_lines,
        n_lat_lines=n_lat_lines, n_events_kept=len(events),
        n_skipped_template=n_skipped_template, n_skipped_after_usgs=n_skipped_after_usgs,
        n_header_parse_fail=n_header_parse_fail, n_lat_parse_fail=n_lat_parse_fail,
    )
    return events, stats


def parse_args():
    p = argparse.ArgumentParser(description="Parse a folder of JMA deck-format .txt files into one CSV.")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                    help=f"Folder containing JMA .txt files (default: {DEFAULT_INPUT_DIR})")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                    help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"Glob pattern for input files within --input-dir (default: {DEFAULT_PATTERN})")
    p.add_argument("--no-dedup", action="store_true",
                    help="Skip cross-file de-duplication (keep every parsed event as-is)")
    return p.parse_args()


def dedup_events(df: pd.DataFrame):
    """Drop duplicate events that appear in more than one input file
    (e.g. overlapping month-boundary coverage). Keeps the first occurrence."""
    t = pd.to_datetime(df["timestamp_utc"]).dt.round(DEDUP_TIME_ROUND)
    lat = df["latitude"].round(DEDUP_LATLON_ROUND)
    lon = df["longitude"].round(DEDUP_LATLON_ROUND)
    key = t.astype(str) + "|" + lat.astype(str) + "|" + lon.astype(str)
    before = len(df)
    df = df.loc[~key.duplicated()].copy()
    removed = before - len(df)
    return df, removed


def main():
    args = parse_args()

    search_path = os.path.join(args.input_dir, args.pattern)
    files = sorted(glob.glob(search_path))

    if not files:
        log.error("No files found matching %s — check --input-dir / --pattern.", search_path)
        return

    log.info("Found %d input file(s) in %s", len(files), args.input_dir)

    all_events = []
    all_stats = []

    for path in files:
        events, stats = parse_file(path)
        all_events.extend(events)
        all_stats.append(stats)
        log.info(
            "%-20s headers=%-5d lat_lines=%-5d kept=%-5d "
            "skipped(template=%d, after_usgs=%d) parse_fail(header=%d, lat=%d)",
            os.path.basename(stats["file"]), stats["n_header_lines"], stats["n_lat_lines"],
            stats["n_events_kept"], stats["n_skipped_template"], stats["n_skipped_after_usgs"],
            stats["n_header_parse_fail"], stats["n_lat_parse_fail"],
        )

    if not all_events:
        log.error("No events parsed from any file.")
        return

    df = pd.DataFrame(all_events)

    columns = [
        "timestamp_jst", "timestamp_utc", "region", "event_type",
        "latitude", "longitude", "depth_km",
        "magnitude", "magnitude_type",
        "origin_time_uncertainty_sec", "latitude_uncertainty_km",
        "longitude_uncertainty_km", "depth_uncertainty_km",
        "num_p_stations", "num_total_stations", "source_file",
    ]
    df = df[columns].sort_values("timestamp_utc").reset_index(drop=True)

    n_removed_dupes = 0
    if not args.no_dedup:
        df, n_removed_dupes = dedup_events(df)
        df = df.reset_index(drop=True)

    df.to_csv(args.output, index=False)

    print("\n" + "=" * 70)
    print("JMA PARSE SUMMARY")
    print("=" * 70)
    stats_df = pd.DataFrame(all_stats)
    stats_df["file"] = stats_df["file"].apply(os.path.basename)
    print(stats_df.to_string(index=False))
    print(f"\nFiles processed          : {len(files)}")
    print(f"Cross-file duplicates removed : {n_removed_dupes:,}"
          + ("  (--no-dedup was set, so none removed)" if args.no_dedup else ""))
    print(f"Total events kept        : {len(df):,}")
    print(f"Output                   : {args.output}")
    print("\nBy event_type:")
    print(df["event_type"].value_counts().to_string())
    print(f"\nEvents missing magnitude : {df['magnitude'].isna().sum():,}")
    print(f"Time range               : {df['timestamp_utc'].min()}  ->  {df['timestamp_utc'].max()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
