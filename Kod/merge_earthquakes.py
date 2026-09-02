"""
merge_earthquakes.py
--------------------
Merges three Japan earthquake data sources into a single consistent CSV:
  1. Japan_earthquakes_2001_-_2018.csv          (USGS format, 22 cols)
  2. Kaggle_2.zip  (multiple USGS-format CSVs inside)
  3. Table_S1_HomogeneousCatalog_Mw.xlsx        (homogeneous Mw catalog, different schema)

Output: merged_japan_earthquakes.csv

Deduplication strategy:
  - USGS records: deduplicated on the 'id' column (authoritative event ID).
  - XLSX records: no USGS id — deduplicated on rounded (time, lat, lon, depth) key,
    then cross-checked against the USGS pool to avoid double-counting the same event.

Column schema of the output (superset of all sources):
  time, latitude, longitude, depth, mag, magType,
  nst, gap, dmin, rms, net, id, updated, place, type,
  horizontalError, depthError, magError, magNst, status,
  locationSource, magSource, Mw, Mw_Std, source_file
"""

import os
import io
import zipfile
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── paths (edit if running from a different working directory) ─────────────────
CSV_PATH  = "Japan_earthquakes_2001_-_2018.csv"
ZIP_PATH  = "Kaggle_2.zip"
XLSX_PATH = "Table_S1_HomogeneousCatalog_Mw.xlsx"
OUT_PATH  = "merged_japan_earthquakes.csv"

# Round-to for fuzzy deduplication between USGS and XLSX events
LAT_LON_ROUND = 2   # ~1 km precision
DEPTH_ROUND   = 1   # km
TIME_ROUND    = "1min"  # pandas rounding freq


# ── helpers ───────────────────────────────────────────────────────────────────

def read_usgs_csv(path_or_bytes, source_label: str) -> pd.DataFrame:
    """Read a USGS-format CSV and tag with source_file."""
    df = pd.read_csv(path_or_bytes, low_memory=False)
    df["source_file"] = source_label
    return df


def build_datetime_from_xlsx(df: pd.DataFrame) -> pd.Series:
    """Construct ISO datetime strings from the XLSX Year/Month/Date/Hr/Min/Sec columns."""
    def _to_dt(row):
        try:
            sec_int = int(row["Sec"])
            microsec = int(round((row["Sec"] - sec_int) * 1_000_000))
            return pd.Timestamp(
                year=int(row["Year"]),
                month=int(row["Month"]),
                day=int(row["Date"]),
                hour=int(row["Hr"]),
                minute=int(row["Min"]),
                second=sec_int,
                microsecond=microsec,
                tz="UTC",
            ).isoformat()
        except Exception:
            return pd.NaT

    return df.apply(_to_dt, axis=1)


def make_dedup_key(df: pd.DataFrame) -> pd.Series:
    """
    Create a string deduplication key from (time_rounded, lat, lon, depth).
    Used to cross-check XLSX records against the USGS pool.
    """
    t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.round(TIME_ROUND)
    lat = df["latitude"].round(LAT_LON_ROUND)
    lon = df["longitude"].round(LAT_LON_ROUND)
    dep = df["depth"].round(DEPTH_ROUND)
    return t.astype(str) + "|" + lat.astype(str) + "|" + lon.astype(str) + "|" + dep.astype(str)


# ── 1. Load USGS CSVs ─────────────────────────────────────────────────────────

usgs_frames: list[pd.DataFrame] = []

# 1a. Standalone CSV
if os.path.exists(CSV_PATH):
    df = read_usgs_csv(CSV_PATH, os.path.basename(CSV_PATH))
    log.info("Loaded standalone CSV: %d rows", len(df))
    usgs_frames.append(df)
else:
    log.warning("Standalone CSV not found: %s", CSV_PATH)

# 1b. ZIP contents
if os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            data = io.BytesIO(zf.read(name))
            df = read_usgs_csv(data, os.path.basename(name))
            log.info("  Loaded ZIP entry %-45s  %d rows", name, len(df))
            usgs_frames.append(df)
else:
    log.warning("ZIP not found: %s", ZIP_PATH)

# Combine all USGS frames
usgs_all = pd.concat(usgs_frames, ignore_index=True)
log.info("Combined USGS rows (before dedup): %d", len(usgs_all))

# Deduplicate on USGS event id (authoritative)
usgs_dedup = usgs_all.drop_duplicates(subset=["id"])
log.info("Combined USGS rows (after id dedup): %d  (removed %d duplicates)",
         len(usgs_dedup), len(usgs_all) - len(usgs_dedup))


# ── 2. Load XLSX ──────────────────────────────────────────────────────────────

if os.path.exists(XLSX_PATH):
    xlsx_raw = pd.read_excel(XLSX_PATH)
    log.info("Loaded XLSX: %d rows", len(xlsx_raw))

    # Build normalised DataFrame matching the USGS schema
    xlsx_norm = pd.DataFrame()
    xlsx_norm["time"]      = build_datetime_from_xlsx(xlsx_raw)
    xlsx_norm["latitude"]  = xlsx_raw["Latitude"]
    xlsx_norm["longitude"] = xlsx_raw["Longitude"]
    xlsx_norm["depth"]     = xlsx_raw["Depth"]
    xlsx_norm["mag"]       = xlsx_raw["Mw"]          # Mw as primary magnitude
    xlsx_norm["magType"]   = "Mw"
    xlsx_norm["Mw"]        = xlsx_raw["Mw"]           # keep dedicated Mw column too
    xlsx_norm["Mw_Std"]    = xlsx_raw["Mw_Std"]
    xlsx_norm["source_file"] = os.path.basename(XLSX_PATH)

    # Remaining USGS columns are not available; leave as NaN (they will be added
    # automatically when we concat due to the superset schema below)

    # Cross-deduplicate: remove XLSX rows already covered by a USGS record
    usgs_key_set = set(make_dedup_key(usgs_dedup))
    xlsx_key     = make_dedup_key(xlsx_norm)
    in_usgs_mask = xlsx_key.isin(usgs_key_set)
    log.info("XLSX rows already present in USGS pool (removed): %d", in_usgs_mask.sum())
    xlsx_unique = xlsx_norm[~in_usgs_mask].copy()
    log.info("XLSX rows added uniquely: %d", len(xlsx_unique))
else:
    log.warning("XLSX not found: %s", XLSX_PATH)
    xlsx_unique = pd.DataFrame()


# ── 3. Merge all sources ──────────────────────────────────────────────────────

merged = pd.concat([usgs_dedup, xlsx_unique], ignore_index=True)
log.info("Total merged rows: %d", len(merged))

# Sort by time ascending
merged["time_dt"] = pd.to_datetime(merged["time"], utc=True, errors="coerce")
merged.sort_values("time_dt", inplace=True)
merged.drop(columns=["time_dt"], inplace=True)
merged.reset_index(drop=True, inplace=True)

# ── 4. Save ───────────────────────────────────────────────────────────────────

merged.to_csv(OUT_PATH, index=False)
log.info("Saved → %s  (%d rows × %d cols)", OUT_PATH, *merged.shape)

# ── 5. Summary ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MERGE SUMMARY")
print("="*60)
print(f"Output file  : {OUT_PATH}")
print(f"Total rows   : {len(merged):,}")
print(f"Columns      : {list(merged.columns)}")
print(f"\nRows by source:")
for src, cnt in merged["source_file"].value_counts().items():
    print(f"  {src:<50} {cnt:>8,}")
print(f"\nTime range   : {merged['time'].min()}  →  {merged['time'].max()}")
print(f"Mag range    : {merged['mag'].min():.1f} – {merged['mag'].max():.1f}")
print("="*60)
