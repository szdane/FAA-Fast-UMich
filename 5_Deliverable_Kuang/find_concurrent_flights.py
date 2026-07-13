"""
find_concurrent_flights.py
--------------------------
Identifies flights in filtered_rows.csv that enter the airspace within a
configurable time window of each other.

Entry time = first recTime record for each acId.

Outputs
-------
- Console: grouped clusters of concurrent flights (sorted by entry time)
- CSV:     concurrent_flights.csv  (one row per flight in a multi-flight cluster)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Parameters ──────────────────────────────────────────────────────────────
CSV_PATH       = Path(__file__).parent / "Input" / "filtered_rows.csv"
OUTPUT_CSV     = Path(__file__).parent / "Output" / "concurrent_flights.csv"
WINDOW_MINUTES = 15     # two flights are "concurrent" if entry times are within this window
MIN_CLUSTER    = 2      # only report clusters with at least this many flights
# ────────────────────────────────────────────────────────────────────────────


def find_clusters(entry_series: pd.Series, window_minutes: int) -> pd.Series:
    """
    Assign a cluster ID to each flight using a sliding-window sweep.
    Flights are sorted by entry time; a new cluster starts whenever the gap
    from the cluster's first entry exceeds window_minutes.

    Parameters
    ----------
    entry_series : pd.Series  index=acId, values=entry datetime
    window_minutes : int

    Returns
    -------
    pd.Series  index=acId, values=cluster_id (int, 0-based)
    """
    sorted_entries = entry_series.sort_values()
    cluster_ids    = pd.Series(index=sorted_entries.index, dtype=int)
    cluster_id     = 0
    cluster_start  = sorted_entries.iloc[0]
    window         = pd.Timedelta(minutes=window_minutes)

    for acId, t in sorted_entries.items():
        if t - cluster_start > window:
            cluster_id   += 1
            cluster_start = t
        cluster_ids[acId] = cluster_id

    return cluster_ids


def main():
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df["recTime"] = pd.to_datetime(df["recTime"], format="mixed", errors="raise")

    # ── Entry time = first record per flight ──────────────────────────────
    entry_times = df.groupby("acId")["recTime"].min().sort_values()
    print(f"  {len(entry_times):,} unique flights  |  "
          f"{entry_times.min().strftime('%Y-%m-%d %H:%M')} → "
          f"{entry_times.max().strftime('%Y-%m-%d %H:%M')}")

    # ── Cluster by time window ────────────────────────────────────────────
    cluster_ids = find_clusters(entry_times, WINDOW_MINUTES)

    # Build summary table
    result = (
        entry_times
        .rename("entry_time")
        .to_frame()
        .assign(cluster_id=cluster_ids)
    )
    result["entry_time_str"] = result["entry_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cluster_sizes = result.groupby("cluster_id").size().rename("n_flights")
    result        = result.join(cluster_sizes, on="cluster_id")

    # Keep only clusters with >= MIN_CLUSTER flights
    concurrent = (
        result[result["n_flights"] >= MIN_CLUSTER]
        .sort_values(["cluster_id", "entry_time"])
        .reset_index()
        .rename(columns={"index": "acId"})
    )

    n_clusters = concurrent["cluster_id"].nunique()
    print(f"\n  Window: {WINDOW_MINUTES} min  |  "
          f"Min cluster size: {MIN_CLUSTER}  |  "
          f"Concurrent clusters found: {n_clusters}  |  "
          f"Flights in clusters: {len(concurrent)}")

    # ── Print clusters ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"{'Cluster':>8}  {'Flights':>6}  {'Window start':>19}  {'Window end':>19}")
    print(f"{'─'*70}")

    for cid, grp in concurrent.groupby("cluster_id"):
        t_start = grp["entry_time"].min().strftime("%Y-%m-%d %H:%M:%S")
        t_end   = grp["entry_time"].max().strftime("%Y-%m-%d %H:%M:%S")
        span    = (grp["entry_time"].max() - grp["entry_time"].min()).total_seconds() / 60
        print(f"\n  Cluster {cid:>4}  ({len(grp)} flights,  span {span:.1f} min)")
        for _, row in grp.iterrows():
            print(f"    {row['entry_time_str']}   {row['acId']}")

    print(f"\n{'─'*70}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_cols = ["cluster_id", "n_flights", "acId", "entry_time_str"]
    concurrent[out_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
