"""
Run pairwise Active Corners checks for multiple aircraft, but only until each
aircraft reaches its STAR fix.

This imports the segment-level ACM helper from:

    run_acm_two_aircraft.py
"""

import math
import itertools
import pandas as pd
import sys, os
sys.path.append(os.path.abspath("../2.1_MILP/2.1.2_MILP"))

from run_acm_two_aircraft import acm_check_segment


FT_PER_NM = 6076.12
FT_PER_DEG_LAT = 60.0 * FT_PER_NM


STAR_FIXES = {
    "BONZZ": (41.7483, -82.7972, (21000, 15000)),
    "CRAKN": (41.6730, -82.9405, (26000, 12000)),
    "CUUGR": (42.3643, -83.0975, (11000, 10000)),
    "FERRL": (42.4165, -82.6093, (10000, 8000)),
    "GRAYT": (42.9150, -83.6020, (22000, 17000)),
    "HANBL": (41.7375, -84.1773, (21000, 17000)),
    "HAYLL": (41.9662, -84.2975, (11000, 11000)),
    "HTROD": (42.0278, -83.3442, (12000, 12000)),
    "KKISS": (42.5443, -83.7620, (15000, 12000)),
    "KLYNK": (41.8793, -82.9888, (10000, 9000)),
    "LAYKS": (42.8532, -83.5498, (10000, 10000)),
    "LECTR": (41.9183, -84.0217, (10000, 8000)),
    "RKCTY": (42.6869, -83.9603, (13000, 11000)),
    "VCTRZ": (41.9878, -84.0670, (15000, 12000)),
}


def detect_aircraft_ids(df):
    aircraft = []

    for col in df.columns:
        if col.endswith("_lat"):
            aircraft_id = col[:-4]
            required = {
                f"{aircraft_id}_lat",
                f"{aircraft_id}_lon",
                f"{aircraft_id}_alt_ft",
            }
            if required.issubset(df.columns):
                aircraft.append(aircraft_id)

    def sort_key(name):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else name

    return sorted(aircraft, key=sort_key)


def latlon_to_local_ft_multi(df, aircraft_ids):
    all_lats = pd.concat([df[f"{aid}_lat"] for aid in aircraft_ids])

    lat0 = all_lats.mean()

    ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(lat0))

    for aid in aircraft_ids:
        df[f"{aid}_north_ft"] = (
            (df[f"{aid}_lat"] - lat0) * FT_PER_DEG_LAT
        )

        df[f"{aid}_east_ft"] = (
            (df[f"{aid}_lon"] - df[f"{aid}_lon"].mean())
            * ft_per_deg_lon
        )

    return df


def dist_latlon_ft(lat1, lon1, lat2, lon2):
    lat_mid = math.radians((lat1 + lat2) / 2.0)

    dlat_ft = (lat1 - lat2) * FT_PER_DEG_LAT

    dlon_ft = (
        (lon1 - lon2)
        * FT_PER_DEG_LAT
        * math.cos(lat_mid)
    )

    return math.hypot(dlat_ft, dlon_ft)


def altitude_in_range(alt_ft, alt_range, tol_ft=1000):
    lo = min(alt_range)
    hi = max(alt_range)

    return (lo - tol_ft) <= alt_ft <= (hi + tol_ft)


def find_star_entry_info(df, aircraft_id):

    lat_col = f"{aircraft_id}_lat"
    lon_col = f"{aircraft_id}_lon"
    alt_col = f"{aircraft_id}_alt_ft"

    for idx, row in df.iterrows():

        lat = row[lat_col]
        lon = row[lon_col]
        alt = row[alt_col]

        for star_name, (fix_lat, fix_lon, alt_range) in STAR_FIXES.items():

            d = dist_latlon_ft(
                lat,
                lon,
                fix_lat,
                fix_lon,
            )

            if (
                d <= 1000
                and altitude_in_range(alt, alt_range)
            ):
                return {
                    "entry_index": idx,
                    "entry_time": row["t"],
                    "star_fix": star_name,
                }

    # fallback:
    # detect first constant suffix

    final_lat = df[lat_col].iloc[-1]
    final_lon = df[lon_col].iloc[-1]
    final_alt = df[alt_col].iloc[-1]

    for idx in range(len(df)):

        suffix = df.iloc[idx:]

        same_lat = (
            suffix[lat_col] - final_lat
        ).abs().max() < 1e-10

        same_lon = (
            suffix[lon_col] - final_lon
        ).abs().max() < 1e-10

        same_alt = (
            suffix[alt_col] - final_alt
        ).abs().max() < 1e-6

        if same_lat and same_lon and same_alt:

            return {
                "entry_index": idx,
                "entry_time": df.loc[idx, "t"],
                "star_fix": "UNKNOWN_CONSTANT_SUFFIX",
            }

    return {
        "entry_index": len(df) - 1,
        "entry_time": df.iloc[-1]["t"],
        "star_fix": "UNKNOWN",
    }


def make_relative_columns(df, ownship, intruder):

    prefix = f"rel_{ownship}_minus_{intruder}"

    df[f"{prefix}_north_ft"] = (
        df[f"{ownship}_north_ft"]
        - df[f"{intruder}_north_ft"]
    )

    df[f"{prefix}_east_ft"] = (
        df[f"{ownship}_east_ft"]
        - df[f"{intruder}_east_ft"]
    )

    df[f"{prefix}_alt_ft"] = (
        df[f"{ownship}_alt_ft"]
        - df[f"{intruder}_alt_ft"]
    )

    return {
        "lat_lon": (
            f"{prefix}_north_ft",
            f"{prefix}_east_ft",
            500.0,
            500.0,
        ),

        "lat_alt": (
            f"{prefix}_north_ft",
            f"{prefix}_alt_ft",
            500.0,
            100.0,
        ),

        "lon_alt": (
            f"{prefix}_east_ft",
            f"{prefix}_alt_ft",
            500.0,
            100.0,
        ),
    }


def row_is_active(row, aircraft_id):

    needed = [
        f"{aircraft_id}_lat",
        f"{aircraft_id}_lon",
        f"{aircraft_id}_alt_ft",
    ]

    return all(pd.notna(row[col]) for col in needed)


def segment_pair_is_active_until_star(
    df,
    k,
    a1,
    a2,
    entry_info,
):

    if not (
        row_is_active(df.loc[k], a1)
        and row_is_active(df.loc[k + 1], a1)
        and row_is_active(df.loc[k], a2)
        and row_is_active(df.loc[k + 1], a2)
    ):
        return False

    e1 = entry_info[a1]["entry_index"]
    e2 = entry_info[a2]["entry_index"]

    if (k + 1) > e1:
        return False

    if (k + 1) > e2:
        return False

    return True


def run_acm_multi_until_star(csv_path):

    df = pd.read_csv(csv_path)

    aircraft_ids = detect_aircraft_ids(df)

    df = latlon_to_local_ft_multi(df, aircraft_ids)

    entry_info = {
        aid: find_star_entry_info(df, aid)
        for aid in aircraft_ids
    }

    results = []

    for a1, a2 in itertools.combinations(aircraft_ids, 2):

        planes = make_relative_columns(df, a1, a2)

        for plane_name, (
            xcol,
            ycol,
            half_x,
            half_y,
        ) in planes.items():

            for k in range(len(df) - 1):

                if not segment_pair_is_active_until_star(
                    df,
                    k,
                    a1,
                    a2,
                    entry_info,
                ):
                    continue

                p0 = (
                    df.loc[k, xcol],
                    df.loc[k, ycol],
                )

                p1 = (
                    df.loc[k + 1, xcol],
                    df.loc[k + 1, ycol],
                )

                try:

                    unsafe, mode, _ = acm_check_segment(
                        p0=p0,
                        p1=p1,
                        half_x=half_x,
                        half_y=half_y,
                    )

                    status = "UNSAFE" if unsafe else "SAFE"

                except Exception as e:

                    status = "ERROR"
                    mode = ""
                    print(e)

                results.append({
                    "aircraft_1": a1,
                    "aircraft_2": a2,
                    "plane": plane_name,
                    "segment": k,
                    "t_start": df.loc[k, "t"],
                    "t_end": df.loc[k + 1, "t"],
                    "status": status,
                    "mode": mode,
                    "a1_star_fix": entry_info[a1]["star_fix"],
                    "a2_star_fix": entry_info[a2]["star_fix"],
                })

    return pd.DataFrame(results)


if __name__ == "__main__":

    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "../2.3_Outputs_and_Results/test.csv"
    )

    results = run_acm_multi_until_star(csv_path)

    print(results.to_string(index=False))

    conflicts = results[
        results["status"] == "UNSAFE"
    ]
    REQUIRED_PLANES = {"lat_lon", "lat_alt", "lon_alt"}

    conflicts_3d = (
            conflicts
            .groupby(["aircraft_1", "aircraft_2", "segment", "t_start", "t_end"])
            .filter(lambda g: REQUIRED_PLANES.issubset(set(g["plane"])))
        )

    if not conflicts_3d.empty:
        print("\n=== Confirmed 3D conflicts ===")
        print(conflicts_3d.to_string(index=False))
    else:
        print("\nSAFE_3D: No confirmed 3D conflicts.")