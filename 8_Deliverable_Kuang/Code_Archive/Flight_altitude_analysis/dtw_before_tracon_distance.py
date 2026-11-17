"""Distance to DTW 20 minutes before the project's modeled TRACON entry.

Run without arguments to save a per-flight CSV and histogram and show the
plot. Inputs are Input/filtered_rows.csv and Input/entry_exit_points.csv.
The latter's exit_rectime is EXIT from pre-TRACON, hence ENTRY to TRACON,
as defined in Functions/fuel_emission_analysis_main.py. This STAR-fix-based
model is not a verified FAA operational or altitude-dependent boundary.

Both input timestamps use the same local clock; no BTS arrival data is used.
Each supplied entry event must fall inside a matching fltKey trajectory and
agree with its interpolated position within 2 NM. Ambiguous events, missing
coverage, conflicting samples, gaps over 60 seconds, and interpolation speeds
over 700 knots are not used. No extrapolation or last-point substitution.
Distance is horizontal great-circle distance to DTW, not to the boundary.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtw_18000ft_distance import NM_TO_METERS, SPHERE
from dtw_one_hour_distance import position_at


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Input"
OUTPUT_DIR = SCRIPT_DIR / "Output" / "DTW_before_TRACON"
MINUTES_BEFORE_ENTRY = 20
MAX_ENTRY_POSITION_ERROR_NM = 2.0


def analyze_flight(track, events):
    first = track.iloc[0]
    result = {
        "fltKey": first["fltKey"], "acId": first["acId"],
        "status": "no_supplied_tracon_entry",
        "tracon_entry_time": pd.NaT, "target_time": pd.NaT,
        "minutes_before_entry": MINUTES_BEFORE_ENTRY,
        "entry_position_error_nm": np.nan,
        "track_start": track["recTime"].min(), "track_end": track["recTime"].max(),
        "distance_nm": np.nan, "latitude": np.nan, "longitude": np.nan,
        "bracket_seconds": np.nan,
    }
    samples = ["recTime", "coord1", "coord2"]
    track = track.sort_values("recTime").drop_duplicates(samples)
    if (track[samples].isna().any().any()
            or not np.isfinite(track[["coord1", "coord2"]]).all().all()
            or not track["coord1"].between(-90, 90).all()
            or not track["coord2"].between(-180, 180).all()):
        result["status"] = "invalid_track_sample"
        return result
    if track["recTime"].duplicated().any():
        result["status"] = "conflicting_timestamp_samples"
        return result
    candidates = events.loc[events["acId"].eq(first["acId"])].drop_duplicates(
        ["exit_rectime", "exit_lat", "exit_lon"]
    )
    if candidates.empty:
        return result
    candidates = candidates.loc[candidates["exit_rectime"].between(
        track["recTime"].min(), track["recTime"].max()
    )]
    if len(candidates) != 1:
        result["status"] = "ambiguous_entry_event" if len(candidates) else "entry_outside_this_track"
        return result
    event = candidates.iloc[0]
    target = event["exit_rectime"] - pd.Timedelta(minutes=MINUTES_BEFORE_ENTRY)
    result.update(tracon_entry_time=event["exit_rectime"], target_time=target)
    entry = position_at(track, event["exit_rectime"])
    if entry["status"] != "included":
        result["status"] = "entry_" + entry["status"]
        return result
    if not (pd.notna(event["exit_lat"]) and pd.notna(event["exit_lon"])
            and -90 <= event["exit_lat"] <= 90 and -180 <= event["exit_lon"] <= 180):
        result["status"] = "invalid_entry_coordinates"
        return result
    _, _, error_meters = SPHERE.inv(
        entry["longitude"], entry["latitude"], event["exit_lon"], event["exit_lat"]
    )
    result["entry_position_error_nm"] = error_meters / NM_TO_METERS
    if result["entry_position_error_nm"] > MAX_ENTRY_POSITION_ERROR_NM:
        result["status"] = "entry_position_mismatch"
        return result
    result.update(position_at(track, target))
    return result


def load_results():
    columns = ["fltKey", "acId", "arrival", "recTime", "coord1", "coord2"]
    chunks = pd.read_csv(INPUT_DIR / "filtered_rows.csv", usecols=columns, chunksize=100000)
    tracks = pd.concat([chunk.loc[chunk["arrival"].eq("KDTW")] for chunk in chunks])
    if tracks.empty:
        raise ValueError("No KDTW trajectories found.")
    tracks["recTime"] = pd.to_datetime(tracks["recTime"], format="mixed", errors="coerce")
    for column in ["coord1", "coord2"]:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")
    events = pd.read_csv(INPUT_DIR / "entry_exit_points.csv")
    events["exit_rectime"] = pd.to_datetime(events["exit_rectime"], format="mixed", errors="coerce")
    for column in ["exit_lat", "exit_lon"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    return pd.DataFrame([analyze_flight(track, events) for _, track in tracks.groupby("fltKey")])


def main(show_plot=True):
    results = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "dtw_20min_before_tracon.csv"
    results.to_csv(csv_path, index=False)
    distances = results.loc[results["status"].eq("included"), "distance_nm"]
    print("Distance to DTW 20 minutes before modeled TRACON entry (exit_rectime).")
    print(results["status"].value_counts().to_string())
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    axis.set_title("DTW: 20 Minutes Before TRACON Entry", loc="left", fontsize=18, pad=15)
    axis.set_xlabel("Horizontal great-circle distance to DTW (nautical miles)")
    axis.set_ylabel("Number of flights")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    if not distances.empty:
        print(distances.describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())
        bins = np.arange(0, max(20, np.ceil(distances.max() / 20) * 20) + 20, 20)
        axis.hist(distances, bins=bins, color="#267a78", edgecolor="white")
        axis.axvline(distances.median(), color="#c24b38", label=f"Median {distances.median():.1f} NM")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No supported observations", ha="center", transform=axis.transAxes)
    figure.supxlabel(
        f"{len(distances)} observations / {len(results)} DTW flight keys\n"
        "Entry from supplied pre-TRACON exit timestamps; project STAR-fix boundary.\n"
        "No extrapolation. Not gate arrival, touchdown, or distance to the boundary.", fontsize=10
    )
    png_path = OUTPUT_DIR / "dtw_20min_before_tracon_histogram.png"
    figure.savefig(png_path, dpi=180)
    print(f"\nResults: {csv_path}\nHistogram: {png_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == "__main__":
    main()