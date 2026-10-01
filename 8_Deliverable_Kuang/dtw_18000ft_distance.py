"""Run this file to plot DTW-arrival distance at a descending 18,000-ft crossing.

Uses Input/filtered_rows.csv and writes a per-flight CSV and histogram to
Output/DTW_18000ft_distance, then opens the plot. No command-line arguments
are required; paths are relative to this script, not the working directory.

DTW coordinates match main_v15.py. DTW is assumed at sea level; recorded
altitude (hundreds of feet) is used without a pressure correction. Distance
is horizontal great-circle distance on a mean-radius sphere, in nautical
miles, not slant range or remaining flown distance. Crossing time/position
assume linear altitude change and constant progress along the sample arc.

One result is used per fltKey: the final descending crossing. A level segment
at exactly 18,000 ft counts at its first descending arrival at that altitude.
If the track starts at 18,000 ft, departure below it also qualifies. A failed
quality check on the final crossing does not fall back to an earlier one.
Distances outside the 1.5-IQR fences are flagged, but remain in the histogram.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod


DTW_LAT0, DTW_LON0 = 42.2125, -83.3534
TARGET_ALTITUDE_FT = 18000.0
MAX_CROSSING_GAP_SECONDS = 60.0
MAX_SEGMENT_SPEED_KNOTS = 700.0
BIN_WIDTH_NM = 10.0
NM_TO_METERS = 1852.0
SPHERE = Geod(a=6371008.8, f=0)
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "Input" / "filtered_rows.csv"
OUTPUT_DIR = SCRIPT_DIR / "Output" / "DTW_18000ft_distance"
COLUMNS = ["fltKey", "acId", "arrival", "recTime", "alt", "coord1", "coord2"]


def analyze_flight(flight):
    """Return a distance or an explicit exclusion reason for one DTW flight."""
    result = {
        "fltKey": flight.iloc[0]["fltKey"],
        "acId": flight.iloc[0]["acId"],
        "arrival": "KDTW",
        "status": "no_descending_crossing",
        "records": len(flight),
        "duplicates_removed": 0,
        "descending_crossings": 0,
        "crossing_time": pd.NaT,
        "crossing_latitude": np.nan,
        "crossing_longitude": np.nan,
        "bracket_seconds": np.nan,
        "bracket_speed_knots": np.nan,
        "distance_nm": np.nan,
    }
    sample_columns = ["recTime", "alt", "coord1", "coord2"]
    flight = flight.sort_values("recTime").drop_duplicates(sample_columns)
    result["duplicates_removed"] = result["records"] - len(flight)
    if flight[sample_columns].isna().any().any():
        result["status"] = "missing_or_invalid_sample"
        return result
    if flight["recTime"].duplicated().any():
        result["status"] = "conflicting_timestamp_samples"
        return result
    if not (
        np.isfinite(flight[["alt", "coord1", "coord2"]]).all().all()
        and flight["coord1"].between(-90, 90).all()
        and flight["coord2"].between(-180, 180).all()
    ):
        result["status"] = "invalid_altitude_or_position"
        return result

    flight = flight.reset_index(drop=True)
    altitudes = flight["alt"].to_numpy() * 100.0
    candidates = []
    for index in range(len(flight) - 1):
        upper, lower = altitudes[index:index + 2]
        if not (upper >= TARGET_ALTITUDE_FT >= lower and upper > lower):
            continue
        if upper == TARGET_ALTITUDE_FT and candidates:
            previous_end = candidates[-1] + 1
            if np.all(altitudes[previous_end:index + 1] == TARGET_ALTITUDE_FT):
                continue
        candidates.append(index)
    result["descending_crossings"] = len(candidates)
    if not candidates:
        return result

    index = candidates[-1]
    before, after = flight.iloc[index], flight.iloc[index + 1]
    gap = (after["recTime"] - before["recTime"]).total_seconds()
    result["bracket_seconds"] = gap
    if gap > MAX_CROSSING_GAP_SECONDS:
        result["status"] = "crossing_gap_over_limit"
        return result
    azimuth, _, segment_meters = SPHERE.inv(
        before["coord2"], before["coord1"], after["coord2"], after["coord1"]
    )
    speed_knots = segment_meters / NM_TO_METERS / (gap / 3600.0)
    result["bracket_speed_knots"] = speed_knots
    if speed_knots > MAX_SEGMENT_SPEED_KNOTS:
        result["status"] = "implausible_crossing_segment_speed"
        return result
    fraction = (altitudes[index] - TARGET_ALTITUDE_FT) / (
        altitudes[index] - altitudes[index + 1]
    )
    longitude, latitude, _ = SPHERE.fwd(
        before["coord2"], before["coord1"], azimuth, fraction * segment_meters
    )
    _, _, distance_meters = SPHERE.inv(longitude, latitude, DTW_LON0, DTW_LAT0)
    result.update(
        status="included",
        crossing_time=before["recTime"] + pd.to_timedelta(fraction * gap, unit="s"),
        crossing_latitude=latitude,
        crossing_longitude=longitude,
        distance_nm=distance_meters / NM_TO_METERS,
    )
    return result


def load_results():
    """Read only required columns and analyze all flights destined for DTW."""
    arrivals = []
    for chunk in pd.read_csv(CSV_PATH, usecols=COLUMNS, chunksize=100000):
        arrivals.append(chunk.loc[chunk["arrival"].eq("KDTW")])
    flights = pd.concat(arrivals, ignore_index=True)
    if flights.empty:
        raise ValueError("No KDTW arrivals found in the input CSV.")
    flights["recTime"] = pd.to_datetime(flights["recTime"], format="mixed", errors="coerce")
    for column in ["alt", "coord1", "coord2"]:
        flights[column] = pd.to_numeric(flights[column], errors="coerce")
    results = pd.DataFrame([
        analyze_flight(flight) for _, flight in flights.groupby("fltKey", sort=True)
    ])
    distances = results.loc[results["status"].eq("included"), "distance_nm"]
    quartile_1, quartile_3 = distances.quantile([0.25, 0.75])
    spread = quartile_3 - quartile_1
    results["distance_outlier"] = (
        results["distance_nm"].lt(quartile_1 - 1.5 * spread)
        | results["distance_nm"].gt(quartile_3 + 1.5 * spread)
    )
    return results


def plot_histogram(results):
    """Make a count histogram with one observation per included flight."""
    distances = results.loc[results["status"].eq("included"), "distance_nm"]
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    figure.set_facecolor("#fafafa")
    axis.set_facecolor("#fafafa")
    axis.set_title("DTW Arrivals at 18,000 ft", loc="left", fontsize=19, pad=16)
    axis.set_xlabel("Horizontal great-circle distance to DTW (nautical miles)")
    axis.set_ylabel("Number of flights")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    if distances.empty:
        axis.text(0.5, 0.5, "No eligible descending crossings", ha="center",
                  transform=axis.transAxes)
        return figure
    bin_end = max(BIN_WIDTH_NM, np.ceil(distances.max() / BIN_WIDTH_NM) * BIN_WIDTH_NM)
    bins = np.arange(0, bin_end + BIN_WIDTH_NM, BIN_WIDTH_NM)
    axis.hist(distances, bins=bins, color="#267a78", edgecolor="white", linewidth=1)
    axis.axvline(distances.median(), color="#c24b38", linewidth=2,
                 label=f"Median: {distances.median():.1f} NM")
    outliers = results.loc[results["distance_outlier"], "distance_nm"]
    if not outliers.empty:
        axis.text(0.98, 0.8,
                  f"{len(outliers)} distance outlier(s) retained\n"
                  f"Maximum: {distances.max():.1f} NM",
                  ha="right", transform=axis.transAxes, fontsize=10)
    axis.legend(loc="upper right", frameon=False)
    percentile_10, percentile_90 = distances.quantile([0.1, 0.9])
    figure.supxlabel(
        f"{len(distances)} included / {len(results)} DTW flights | "
        f"Mean {distances.mean():.1f} NM | P10-P90 {percentile_10:.1f}-{percentile_90:.1f} NM\n"
        "Final recorded descending crossing per flight; DTW assumed at sea level.\n"
        "Distance to airport, not remaining flown distance or confirmed touchdown.",
        fontsize=10,
    )
    return figure


def main(show_plot=True):
    print(f"Reading {CSV_PATH}")
    results = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_output = OUTPUT_DIR / "dtw_18000ft_distances.csv"
    plot_output = OUTPUT_DIR / "dtw_18000ft_histogram.png"
    results.to_csv(csv_output, index=False)
    figure = plot_histogram(results)
    figure.savefig(plot_output, dpi=180)
    print(f"\nDTW: {DTW_LAT0}, {DTW_LON0}; assumed elevation 0 ft")
    print(f"Quality limits: crossing gap <= {MAX_CROSSING_GAP_SECONDS:g} s; "
          f"segment speed <= {MAX_SEGMENT_SPEED_KNOTS:g} knots")
    print("\nFlight counts:")
    print(results["status"].value_counts().to_string())
    print(f"Distance outliers retained (1.5-IQR rule): {results['distance_outlier'].sum()}")
    distances = results.loc[results["status"].eq("included"), "distance_nm"]
    if not distances.empty:
        print("\nHorizontal distance to DTW (NM):")
        print(distances.describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())
    print(f"\nResults: {csv_output}\nHistogram: {plot_output}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == "__main__":
    main()