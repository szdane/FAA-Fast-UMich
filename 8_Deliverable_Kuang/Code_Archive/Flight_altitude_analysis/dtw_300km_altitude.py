"""Plot DTW arrival altitudes at the first inward crossing of 300 km.

The flight cohort comes from Input/entry_exit_points.csv. For each listed
flight, the crossing is found on its recorded Input/filtered_rows.csv track;
position is interpolated along the great-circle segment and altitude linearly
between samples. The recorded ``alt`` field is in hundreds of feet.

Only crossings bracketed by valid samples no more than 60 seconds apart and
with segment speed at or below 700 knots are included. No extrapolation is
performed. One first inward crossing is used per event flight.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtw_18000ft_distance import DTW_LAT0, DTW_LON0, NM_TO_METERS, SPHERE
from dtw_one_hour_distance import position_at


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Input"
OUTPUT_DIR = SCRIPT_DIR / "Output" / "DTW_300km_altitude"
TARGET_DISTANCE_METERS = 300_000.0
FEET_PER_ALTITUDE_UNIT = 100.0
MAX_CROSSING_GAP_SECONDS = 60.0
MAX_SEGMENT_SPEED_KNOTS = 700.0
MAX_EVENT_POSITION_ERROR_NM = 2.0
BIN_WIDTH_FT = 2_000.0
TRACK_COLUMNS = ["fltKey", "acId", "arrival", "recTime", "alt", "coord1", "coord2"]


def crossing_fraction(before, azimuth, segment_meters):
    """Find the along-segment fraction whose geodesic distance to DTW is 300 km."""
    lower, upper = 0.0, 1.0
    for _ in range(40):
        fraction = (lower + upper) / 2
        longitude, latitude, _ = SPHERE.fwd(
            before["coord2"], before["coord1"], azimuth, fraction * segment_meters
        )
        _, _, distance = SPHERE.inv(longitude, latitude, DTW_LON0, DTW_LAT0)
        if distance > TARGET_DISTANCE_METERS:
            lower = fraction
        else:
            upper = fraction
    return (lower + upper) / 2


def analyze_flight(track, event):
    """Return the first inward 300-km crossing or an explicit exclusion status."""
    first = track.iloc[0]
    result = {
        "fltKey": first["fltKey"],
        "acId": first["acId"],
        "entry_altitude_ft": event["entry_alt"] * FEET_PER_ALTITUDE_UNIT,
        "exit_altitude_ft": event["exit_alt"] * FEET_PER_ALTITUDE_UNIT,
        "status": "no_300km_inbound_crossing",
        "crossing_time": pd.NaT,
        "crossing_altitude_ft": np.nan,
        "crossing_latitude": np.nan,
        "crossing_longitude": np.nan,
        "crossing_distance_km": np.nan,
        "bracket_seconds": np.nan,
        "bracket_speed_knots": np.nan,
        "track_start": track["recTime"].min(),
        "track_end": track["recTime"].max(),
    }
    sample_columns = ["recTime", "alt", "coord1", "coord2"]
    track = track.sort_values("recTime").drop_duplicates(sample_columns)
    if (
        track[sample_columns].isna().any().any()
        or not np.isfinite(track[["alt", "coord1", "coord2"]]).all().all()
        or not track["coord1"].between(-90, 90).all()
        or not track["coord2"].between(-180, 180).all()
    ):
        result["status"] = "invalid_track_sample"
        return result
    if track["recTime"].duplicated().any():
        result["status"] = "conflicting_timestamp_samples"
        return result

    track = track.reset_index(drop=True)
    _, _, distances = SPHERE.inv(
        track["coord2"].to_numpy(),
        track["coord1"].to_numpy(),
        np.full(len(track), DTW_LON0),
        np.full(len(track), DTW_LAT0),
    )
    for index in range(len(track) - 1):
        before_distance, after_distance = distances[index:index + 2]
        if not (
            before_distance >= TARGET_DISTANCE_METERS
            and after_distance <= TARGET_DISTANCE_METERS
            and before_distance > after_distance
        ):
            continue

        before, after = track.iloc[index], track.iloc[index + 1]
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

        fraction = crossing_fraction(before, azimuth, segment_meters)
        longitude, latitude, _ = SPHERE.fwd(
            before["coord2"], before["coord1"], azimuth, fraction * segment_meters
        )
        _, _, distance_meters = SPHERE.inv(
            longitude, latitude, DTW_LON0, DTW_LAT0
        )
        result.update(
            status="included",
            crossing_time=before["recTime"] + pd.to_timedelta(fraction * gap, unit="s"),
            crossing_altitude_ft=(
                before["alt"] + fraction * (after["alt"] - before["alt"])
            ) * FEET_PER_ALTITUDE_UNIT,
            crossing_latitude=latitude,
            crossing_longitude=longitude,
            crossing_distance_km=distance_meters / 1000.0,
        )
        return result
    return result


def event_matches_track(track, event):
    """Confirm that the event's entry and exit points belong to this track."""
    if pd.isna(event["entry_rectime"]) or pd.isna(event["exit_rectime"]):
        return False
    track = track.sort_values("recTime").drop_duplicates(
        ["recTime", "coord1", "coord2"]
    )
    if not (
        track["recTime"].min() <= event["entry_rectime"] <= track["recTime"].max()
        and track["recTime"].min() <= event["exit_rectime"] <= track["recTime"].max()
    ):
        return False

    for point in ["entry", "exit"]:
        latitude, longitude = event[f"{point}_lat"], event[f"{point}_lon"]
        if (
            pd.isna(latitude)
            or pd.isna(longitude)
            or not (-90 <= latitude <= 90 and -180 <= longitude <= 180)
        ):
            return False
        position = position_at(track, event[f"{point}_rectime"])
        if position["status"] != "included":
            return False
        _, _, error_meters = SPHERE.inv(
            position["longitude"], position["latitude"], longitude, latitude
        )
        if error_meters / NM_TO_METERS > MAX_EVENT_POSITION_ERROR_NM:
            return False
    return True


def load_results():
    """Analyze the DTW trajectories belonging to listed entry/exit events."""
    events = pd.read_csv(INPUT_DIR / "entry_exit_points.csv")
    events = events.drop_duplicates("acId").copy()
    for column in ["entry_alt", "exit_alt"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ["entry_lat", "entry_lon", "exit_lat", "exit_lon"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    for column in ["entry_rectime", "exit_rectime"]:
        events[column] = pd.to_datetime(events[column], format="mixed", errors="coerce")

    event_ids = set(events["acId"].dropna())
    matching_chunks = []
    for chunk in pd.read_csv(
        INPUT_DIR / "filtered_rows.csv",
        usecols=TRACK_COLUMNS,
        chunksize=100_000,
    ):
        matching_chunks.append(
            chunk.loc[chunk["arrival"].eq("KDTW") & chunk["acId"].isin(event_ids)]
        )
    tracks = pd.concat(matching_chunks, ignore_index=True)
    tracks["recTime"] = pd.to_datetime(
        tracks["recTime"], format="mixed", errors="coerce"
    )
    for column in ["alt", "coord1", "coord2"]:
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce")

    by_ac_id = {ac_id: group for ac_id, group in tracks.groupby("acId", sort=False)}
    results = []
    for _, event in events.iterrows():
        event_tracks = by_ac_id.get(event["acId"])
        if event_tracks is None:
            results.append({
                "fltKey": "",
                "acId": event["acId"],
                "entry_altitude_ft": event["entry_alt"] * FEET_PER_ALTITUDE_UNIT,
                "exit_altitude_ft": event["exit_alt"] * FEET_PER_ALTITUDE_UNIT,
                "status": "no_matching_DTW_trajectory",
                "crossing_time": pd.NaT,
                "crossing_altitude_ft": np.nan,
                "crossing_latitude": np.nan,
                "crossing_longitude": np.nan,
                "crossing_distance_km": np.nan,
                "bracket_seconds": np.nan,
                "bracket_speed_knots": np.nan,
                "track_start": pd.NaT,
                "track_end": pd.NaT,
            })
            continue
        matching_tracks = [
            track
            for _, track in event_tracks.groupby("fltKey", sort=False)
            if event_matches_track(track, event)
        ]
        if not matching_tracks:
            results.append({
                "fltKey": "",
                "acId": event["acId"],
                "entry_altitude_ft": event["entry_alt"] * FEET_PER_ALTITUDE_UNIT,
                "exit_altitude_ft": event["exit_alt"] * FEET_PER_ALTITUDE_UNIT,
                "status": "no_matching_event_trajectory",
                "crossing_time": pd.NaT,
                "crossing_altitude_ft": np.nan,
                "crossing_latitude": np.nan,
                "crossing_longitude": np.nan,
                "crossing_distance_km": np.nan,
                "bracket_seconds": np.nan,
                "bracket_speed_knots": np.nan,
                "track_start": event_tracks["recTime"].min(),
                "track_end": event_tracks["recTime"].max(),
            })
            continue
        if len(matching_tracks) != 1:
            results.append({
                "fltKey": "",
                "acId": event["acId"],
                "entry_altitude_ft": event["entry_alt"] * FEET_PER_ALTITUDE_UNIT,
                "exit_altitude_ft": event["exit_alt"] * FEET_PER_ALTITUDE_UNIT,
                "status": "ambiguous_matching_trajectories",
                "crossing_time": pd.NaT,
                "crossing_altitude_ft": np.nan,
                "crossing_latitude": np.nan,
                "crossing_longitude": np.nan,
                "crossing_distance_km": np.nan,
                "bracket_seconds": np.nan,
                "bracket_speed_knots": np.nan,
                "track_start": event_tracks["recTime"].min(),
                "track_end": event_tracks["recTime"].max(),
            })
            continue
        results.append(analyze_flight(matching_tracks[0], event))
    return pd.DataFrame(results)


def plot_histogram(results):
    """Plot included crossing altitudes in feet."""
    altitudes = results.loc[
        results["status"].eq("included"), "crossing_altitude_ft"
    ]
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    figure.set_facecolor("#fafafa")
    axis.set_facecolor("#fafafa")
    axis.set_title("DTW Arrivals 300 km from DTW", loc="left", fontsize=19, pad=16)
    axis.set_xlabel("Altitude at first inward 300 km crossing (ft)")
    axis.set_ylabel("Number of flights")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    if altitudes.empty:
        axis.text(0.5, 0.5, "No eligible 300 km crossings", ha="center",
                  transform=axis.transAxes)
        return figure

    bin_start = np.floor(altitudes.min() / BIN_WIDTH_FT) * BIN_WIDTH_FT
    bin_end = np.ceil(altitudes.max() / BIN_WIDTH_FT) * BIN_WIDTH_FT
    bins = np.arange(bin_start, bin_end + BIN_WIDTH_FT, BIN_WIDTH_FT)
    axis.hist(altitudes, bins=bins, color="#267a78", edgecolor="white", linewidth=1)
    median = altitudes.median()
    axis.axvline(median, color="#c24b38", linewidth=2,
                 label=f"Median: {median:,.0f} ft")
    axis.legend(loc="upper right", frameon=False)
    percentile_10, percentile_90 = altitudes.quantile([0.1, 0.9])
    figure.supxlabel(
        f"{len(altitudes)} included / {len(results)} event flights | "
        f"Mean {altitudes.mean():,.0f} ft | P10-P90 "
        f"{percentile_10:,.0f}-{percentile_90:,.0f} ft\n"
        "First inward 300 km crossing per listed flight; altitude linearly interpolated.\n"
        "Recorded trajectory samples only; no extrapolation.",
        fontsize=10,
    )
    return figure


def main(show_plot=True):
    print(f"Reading event flights from {INPUT_DIR / 'entry_exit_points.csv'}")
    results = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_output = OUTPUT_DIR / "dtw_300km_altitudes.csv"
    plot_output = OUTPUT_DIR / "dtw_300km_altitude_histogram.png"
    results.to_csv(csv_output, index=False)

    figure = plot_histogram(results)
    print("\nFlight counts:")
    print(results["status"].value_counts().to_string())
    altitudes = results.loc[
        results["status"].eq("included"), "crossing_altitude_ft"
    ]
    if not altitudes.empty:
        print("\nAltitude at first inward 300 km crossing (ft):")
        print(altitudes.describe(percentiles=[0.1, 0.5, 0.9]).round(0).to_string())
    figure.savefig(plot_output, dpi=180)
    print(f"\nResults: {csv_output}\nHistogram: {plot_output}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == "__main__":
    main()
