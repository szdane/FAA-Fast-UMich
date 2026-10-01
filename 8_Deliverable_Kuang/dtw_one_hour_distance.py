"""Run to calculate horizontal distance to DTW one hour before BTS gate arrival.

Inputs and outputs are relative to this file. Run without arguments to save
a per-flight CSV and histogram, and display the histogram.

The current December 2023 filtered_rows.csv uses Detroit local time: matched
samples are UTC minus five hours relative to the original ZOB data. BTS
ARR_TIME is destination-local gate time with minute precision, not touchdown.
All calculations use America/Detroit. Do not apply this time assumption to
a replacement trajectory file without verifying its timestamp convention.

Only direct mainline matches are accepted: ICAO callsign mapped to marketing
carrier, matching marketing AND operating flight numbers, route and date.
The BTS export lacks OP_UNIQUE_CARRIER, so regional/codeshare operators are
left unmatched. Callsigns are not unique flights; results retain fltKey.

For continental-US origins, arrival at DTW is in the same or an easterly time
zone. ARR_TIME earlier than DEP_TIME therefore implies next-day arrival.
2400 means next midnight. Arrival dates are inferred, not explicitly supplied
by BTS; delays over 24 hours cannot be resolved from this export alone.
Matches must end within four hours before gate arrival. Missing coverage at
the target time is never extrapolated or replaced by the last track point.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtw_18000ft_distance import DTW_LAT0, DTW_LON0, NM_TO_METERS, SPHERE


SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Input"
OUTPUT_DIR = SCRIPT_DIR / "Output" / "DTW_one_hour_distance"
TIMEZONE = "America/Detroit"
MAX_GAP_SECONDS = 60.0
MAX_SPEED_KNOTS = 700.0
MAX_GATE_LAG_HOURS = 4.0
MAINLINE_CARRIERS = {
    "AAL": "AA", "DAL": "DL", "UAL": "UA", "SWA": "WN",
    "NKS": "NK", "FFT": "F9", "JBU": "B6", "ASA": "AS",
    "SCX": "SY", "AAY": "G4", "HAL": "HA",
}


def clock_minutes(value):
    """Parse a BTS HHMM clock, including 2400, without filling missing times."""
    if pd.isna(value) or not re.fullmatch(r"\d{1,4}", str(value)):
        return None
    hours, minutes = divmod(int(value), 100)
    if hours == 24 and minutes == 0:
        return 1440
    if hours >= 24 or minutes >= 60:
        return None
    return hours * 60 + minutes


def gate_timestamp(row):
    departure = clock_minutes(row["DEP_TIME"])
    arrival = clock_minutes(row["ARR_TIME"])
    if departure is None or arrival is None:
        return pd.NaT
    date = pd.to_datetime(row["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    if arrival < departure:
        arrival += 1440
    return (date + pd.Timedelta(minutes=arrival)).tz_localize(TIMEZONE)


def position_at(track, target):
    """Interpolate on the great-circle arc between adjacent valid samples."""
    result = {"status": "target_outside_recorded_track", "distance_nm": np.nan,
              "latitude": np.nan, "longitude": np.nan, "bracket_seconds": np.nan}
    if target < track["recTime"].min() or target > track["recTime"].max():
        return result
    after_index = track["recTime"].searchsorted(target)
    after = track.iloc[after_index]
    if after["recTime"] == target:
        latitude, longitude = after["coord1"], after["coord2"]
        result["bracket_seconds"] = 0.0
    else:
        before = track.iloc[after_index - 1]
        gap = (after["recTime"] - before["recTime"]).total_seconds()
        result["bracket_seconds"] = gap
        if gap > MAX_GAP_SECONDS:
            result["status"] = "target_gap_over_60_seconds"
            return result
        azimuth, _, length = SPHERE.inv(
            before["coord2"], before["coord1"], after["coord2"], after["coord1"]
        )
        if length / NM_TO_METERS / (gap / 3600) > MAX_SPEED_KNOTS:
            result["status"] = "implausible_segment_speed"
            return result
        fraction = (target - before["recTime"]).total_seconds() / gap
        longitude, latitude, _ = SPHERE.fwd(
            before["coord2"], before["coord1"], azimuth, fraction * length
        )
    _, _, distance = SPHERE.inv(longitude, latitude, DTW_LON0, DTW_LAT0)
    result.update(status="included", distance_nm=distance / NM_TO_METERS,
                  latitude=latitude, longitude=longitude)
    return result


def analyze_flight(track, bts):
    first = track.iloc[0]
    result = {
        "fltKey": first["fltKey"], "acId": first["acId"],
        "origin": first["dep"], "destination": "KDTW",
        "status": "unsupported_carrier_or_route", "arrival_basis": "BTS gate ARR_TIME",
        "bts_flight_date": "", "marketing_carrier": "", "flight_number": "",
        "tail_number": "", "gate_arrival_local": pd.NaT,
        "one_hour_before_local": pd.NaT, "last_track_to_gate_minutes": np.nan,
        "track_start_local": track["recTime"].min(),
        "track_end_local": track["recTime"].max(),
        "distance_nm": np.nan, "latitude": np.nan, "longitude": np.nan,
        "bracket_seconds": np.nan,
    }
    sample_columns = ["recTime", "coord1", "coord2"]
    track = track.sort_values("recTime").drop_duplicates(sample_columns)
    if (track[sample_columns].isna().any().any()
            or not np.isfinite(track[["coord1", "coord2"]]).all().all()
            or not track["coord1"].between(-90, 90).all()
            or not track["coord2"].between(-180, 180).all()):
        result["status"] = "invalid_track_sample"
        return result
    if track["recTime"].duplicated().any():
        result["status"] = "conflicting_timestamp_samples"
        return result
    identity = re.fullmatch(r"([A-Z]{3})(\d+)_.*", first["acId"])
    if identity is None or identity[1] not in MAINLINE_CARRIERS:
        return result
    if not re.fullmatch(r"K[A-Z]{3}", first["dep"]):
        return result
    carrier, number = MAINLINE_CARRIERS[identity[1]], str(int(identity[2]))
    result.update(marketing_carrier=carrier, flight_number=number)
    candidates = bts.loc[
        bts["MKT_CARRIER"].eq(carrier)
        & bts["MKT_CARRIER_FL_NUM"].eq(number)
        & bts["OP_CARRIER_FL_NUM"].eq(number)
        & bts["ORIGIN"].eq(first["dep"][1:])
    ].copy()
    start_date = track["recTime"].min().tz_localize(None).normalize()
    candidates = candidates.loc[candidates["flight_date"].between(
        start_date - pd.Timedelta(days=1), start_date
    )]
    if candidates.empty:
        result["status"] = "no_bts_identity_date_match"
        return result
    candidates["gate_time"] = candidates.apply(gate_timestamp, axis=1)
    if candidates["gate_time"].isna().all():
        result["status"] = "missing_bts_actual_times"
        return result
    candidates = candidates.dropna(subset=["gate_time"])
    lag = candidates["gate_time"] - track["recTime"].max()
    candidates = candidates.loc[lag.between(pd.Timedelta(0), pd.Timedelta(hours=MAX_GATE_LAG_HOURS))]
    candidates = candidates.drop_duplicates([
        "FL_DATE", "MKT_CARRIER", "MKT_CARRIER_FL_NUM", "TAIL_NUM", "ORIGIN", "gate_time"
    ])
    if len(candidates) != 1:
        result["status"] = "ambiguous_bts_match" if len(candidates) else "arrival_track_time_mismatch"
        return result
    match = candidates.iloc[0]
    target = match["gate_time"] - pd.Timedelta(hours=1)
    result.update(
        bts_flight_date=match["flight_date"].date().isoformat(),
        tail_number=match["TAIL_NUM"], gate_arrival_local=match["gate_time"],
        one_hour_before_local=target,
        last_track_to_gate_minutes=(match["gate_time"] - track["recTime"].max()).total_seconds() / 60,
    )
    result.update(position_at(track, target))
    return result


def load_results():
    columns = ["fltKey", "acId", "dep", "arrival", "recTime", "coord1", "coord2"]
    chunks = pd.read_csv(INPUT_DIR / "filtered_rows.csv", usecols=columns, chunksize=100000)
    tracks = pd.concat([chunk.loc[chunk["arrival"].eq("KDTW")] for chunk in chunks])
    tracks["recTime"] = pd.to_datetime(tracks["recTime"], format="mixed").dt.tz_localize(TIMEZONE)
    bts = pd.read_csv(INPUT_DIR / "T_ONTIME_MARKETING-8.txt", dtype=str)
    bts = bts.loc[bts["DEST"].eq("DTW")].copy()
    bts["flight_date"] = pd.to_datetime(bts["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p")
    return pd.DataFrame([analyze_flight(track, bts) for _, track in tracks.groupby("fltKey")])


def main(show_plot=True):
    results = load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "dtw_one_hour_distances.csv"
    results.to_csv(csv_path, index=False)
    distances = results.loc[results["status"].eq("included"), "distance_nm"]
    print("One hour before actual GATE arrival; Detroit local time; distances in NM.")
    print("Arrival calendar dates are inferred; regional/codeshare matches are not assumed.")
    print(results["status"].value_counts().to_string())
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    axis.set_title("DTW: One Hour Before Gate Arrival", loc="left", fontsize=18, pad=15)
    axis.set_xlabel("Horizontal great-circle distance to DTW (nautical miles)")
    axis.set_ylabel("Number of flights")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    if not distances.empty:
        print(distances.describe(percentiles=[0.1, 0.5, 0.9]).round(2).to_string())
        bins = np.arange(0, max(25, np.ceil(distances.max() / 25) * 25) + 25, 25)
        axis.hist(distances, bins=bins, color="#267a78", edgecolor="white")
        axis.axvline(distances.median(), color="#c24b38", label=f"Median {distances.median():.1f} NM")
        axis.legend(frameon=False)
    else:
        axis.text(0.5, 0.5, "No supported observations", ha="center", transform=axis.transAxes)
    figure.supxlabel(
        f"{len(distances)} observations / {len(results)} DTW flight keys\n"
        "BTS gate arrival, not touchdown. No extrapolation beyond recorded trajectories.\n"
        "Subset limited by BTS matching and trajectory coverage; not all DTW arrivals.", fontsize=10
    )
    png_path = OUTPUT_DIR / "dtw_one_hour_histogram.png"
    figure.savefig(png_path, dpi=180)
    print(f"\nResults: {csv_path}\nHistogram: {png_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(figure)
    return results


if __name__ == "__main__":
    main()