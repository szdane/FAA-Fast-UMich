r"""Estimate Boeing 787-8 fuel for every flight in a converted IFF CSV.

Install requirements_fuel.txt in a separate environment, then run:
    python "1_Data_Collection_Yuwei\compute_fuel_history.py" INPUT.csv OUTPUT_DIR --date 2023-12-22
    python "1_Data_Collection_Yuwei\animate_zob_day.py" INPUT.csv OUTPUT.mp4 --date 2023-12-22 --fuel-history OUTPUT_DIR

The method follows compute_fuel_flow in
8_Deliverable_Kuang\Functions\fuel_emission_analysis_computation.py: evaluate
OpenAP enroute at each observation, multiply kg/s by the preceding interval
in seconds, and subtract that fuel from aircraft mass. Unlike the historic
preprocessing, the first point has zero duration (no invented preceding leg).
Flights are identified by Source + fltKey, never by reusable callsigns.

Assumptions: every aircraft is B788 with its OpenAP default engine, initially
0.8 * MTOW at the first available observation; clean configuration, zero
acceleration, ISA, OpenAP's default thrust limits. Ground speed is used as TAS
because wind/airspeed data are absent. These are hypothetical model estimates,
not actual aircraft fuel measurements or complete airport-to-airport totals.
OpenAP 2.4 is pinned: 2.2 has a fallback fuel-scaling/engine-count defect for B788.

Retain all source points, including those outside --date, in the per-flight
history. Do not extrapolate before/after observations or across gaps >120 s.
An interval touching a speed outside [100,650] kt is excluded and flagged, not
clamped/repaired. The lower threshold excludes ground/very-slow states where
this airborne model is inappropriate and can overflow; it is not a taxi model.
650 kt is a conservative input sanity threshold, not an aircraft speed limit.
States above the model ceiling are flagged but still evaluated as the requested
B788 what-if extrapolation.
Excluded intervals have zero ACCOUNTED fuel, not an assertion of zero burn.
Invalid/negative OpenAP results abort the run; they are never changed to zero.

OUTPUT_DIR is created atomically and must not exist. It contains:
  flight_fuel_history.csv: one row per distinct observation; interval fuel,
      full-record and selected-day cumulative kg, mass, flow, and quality flags.
  flight_fuel_summary.csv: every flight's totals and excluded time.
  daily_cumulative_fuel.csv: exact fleet-wide rate-change times, including
      midnight boundaries. Linear interpolation is exact under this model.
  run_metadata.json: units, assumptions, source hash, model version, counts,
      exclusions and measured first-three-flight runtime estimates.
The daily curve sums only intervals clipped to the selected local day;
previous-day fuel reduces aircraft mass but is not counted in that day's total.
Naive timestamps require matching --timezone (America/Detroit by default).
For DST-transition days, use a UTC-converted CSV and --timezone UTC.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from animate_zob_day import DAY_SECONDS, MAX_GAP_SECONDS, Flight, load_flights

LOGGER = logging.getLogger(__name__)
AIRCRAFT = "b788"
MASS_FRACTION = 0.8
MAX_INPUT_SPEED_KT = 650
MIN_ENROUTE_SPEED_KT = 100


class FuelModel(Protocol):
    def enroute(self, *, mass: float, tas: float, alt: float, vs: float) -> float:
        ...


@dataclass
class FuelHistory:
    flow: np.ndarray
    duration: np.ndarray
    modeled_duration: np.ndarray
    interval_fuel: np.ndarray
    cumulative_fuel: np.ndarray
    mass: np.ndarray
    day_duration: np.ndarray
    day_fuel: np.ndarray
    status: list[str]
    flags: list[str]


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_day(day: date, timezone_name: str) -> None:
    start = datetime.combine(day, time(), ZoneInfo(timezone_name))
    if start.utcoffset() != (start + timedelta(days=1)).utcoffset():
        raise ValueError("DST-transition days require a UTC-converted CSV and --timezone UTC.")


def create_model():
    if version("openap") != "2.4":
        raise ValueError("This calculation requires OpenAP 2.4; install requirements_fuel.txt in a separate environment.")
    from openap import FuelFlow, prop

    aircraft = prop.aircraft(AIRCRAFT)
    model = FuelFlow(AIRCRAFT)
    mass = MASS_FRACTION * aircraft["mtow"]
    for tas, altitude, vertical_rate in ((450, 35000, 0), (250, 10000, 1500), (220, 5000, -1500)):
        flow = float(model.enroute(mass=mass, tas=tas, alt=altitude, vs=vertical_rate))
        if not np.isfinite(flow) or flow <= 0:
            raise ValueError(f"OpenAP B788 sanity check returned invalid fuel flow: {flow} kg/s.")
    return model, aircraft


def compute_flight(flight: Flight, model: FuelModel, initial_mass: float,
                   empty_mass: float, ceiling_ft: float) -> FuelHistory:
    samples, kinematics = flight.samples, flight.kinematics
    if kinematics is None or kinematics.shape != (len(samples), 2) or not len(samples):
        raise ValueError(f"Flight {flight.key} requires speed/climb data for every observation.")
    if not np.isfinite(samples).all() or not np.isfinite(kinematics).all():
        raise ValueError(f"Flight {flight.key} has nonfinite inputs.")
    if not 0 < empty_mass < initial_mass or not np.isfinite(initial_mass):
        raise ValueError("Initial mass must be finite and above operating empty weight.")
    duration = np.r_[0.0, np.diff(samples[:, 0])]
    if np.any(duration[1:] <= 0):
        raise ValueError(f"Flight {flight.key} timestamps must be strictly increasing.")
    n = len(samples)
    flow, modeled = np.full(n, np.nan), np.zeros(n)
    burn, masses = np.zeros(n), np.zeros(n)
    status, flags = [], []
    usable_speed = ((kinematics[:, 0] >= MIN_ENROUTE_SPEED_KT)
                    & (kinematics[:, 0] <= MAX_INPUT_SPEED_KT))
    mass = initial_mass
    for i, (sample, state) in enumerate(zip(samples, kinematics)):
        _, _, _, altitude = sample
        speed, vertical_rate = state
        state_flags = []
        if speed <= 0 or speed > MAX_INPUT_SPEED_KT:
            state_flags.append("invalid_reported_speed")
        elif speed < MIN_ENROUTE_SPEED_KT:
            state_flags.append("below_100kt_outside_B788_enroute_model")
        if altitude > ceiling_ft:
            state_flags.append("above_B788_ceiling_extrapolation")
        flags.append(";".join(state_flags))
        if usable_speed[i]:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    flow[i] = float(model.enroute(mass=mass, tas=speed, alt=altitude, vs=vertical_rate))
            except FloatingPointError as error:
                raise ValueError(f"OpenAP numerical failure for fltKey {flight.key} at elapsed {sample[0]} "
                                 f"(TAS={speed} kt, alt={altitude} ft, VS={vertical_rate} ft/min): {error}") from error
            if not np.isfinite(flow[i]) or flow[i] <= 0:
                raise ValueError(f"Invalid OpenAP fuel flow for fltKey {flight.key}, elapsed {sample[0]}: {flow[i]}.")
        if i == 0:
            status.append("first_observation")
        elif duration[i] > MAX_GAP_SECONDS:
            status.append("excluded_gap_over_120s")
        elif not (usable_speed[i - 1] and usable_speed[i]):
            status.append("excluded_speed_endpoint")
        else:
            status.append("modeled")
            modeled[i] = duration[i]
            burn[i] = flow[i] * duration[i]
            mass -= burn[i]
        if mass <= empty_mass:
            raise ValueError(f"Flight {flight.key}: modeled mass fell below operating empty weight.")
        masses[i] = mass
    begin = samples[:, 0] - duration
    overlap = np.maximum(0, np.minimum(samples[:, 0], DAY_SECONDS) - np.maximum(begin, 0))
    day_duration = np.where(modeled > 0, overlap, 0)
    day_fuel = np.divide(burn * day_duration, modeled, out=np.zeros(n), where=modeled > 0)
    return FuelHistory(flow, duration, modeled, burn, np.cumsum(burn), masses,
                       day_duration, day_fuel, status, flags)


def fuel_events(flight: Flight, history: FuelHistory) -> np.ndarray:
    used = history.day_duration > 0
    end = np.minimum(flight.samples[used, 0], DAY_SECONDS)
    begin = end - history.day_duration[used]
    rate = history.flow[used]
    return np.column_stack((np.r_[begin, end], np.r_[rate, -rate]))


def aggregate_day(events: list[np.ndarray]) -> np.ndarray:
    changes = np.concatenate([np.array([[0.0, 0.0], [DAY_SECONDS, 0.0]]), *events])
    changes = changes[np.argsort(changes[:, 0], kind="stable")]
    instants, indices = np.unique(changes[:, 0], return_index=True)
    rate = np.cumsum(np.add.reduceat(changes[:, 1], indices))
    if rate.min() < -1e-7 or abs(rate[-1]) > 1e-7:
        raise ValueError("Fleet fuel-rate aggregation did not conserve interval start/end events.")
    rate = np.maximum(rate, 0)  # Roundoff from canceling positive and negative events.
    cumulative = np.r_[0.0, np.cumsum(rate[:-1] * np.diff(instants))]
    return np.column_stack((instants, cumulative))


def load_daily_fuel(directory: Path, input_path: Path, day: date, timezone_name: str) -> np.ndarray:
    with (directory / "run_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (metadata["date"] != day.isoformat() or metadata["timezone"] != timezone_name
            or metadata["aircraft"] != AIRCRAFT or metadata["mass_fraction"] != MASS_FRACTION
            or metadata["source_sha256"] != source_sha256(input_path)):
        raise ValueError("Fuel history does not match this input CSV, date, timezone, or B788 scenario.")
    with (directory / "daily_cumulative_fuel.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["recTime", "elapsed_seconds", "cumulative_fuel_kg"]:
            raise ValueError("Unexpected daily fuel CSV columns.")
        curve = np.array([(float(row["elapsed_seconds"]), float(row["cumulative_fuel_kg"])) for row in reader])
    if (curve.ndim != 2 or curve.shape[0] < 2 or curve.shape[1] != 2
            or not np.isfinite(curve).all() or tuple(curve[[0, -1], 0]) != (0, DAY_SECONDS)
            or curve[0, 1] != 0 or np.any(np.diff(curve[:, 0]) <= 0)
            or np.any(np.diff(curve[:, 1]) < 0)
            or not np.isclose(curve[-1, 1], metadata["day_fuel_kg"], rtol=1e-10, atol=1e-6)):
        raise ValueError("Daily cumulative fuel must span the day, increase monotonically, and match the recorded total.")
    return curve


def compute_file(input_path: Path, output_dir: Path, day: date,
                 timezone_name: str = "America/Detroit") -> dict:
    started = perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    validate_day(day, timezone_name)
    source_hash = source_sha256(input_path)
    flights = load_flights(input_path, day, include_kinematics=True, full_history=True)
    if not any(np.any((f.samples[:, 0] >= 0) & (f.samples[:, 0] < DAY_SECONDS)) for f in flights):
        raise ValueError(f"No observations on {day}.")
    model, aircraft = create_model()
    initial_mass = MASS_FRACTION * aircraft["mtow"]
    total_points = sum(len(f.samples) for f in flights)
    LOGGER.info("Loaded %s flights / %s observations. B788, %s, initial mass %.0f kg.",
                f"{len(flights):,}", f"{total_points:,}", aircraft["engine"]["default"], initial_mass)
    LOGGER.warning("GS is treated as TAS without wind. Gaps >120 s and intervals touching speeds outside [100,650] kt are excluded, not zero-burn estimates.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    origin = datetime.combine(day, time())

    def timestamp(seconds):
        return (origin + timedelta(seconds=float(seconds))).isoformat(sep=" ", timespec="microseconds")

    compute_seconds, pilot_points = 0.0, 0
    events, summaries = [], []
    intervals, state_counts = Counter(), Counter()
    pilot = {}
    with TemporaryDirectory(prefix=".fuel-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        processing_start = perf_counter()
        with (staging / "flight_fuel_history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Source", "fltKey", "acId", "dep", "arrival", "recTime",
                             "elapsed_seconds", "alt_ft", "groundSpeed_kt", "rateOfClimb_fpm",
                             "interval_seconds", "modeled_seconds", "fuel_flow_kg_s",
                             "interval_fuel_kg", "cumulative_fuel_kg", "mass_kg",
                             "day_modeled_seconds", "day_interval_fuel_kg",
                             "day_cumulative_fuel_kg", "interval_status", "state_flags"])
            for index, flight in enumerate(flights):
                tick = perf_counter()
                history = compute_flight(flight, model, initial_mass, aircraft["oew"],
                                         aircraft["ceiling"] / 0.3048)
                elapsed = perf_counter() - tick
                compute_seconds += elapsed
                events.append(fuel_events(flight, history))
                intervals.update(history.status)
                state_counts.update(flag for flags in history.flags for flag in flags.split(";") if flag)
                day_cumulative = np.cumsum(history.day_fuel)
                for i, sample in enumerate(flight.samples):
                    writer.writerow([
                        flight.facility, flight.key, flight.callsign, flight.departure, flight.arrival,
                        timestamp(sample[0]), f"{sample[0]:.6f}", sample[3], *flight.kinematics[i],
                        f"{history.duration[i]:.6f}", f"{history.modeled_duration[i]:.6f}",
                        f"{history.flow[i]:.9f}" if np.isfinite(history.flow[i]) else "",
                        f"{history.interval_fuel[i]:.9f}", f"{history.cumulative_fuel[i]:.9f}",
                        f"{history.mass[i]:.9f}", f"{history.day_duration[i]:.6f}",
                        f"{history.day_fuel[i]:.9f}", f"{day_cumulative[i]:.9f}",
                        history.status[i], history.flags[i],
                    ])
                day_span = np.maximum(0, np.minimum(flight.samples[:, 0], DAY_SECONDS)
                                      - np.maximum(flight.samples[:, 0] - history.duration, 0))
                summaries.append({
                    "Source": flight.facility, "fltKey": flight.key, "acId": flight.callsign,
                    "dep": flight.departure, "arrival": flight.arrival,
                    "first_observation": timestamp(flight.samples[0, 0]),
                    "last_observation": timestamp(flight.samples[-1, 0]),
                    "observations": len(flight.samples),
                    "modeled_intervals": history.status.count("modeled"),
                    "modeled_seconds": float(history.modeled_duration.sum()),
                    "excluded_seconds": float((history.duration - history.modeled_duration).sum()),
                    "fuel_kg": float(history.cumulative_fuel[-1]),
                    "day_fuel_kg": float(day_cumulative[-1]),
                    "day_modeled_seconds": float(history.day_duration.sum()),
                    "day_excluded_seconds": float((day_span - history.day_duration).sum()),
                    "initial_mass_kg": initial_mass, "final_mass_kg": float(history.mass[-1]),
                    "flagged_observations": sum(bool(flag) for flag in history.flags),
                    "coverage": ("unavailable_no_modeled_intervals" if not np.any(history.modeled_duration > 0)
                                 else "partial" if np.any(history.duration != history.modeled_duration)
                                 else "observed_intervals_only"),
                })
                if index < 3:
                    pilot_points += len(flight.samples)
                    LOGGER.info("Pilot %d: fltKey=%s, %d points, fuel computation %.3f s.",
                                index + 1, flight.key, len(flight.samples), elapsed)
                if index + 1 == min(3, len(flights)):
                    processing = perf_counter() - processing_start
                    pilot = {
                        "flights": index + 1, "observations": pilot_points,
                        "compute_seconds": compute_seconds, "processing_seconds": processing,
                        "estimated_all_flights_compute_seconds": compute_seconds * total_points / pilot_points,
                        "estimated_remaining_processing_seconds": processing * (total_points - pilot_points) / pilot_points,
                    }
                    LOGGER.info("INITIAL ESTIMATE from %d flights: %.1f min total fuel computation; %.1f min remaining processing/CSV writing. Excludes final aggregation and video rendering.",
                                pilot["flights"], pilot["estimated_all_flights_compute_seconds"] / 60,
                                pilot["estimated_remaining_processing_seconds"] / 60)
                if (index + 1) % 500 == 0:
                    LOGGER.info("Processed %s/%s flights (%.1f min elapsed).", index + 1, len(flights),
                                (perf_counter() - processing_start) / 60)
        curve = aggregate_day(events)
        total_day = sum(row["day_fuel_kg"] for row in summaries)
        if not np.isclose(curve[-1, 1], total_day, rtol=1e-10, atol=1e-6):
            raise ValueError("Fleet curve and per-flight fuel totals disagree.")
        with (staging / "flight_fuel_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        with (staging / "daily_cumulative_fuel.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["recTime", "elapsed_seconds", "cumulative_fuel_kg"])
            writer.writerows((timestamp(t), f"{t:.6f}", f"{fuel:.9f}") for t, fuel in curve)
        metadata = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "input": str(input_path.resolve()), "source_sha256": source_hash,
            "date": day.isoformat(), "timezone": timezone_name, "aircraft": AIRCRAFT,
            "engine": aircraft["engine"]["default"], "mass_fraction": MASS_FRACTION,
            "initial_mass_kg": initial_mass, "ceiling_ft": aircraft["ceiling"] / 0.3048,
            "minimum_enroute_speed_kt": MIN_ENROUTE_SPEED_KT, "maximum_input_speed_kt": MAX_INPUT_SPEED_KT,
            "versions": {package: version(package) for package in ("openap", "numpy", "pandas", "scipy")},
            "flight_count": len(flights), "observation_count": total_points,
            "day_fuel_kg": total_day, "full_record_fuel_kg": sum(row["fuel_kg"] for row in summaries),
            "modeled_seconds": sum(row["modeled_seconds"] for row in summaries),
            "excluded_seconds": sum(row["excluded_seconds"] for row in summaries),
            "day_modeled_seconds": sum(row["day_modeled_seconds"] for row in summaries),
            "day_excluded_seconds": sum(row["day_excluded_seconds"] for row in summaries),
            "interval_status_counts": dict(intervals), "state_flag_counts": dict(state_counts),
            "flights_without_modeled_intervals": sum(row["modeled_intervals"] == 0 for row in summaries),
            "pilot_timing": pilot, "actual_compute_seconds": compute_seconds,
            "actual_processing_seconds": perf_counter() - processing_start,
            "total_seconds": perf_counter() - started,
            "method": "Right-endpoint OpenAP enroute flow times preceding dt; decreasing mass; first dt=0.",
            "assumptions": [
                "Every aircraft is B788 at 0.8 MTOW at its first recorded point.",
                "Ground speed approximates TAS; no wind correction; ISA; clean configuration; acceleration=0.",
                "All reported tracks are retained, including tracks outside the ARTCC contours.",
                "No time extrapolation; intervals >120 s or touching speeds outside [100,650] kt are excluded, not zero burn.",
                "Below 100 kt the airborne model is inappropriate; no taxi fuel is invented.",
                "Above the B788 ceiling, states are flagged what-if model extrapolations.",
                "Mass is reduced only by modeled fuel, so mass after missing intervals is uncertain.",
                "Daily fuel excludes other dates; full-record histories include every date in this input.",
                "These are modeled observed-segment totals, not measured or whole-journey consumption.",
            ],
        }
        (staging / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if source_sha256(input_path) != source_hash:
            raise ValueError("Input changed during fuel computation; results were not published.")
        if output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        staging.rename(output_dir)
    LOGGER.warning("Quality report: %s; state flags: %s; excluded %.1f aircraft-seconds in the selected day.",
                   dict(intervals), dict(state_counts), metadata["day_excluded_seconds"])
    if metadata["flights_without_modeled_intervals"]:
        LOGGER.warning("%s flights have no usable airborne intervals: their zero accounted fuel is NOT a fuel-consumption estimate.",
                       metadata["flights_without_modeled_intervals"])
    LOGGER.info("Saved all %s flight histories. Day total: %.3f kg; actual compute time: %.2f min; total: %.2f min.",
                len(flights), total_day, compute_seconds / 60, metadata["total_seconds"] / 60)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--date", required=True, type=date.fromisoformat)
    parser.add_argument("--timezone", default="America/Detroit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        compute_file(args.input, args.output_dir, args.date, args.timezone)
    except (OSError, ValueError, csv.Error, FloatingPointError, ZoneInfoNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
