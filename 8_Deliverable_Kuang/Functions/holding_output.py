"""Shared v18-style holding reports and plots; no model construction or solving.

write_case_output accepts solved per-aircraft position, arrival, hold and fuel
variables on local grids, plus exact entry offsets in the scenario. It exports
feasible incumbents even at a time limit and rejects missing solutions or failed
weather/separation checks. The caller supplies the chosen STAR in each case.

write_comparison_output writes fleet/per-aircraft comparisons and the shaded
weather/holding plot after both cases finish. Fuel, distance and airborne time
remain physical quantities regardless of the holding objective multiplier.
"""

import json
from itertools import combinations
from pathlib import Path

import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from Functions.free_holding import (
    hold_bounds_xy, overlapping_intervals, position_at, report_free_holding, weather_active, weather_windows,
)


def report_aircraft(scenario, aircraft, variables, fuel_table, output):
    xyz, arrived, hold, route_fuel, hold_fuel = variables
    n, dt = scenario["horizon_intervals"], scenario["time_step_seconds"]
    offset = aircraft["entry_seconds"]
    epoch = pd.Timestamp(scenario["entry_utc"])
    data = pd.DataFrame([
        {"t": offset + k * dt, "t_step": k,
         "t_utc": (epoch + pd.Timedelta(seconds=offset + k * dt)).isoformat(),
         "f1_x": xyz[0][k].X, "f1_y": xyz[1][k].X, "f1_z": xyz[2][k].X,
         "f1_status": 2 if arrived[k].X > 0.5 else 1, "f1_hold": int(hold[k].X > 0.5)}
        for k in range(n + 1)
    ])
    data.to_csv(output / "df_wide.csv", index=False)
    interval_rows = [
        {"end_step": k, "start_t": offset + (k - 1) * dt, "end_t": offset + k * dt,
         "x": xyz[0][k].X, "y": xyz[1][k].X, "z": xyz[2][k].X,
         "hold_pwl_fuel_kg": hold_fuel[k].X,
         "hold_fuel_kg": dt * float(fuel_table.exact(scenario["hold_speed_mps"], xyz[2][k].X, 0))}
        for k in range(1, n + 1) if hold[k].X > 0.5
    ]
    pd.DataFrame(interval_rows, columns=[
        "end_step", "start_t", "end_t", "x", "y", "z", "hold_pwl_fuel_kg", "hold_fuel_kg"
    ]).to_csv(output / "holding_intervals.csv", index=False)
    report = report_free_holding(data, aircraft, scenario, output, scenario["mass_fraction"]).iloc[0]
    route_distance = sum(
        float(np.linalg.norm(data.loc[k, ["f1_x", "f1_y"]].to_numpy(dtype=float)
                             - data.loc[k - 1, ["f1_x", "f1_y"]].to_numpy(dtype=float)))
        for k in range(1, n + 1) if data.f1_status.iloc[k - 1] == 1 and not data.f1_hold.iloc[k])
    result = {
        "acId": aircraft["acId"], "entry_seconds": offset, "star_fix": aircraft["star_fix"],
        "arrival_seconds": offset + float(report.total_seconds),
        "airborne_seconds": float(report.total_seconds), "hold_seconds": float(report.hold_seconds),
        "hold_intervals": len(interval_rows), "route_distance_km": route_distance / 1000,
        "total_distance_km": (route_distance + report.hold_seconds * scenario["hold_speed_mps"]) / 1000,
        "route_fuel_kg": float(report.route_fuel_kg), "hold_fuel_kg": float(report.hold_fuel_kg),
        "total_fuel_kg": float(report.total_fuel_kg), "total_CO2_kg": float(report.total_CO2_g / 1000),
        "pwl_fuel_kg": sum(v.X for v in route_fuel.values()) + sum(v.X for v in hold_fuel.values()),
        "hold_pwl_fuel_kg": sum(v.X for v in hold_fuel.values()),
    }
    return result, data


def validate_separation(scenario, schedules, output):
    lower, upper = hold_bounds_xy(scenario)
    rows = []
    for a, b in combinations(range(len(schedules)), 2):
        first, second = scenario["aircraft"][a], scenario["aircraft"][b]
        da, db = schedules[a], schedules[b]
        for k, j, start, end in overlapping_intervals(first, second, scenario):
            if da.f1_status.iloc[k - 1] != 1 or db.f1_status.iloc[j - 1] != 1:
                continue
            gaps = []
            for time_s in (start, end):
                positions = []
                for data, aircraft, step in ((da, first, k), (db, second, j)):
                    xyz = [data[column].to_numpy() for column in ("f1_x", "f1_y", "f1_z")]
                    positions.append(np.asarray(position_at(
                        xyz, aircraft, step, time_s, scenario["time_step_seconds"])))
                pa, pb = positions
                ha, hb = da.f1_hold.iloc[k], db.f1_hold.iloc[j]
                signed = []
                for axis in range(3):
                    lo, hi = (lower[axis], upper[axis]) if axis < 2 else (0, 0)
                    clearance = scenario["horizontal_separation_m"] if axis < 2 else scenario["vertical_separation_m"]
                    signed.extend([pa[axis] + lo * ha - pb[axis] - hi * hb - clearance,
                                   pb[axis] + lo * hb - pa[axis] - hi * ha - clearance])
                gaps.append(signed)
            margin = float(np.min(gaps, axis=0).max())
            if margin < -0.01:
                raise ValueError(f"Separation check failed for {first['acId']}/{second['acId']} at {start}-{end}s.")
            rows.append({"aircraft_a": first["acId"], "aircraft_b": second["acId"],
                         "start_t": start, "end_t": end, "separating_margin_m": margin})
    pd.DataFrame(rows, columns=["aircraft_a", "aircraft_b", "start_t", "end_t",
                               "separating_margin_m"]).to_csv(output / "separation_checks.csv", index=False)


def write_case_output(model, scenario, variables, fuel_table, output, holding_enabled,
                      initialization_seconds=0.0, warm_start_labels=()):
    status_names = {gp.GRB.OPTIMAL: "OPTIMAL", gp.GRB.TIME_LIMIT: "TIME_LIMIT",
                    gp.GRB.INFEASIBLE: "INFEASIBLE", gp.GRB.INF_OR_UNBD: "INF_OR_UNBD",
                    gp.GRB.INTERRUPTED: "INTERRUPTED", gp.GRB.NUMERIC: "NUMERIC"}
    result = {
        "case": "holding_on" if holding_enabled else "holding_off",
        "holding_enabled": holding_enabled,
        "status": status_names.get(model.Status, str(model.Status)),
        "solution_count": model.SolCount, "runtime_seconds": model.Runtime,
        "time_limit_seconds": model.Params.TimeLimit, "target_mip_gap": model.Params.MIPGap,
        "quadratic_constraints": model.NumQConstrs,
        "holding_cost_factor": scenario["holding_cost_factor"],
        "initialization_seconds": initialization_seconds, "warm_start": list(warm_start_labels),
    }
    (output / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    if model.SolCount == 0:
        (output / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise RuntimeError(f"{result['case']}: no feasible incumbent ({result['status']}); see {output}.")
    reports, schedules = [], []
    for aircraft, flight_vars in zip(scenario["aircraft"], variables):
        destination = output / aircraft["acId"]
        destination.mkdir()
        report, data = report_aircraft(scenario, aircraft, flight_vars, fuel_table, destination)
        reports.append(report)
        schedules.append(data)
    pd.DataFrame(reports).to_csv(output / "aircraft_summary.csv", index=False)
    pd.concat([data.assign(acId=aircraft["acId"]) for aircraft, data in zip(
        scenario["aircraft"], schedules)], ignore_index=True).to_csv(output / "trajectories.csv", index=False)
    validate_separation(scenario, schedules, output)
    result.update({
        "objective": model.ObjVal, "objective_bound": model.ObjBound, "mip_gap": model.MIPGap,
        "flight_count": len(reports), "weather_regions": len(scenario["weather_boxes_uv_m"]),
    })
    for metric in ("airborne_seconds", "hold_seconds", "hold_intervals", "route_distance_km",
                   "total_distance_km", "route_fuel_kg", "hold_fuel_kg", "total_fuel_kg",
                   "total_CO2_kg", "pwl_fuel_kg", "hold_pwl_fuel_kg"):
        result[metric] = sum(report[metric] for report in reports)
    weights = scenario["objective_weights"]
    holding_cost = (
        weights["time"] * result["hold_seconds"]
        + weights["distance"] * scenario["hold_speed_mps"] * result["hold_seconds"]
        + weights["fuel"] * result["hold_pwl_fuel_kg"])
    result["holding_objective_before_discount"] = holding_cost
    result["holding_objective_discount"] = (1 - scenario["holding_cost_factor"]) * holding_cost
    result["objective_at_full_holding_cost"] = model.ObjVal + result["holding_objective_discount"]
    result["last_arrival_seconds"] = max(report["arrival_seconds"] for report in reports)
    (output / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result, schedules


def _plot_weather_timed_path(axis, times, xy, window, enabled, color, label, labels, linewidth=1.5):
    start, end = window
    cuts = sorted({float(times[0]), float(times[-1]),
                   *(boundary for boundary in window if times[0] < boundary < times[-1])})
    for left, right in zip(cuts, cuts[1:]):
        samples = np.r_[left, times[(times > left) & (times < right)], right]
        active = enabled and start <= (left + right) / 2 < end
        phase_label = label + (" active)" if active else " inactive)")
        axis.plot(np.interp(samples, times, xy[:, 0]), np.interp(samples, times, xy[:, 1]),
                  linestyle="-" if active else "--", color=color, linewidth=linewidth,
                  label=phase_label if phase_label not in labels else None)
        labels.add(phase_label)


def plot_comparison(scenario, schedules, output, aircraft_weather_regions=None):
    """Optionally style paths by an acId -> zero-based region mapping.

    Mapped paths are split at exact weather boundaries for display only; model
    constraints still protect whole intervals against every active region.
    """
    windows = weather_windows(scenario)
    if aircraft_weather_regions is not None:
        for aircraft in scenario["aircraft"]:
            region = aircraft_weather_regions.get(aircraft["acId"])
            if not isinstance(region, int) or not 0 <= region < len(windows):
                raise ValueError(f"Missing or invalid plot weather region for {aircraft['acId']}.")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    origin, basis = np.asarray(scenario["origin_xy_m"]), np.asarray(scenario["basis"])
    for axis, (name, flight_schedules) in zip(axes, schedules.items()):
        for region, (u0, v0, u1, v1) in enumerate(scenario["weather_boxes_uv_m"]):
            polygon = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ basis.T + origin
            start, end = windows[region]
            axis.add_patch(Polygon(polygon / 1000, facecolor="mistyrose", edgecolor="firebrick",
                                   label=f"W{region + 1}: {start:g}-{end:g} s" if scenario["weather_enabled"]
                                   else f"W{region + 1}: disabled"))
            axis.annotate(f"W{region + 1}", polygon.mean(axis=0) / 1000, color="firebrick")
        for index, (aircraft, data) in enumerate(zip(scenario["aircraft"], flight_schedules)):
            color = f"C{index}"
            labels = set()
            region = None if aircraft_weather_regions is None else aircraft_weather_regions[aircraft["acId"]]
            for k in range(1, len(data)):
                previous, current = data.iloc[k - 1], data.iloc[k]
                if previous.f1_status != 1 or current.f1_hold:
                    continue
                if region is None:
                    active = weather_active(previous.t, current.t, scenario)
                    label = aircraft["acId"].split("_")[0] + (" (during weather)" if active else " (outside window)")
                    axis.plot(data.f1_x.iloc[k - 1:k + 1] / 1000, data.f1_y.iloc[k - 1:k + 1] / 1000,
                              marker="o", linestyle="-" if active else "--", color=color,
                              label=label if label not in labels else None)
                    labels.add(label)
                else:
                    _plot_weather_timed_path(
                        axis, data.t.iloc[k - 1:k + 1].to_numpy(),
                        data[["f1_x", "f1_y"]].iloc[k - 1:k + 1].to_numpy() / 1000,
                        windows[region], scenario["weather_enabled"], color,
                        f"{aircraft['acId'].split('_')[0]} (W{region + 1}", labels)
                    axis.plot(data.f1_x.iloc[k - 1:k + 1] / 1000, data.f1_y.iloc[k - 1:k + 1] / 1000,
                              marker="o", linestyle="none", color=color)
            tracks = pd.read_csv(output / name / aircraft["acId"] / "hold_tracks.csv")
            for track_index, (_, track) in enumerate(tracks.groupby("end_step")):
                if region is None:
                    axis.plot(track.x / 1000, track.y / 1000, color=color, linewidth=3,
                              label=f"{aircraft['acId'].split('_')[0]} HOLD" if track_index == 0 else None)
                else:
                    _plot_weather_timed_path(
                        axis, track.t.to_numpy(), track[["x", "y"]].to_numpy() / 1000,
                        windows[region], scenario["weather_enabled"], color,
                        f"{aircraft['acId'].split('_')[0]} HOLD (W{region + 1}", labels, linewidth=3)
                axis.annotate(f"HOLD {track.t.iloc[0]:g}-{track.t.iloc[-1]:g}s",
                              (track.x.iloc[0] / 1000, track.y.iloc[0] / 1000),
                              xytext=(8, 12), textcoords="offset points", color=color, fontsize=8)
            for _, row in data[data.f1_status == 1].iterrows():
                axis.annotate(f"{row.t:g}s", (row.f1_x / 1000, row.f1_y / 1000), fontsize=7, color=color)
            axis.scatter(data.f1_x.iloc[-1] / 1000, data.f1_y.iloc[-1] / 1000, marker="*", s=100, color=color)
        axis.text(0.02, 0.98, f"Selected holds: {sum(int(data.f1_hold.sum()) for data in flight_schedules)}",
                  transform=axis.transAxes, va="top", bbox={"facecolor": "white", "alpha": 0.8})
        axis.set(title=name.replace("_", " ").upper(), xlabel="East of DTW (km)", ylabel="North of DTW (km)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("Same traffic and weather; only holding availability changes. Times since first entry."
                   + ("\nSolid = corresponding weather active; dashed = corresponding weather inactive."
                      if aircraft_weather_regions is not None else ""))
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)


def animation_track(data, hold_tracks):
    """Expand closed holds into flown samples and stop at the first STAR arrival."""
    arrivals = np.flatnonzero(data.f1_status.to_numpy() == 2)
    if not len(arrivals):
        raise ValueError("Animation requires a recorded STAR arrival.")
    first = data.iloc[0]
    points = [[first.t, first.f1_x, first.f1_y]]
    holds = []
    for k in range(1, int(arrivals[0]) + 1):
        previous, current = data.iloc[k - 1], data.iloc[k]
        if current.f1_hold:
            circuit = hold_tracks.loc[hold_tracks.end_step == k, ["t", "x", "y"]].to_numpy()
            if (len(circuit) < 2
                    or not np.allclose(circuit[[0, -1], 0], [previous.t, current.t], rtol=0, atol=0.01)
                    or not np.allclose(circuit[[0, -1], 1:],
                                       [[previous.f1_x, previous.f1_y], [current.f1_x, current.f1_y]],
                                       rtol=0, atol=0.01)):
                raise ValueError(f"Missing or inconsistent holding track for interval {k}.")
            points.extend(circuit[1:].tolist())
            holds.append((previous.t, current.t))
        else:
            points.append([current.t, current.f1_x, current.f1_y])
    track = np.asarray(points, dtype=float)
    if not np.isfinite(track).all() or np.any(np.diff(track[:, 0]) <= 0):
        raise ValueError("Animation track must be finite with strictly increasing times.")
    return track, holds


def animate_comparison(output, frame_step_seconds=10.0, fps=8, dpi=100):
    """Write comparison.gif from saved OFF/ON results, without running optimization.

    Transit is interpolated between waypoints; holds use saved racetrack samples.
    Weather shading uses each region's instantaneous [start, end) active window.
    """
    if not np.isfinite(frame_step_seconds) or frame_step_seconds <= 0 or fps <= 0 or dpi <= 0:
        raise ValueError("Animation frame step, FPS and DPI must be positive.")
    output = Path(output)
    scenario = json.loads((output / "scenario.json").read_text(encoding="utf-8"))
    windows = weather_windows(scenario)
    origin, basis = np.asarray(scenario["origin_xy_m"]), np.asarray(scenario["basis"])
    polygons = [
        (np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]]) @ basis.T + origin) / 1000
        for u0, v0, u1, v1 in scenario["weather_boxes_uv_m"]
    ]
    cases = {}
    for case in ("holding_off", "holding_on"):
        flights = []
        for aircraft in scenario["aircraft"]:
            folder = output / case / aircraft["acId"]
            track, holds = animation_track(
                pd.read_csv(folder / "df_wide.csv"), pd.read_csv(folder / "hold_tracks.csv"))
            flights.append((aircraft["acId"].split("_")[0], track, holds))
        cases[case] = flights
    end_time = max(track[-1, 0] for flights in cases.values() for _, track, _ in flights)
    times = np.r_[np.arange(0, end_time, frame_step_seconds), end_time]
    coordinates = np.vstack(polygons + [track[:, 1:] / 1000
                                        for flights in cases.values() for _, track, _ in flights])
    lower, upper = coordinates.min(axis=0) - 25, coordinates.max(axis=0) + 25
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    panels = []
    for axis, (case, flights) in zip(axes, cases.items()):
        patches, labels, artists = [], [], []
        for region, polygon in enumerate(polygons):
            patch = Polygon(polygon, linewidth=1.5)
            axis.add_patch(patch)
            patches.append(patch)
            labels.append(axis.text(*polygon.mean(axis=0), "", ha="center", fontsize=8))
        for i, (name, track, _) in enumerate(flights):
            trail, = axis.plot([], [], color=f"C{i}", linewidth=1.8, label=name)
            marker, = axis.plot([], [], color=f"C{i}", marker="o", markersize=8,
                                markeredgecolor="white", linestyle="none", zorder=5)
            annotation = axis.annotate("", (0, 0), xytext=(7, 7), textcoords="offset points",
                                       color=f"C{i}", fontsize=9, fontweight="bold")
            axis.scatter(*track[-1, 1:] / 1000, marker="*", s=90, color=f"C{i}", alpha=0.3)
            artists.append((trail, marker, annotation))
        status = axis.text(0.02, 0.02, "", transform=axis.transAxes, fontsize=9,
                           va="bottom", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"})
        axis.set(title=case.replace("_", " ").upper(), xlabel="East of DTW (km)",
                 ylabel="North of DTW (km)", xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right", fontsize=8)
        panels.append((patches, labels, artists, status))
    clock = figure.suptitle("")
    figure.text(0.5, 0.06, "Pink fill = ACTIVE weather   |   Dashed outline = INACTIVE footprint\n"
                "Trails show past movement; HOLD follows the saved circuit; stars mark destination fixes.",
                ha="center", fontsize=10)
    figure.subplots_adjust(top=0.85, bottom=0.16, left=0.06, right=0.98, wspace=0.12)
    holding_cost_factor = scenario.get("holding_cost_factor")
    factor_label = (
        f"holding cost factor {holding_cost_factor:g}"
        if holding_cost_factor is not None
        else "holding cost factor not recorded"
    )

    def draw_frame(time_s):
        clock.set_text(f"Holding OFF vs ON | {time_s:g} s since first entry\n"
                       f"Weather and aircraft shown on the same clock | {factor_label}")
        for flights, (patches, labels, artists, status) in zip(cases.values(), panels):
            for region, ((start, end), patch, label) in enumerate(zip(windows, patches, labels)):
                active = scenario["weather_enabled"] and start <= time_s < end
                patch.set_facecolor("mistyrose" if active else "none")
                patch.set_edgecolor("firebrick" if active else "0.65")
                patch.set_linestyle("-" if active else "--")
                label.set_color("firebrick" if active else "0.5")
                state = "ACTIVE" if active else "inactive"
                label.set_text(f"W{region + 1}: {start:g}-{end:g} s\n{state}")
            statuses = []
            for (name, track, holds), (trail, marker, annotation) in zip(flights, artists):
                entered = time_s >= track[0, 0]
                marker.set_visible(entered)
                annotation.set_visible(entered)
                if not entered:
                    trail.set_data([], [])
                    statuses.append(f"{name}: not entered")
                    continue
                current = np.array([np.interp(time_s, track[:, 0], track[:, axis]) for axis in (1, 2)]) / 1000
                past = track[track[:, 0] <= time_s, 1:] / 1000
                trail.set_data(np.r_[past[:, 0], current[0]], np.r_[past[:, 1], current[1]])
                marker.set_data([current[0]], [current[1]])
                holding = any(start <= time_s < end for start, end in holds)
                arrived = time_s >= track[-1, 0]
                state = "HOLD" if holding else "at STAR" if arrived else "transit"
                marker.set_marker("*" if arrived else "D" if holding else "o")
                annotation.xy = tuple(current)
                annotation.set_text(name + (" HOLD" if holding else ""))
                statuses.append(f"{name}: {state}")
            status.set_text("\n".join(statuses))
        return []

    destination = output / "comparison.gif"
    try:
        animation = FuncAnimation(figure, draw_frame, frames=times, interval=1000 / fps, repeat=True)
        animation.save(str(destination), writer="pillow", fps=fps, dpi=dpi)
    finally:
        plt.close(figure)
    return destination


def write_comparison_output(scenario, results, schedules, output):
    comparison = pd.DataFrame(results)
    comparison.to_csv(output / "comparison.csv", index=False)
    pd.concat([pd.read_csv(output / case / "aircraft_summary.csv").assign(case=case)
               for case in schedules], ignore_index=True).to_csv(output / "aircraft_comparison.csv", index=False)
    off, on = results
    metrics = ["objective", "airborne_seconds", "hold_seconds", "total_distance_km",
               "total_fuel_kg", "total_CO2_kg", "pwl_fuel_kg"]
    deltas = {metric: on[metric] - off[metric] for metric in metrics}
    summary = {
        "scenario": scenario, "cases": results, "holding_on_minus_off": deltas,
        "fuel_reduction_percent": 100 * (off["total_fuel_kg"] - on["total_fuel_kg"]) / off["total_fuel_kg"],
        "interpretation": "Feasible incumbent comparison, not a proven optimal saving. Holding ON may choose zero holds.",
    }
    if on["hold_intervals"] == 0:
        summary["interpretation"] += (
            " No holds were selected: differences reflect solver incumbents, not demonstrated holding benefits.")
    if scenario["holding_cost_factor"] != 1:
        summary["interpretation"] += (
            " Holding objective costs are scaled; reported physical time, distance, fuel and emissions are not.")
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_comparison(scenario, schedules, output)
    print("\n=== PAIRED COMPARISON ===")
    print(comparison[["case", "status", "mip_gap", "airborne_seconds", "hold_seconds",
                      "total_distance_km", "total_fuel_kg", "objective"]].to_string(index=False))
    print(f"\nHolding ON minus OFF: fuel {deltas['total_fuel_kg']:+.2f} kg; "
          f"airborne time {deltas['airborne_seconds']:+.0f} s; distance {deltas['total_distance_km']:+.2f} km.")
    print(summary["interpretation"])
    print(f"Results: {output.resolve()}")
    return summary
