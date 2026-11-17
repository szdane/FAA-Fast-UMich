"""Run unchanged v16.2 optimization settings and compare MILP/NLP at matching times.

Usage:
    python analyze_v162_trajectory_distance.py
    python analyze_v162_trajectory_distance.py --reuse <previous-run-directory>

Outputs are isolated under Output/v162_distance/<timestamp>. Both absolute UTC
and elapsed-since-start alignment are reported, without time stretching, nearest
path matching, extrapolation, or holding an aircraft stationary after arrival.
Distances use DTW-centred projected east/north and MSL altitude, all in metres.
The MILP is linearly interpolated between active waypoints including arrival;
the NLP is linearly interpolated between its exported trajectory samples.
Midpoint samples cover equal-duration cells no longer than --step seconds.
Statistics/ECDFs are duration-weighted; overall_equal_flight instead gives every
flight equal total weight. Maxima are exact for these piecewise-linear curves,
evaluated at the union of their time knots, not just the midpoint samples.
The combined GIF uses a common UTC clock (up to 10 simulated seconds per frame,
10 frames per second), circles/solid trails for MILP and triangles/dashed trails
for NLP. Completed trails remain visible; moving markers only exist in flight.
"""

import argparse
import json
import math
import os
from pathlib import Path
import runpy

os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = ("distance_3d_m", "horizontal_m", "vertical_abs_m")


def validate_curve(times, positions):
    if (len(times) < 2 or positions.shape != (len(times), 3)
            or not np.isfinite(times).all() or not np.isfinite(positions).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Trajectory requires finite XYZ and strictly increasing times (>=2 rows).")


def distances_at(times, milp_t, milp_xyz, nlp_t, nlp_xyz):
    milp = np.column_stack([np.interp(times, milp_t, milp_xyz[:, i]) for i in range(3)])
    nlp = np.column_stack([np.interp(times, nlp_t, nlp_xyz[:, i]) for i in range(3)])
    delta = nlp - milp
    return milp, nlp, np.column_stack((
        np.linalg.norm(delta, axis=1),
        np.linalg.norm(delta[:, :2], axis=1),
        np.abs(delta[:, 2]),
    ))


def compare_curves(milp_t, milp_xyz, nlp_t, nlp_xyz, step):
    validate_curve(milp_t, milp_xyz)
    validate_curve(nlp_t, nlp_xyz)
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Sampling step must be finite and positive.")
    start = max(milp_t[0], nlp_t[0])
    end = min(milp_t[-1], nlp_t[-1])
    if end <= start:
        raise ValueError("MILP and NLP have no positive-duration overlapping interval.")
    count = math.ceil((end - start) / step)
    width = (end - start) / count
    times = start + (np.arange(count) + 0.5) * width
    milp, nlp, distances = distances_at(times, milp_t, milp_xyz, nlp_t, nlp_xyz)
    frame = pd.DataFrame(distances, columns=METRICS)
    frame.insert(0, "time_s", times)
    frame["weight_s"] = width
    for i, axis in enumerate("xyz"):
        frame[f"milp_{axis}_m"] = milp[:, i]
        frame[f"nlp_{axis}_m"] = nlp[:, i]
        frame[f"delta_{axis}_m"] = nlp[:, i] - milp[:, i]
    knots = np.unique(np.concatenate(([start, end], milp_t, nlp_t)))
    knots = knots[(knots >= start) & (knots <= end)]
    _, _, knot_distances = distances_at(knots, milp_t, milp_xyz, nlp_t, nlp_xyz)
    peaks = {}
    for i, metric in enumerate(METRICS):
        index = int(np.argmax(knot_distances[:, i]))
        peaks[metric] = {"max_m": float(knot_distances[index, i]),
                         "max_time_s": float(knots[index])}
    coverage = {
        "overlap_start_s": float(start), "overlap_end_s": float(end),
        "overlap_s": float(end - start),
        "milp_duration_s": float(milp_t[-1] - milp_t[0]),
        "nlp_duration_s": float(nlp_t[-1] - nlp_t[0]),
        "nlp_start_minus_milp_start_s": float(nlp_t[0] - milp_t[0]),
        "milp_coverage_pct": float(100 * (end - start) / (milp_t[-1] - milp_t[0])),
        "nlp_coverage_pct": float(100 * (end - start) / (nlp_t[-1] - nlp_t[0])),
    }
    return frame, peaks, coverage


def statistics(values, weights):
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / weights.sum()
    quantiles = values[order][np.minimum(
        np.searchsorted(cdf, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]), len(values) - 1)]
    mean = float(np.average(values, weights=weights))
    result = {
        "sample_count": len(values), "mean_m": mean,
        "std_m": float(np.sqrt(np.average((values - mean) ** 2, weights=weights))),
        "rms_m": float(np.sqrt(np.average(values ** 2, weights=weights))),
        "sample_min_m": float(values.min()),
    }
    result.update(zip(("p05_m", "p25_m", "median_m", "p75_m", "p90_m", "p95_m", "p99_m"),
                      map(float, quantiles)))
    for threshold in (100, 500, 1000, 5000, 10000, 20000):
        result[f"pct_above_{threshold}m"] = float(
            100 * np.average(values > threshold, weights=weights))
    return result


def run_model(output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    previous = os.environ.get("V162_OUTPUT_DIR")
    os.environ["V162_OUTPUT_DIR"] = str(output)
    try:
        state = runpy.run_path(str(SCRIPT_DIR / "main_v16.2.py"), run_name="__main__")
    finally:
        if previous is None:
            os.environ.pop("V162_OUTPUT_DIR", None)
        else:
            os.environ["V162_OUTPUT_DIR"] = previous
    if "results" not in state:
        raise RuntimeError(f"v16.2 did not produce NLP results; Gurobi status={state['m'].Status}")
    results = state["results"]
    info = results["nlp_info"]
    metadata = {
        "source": "main_v16.2.py", "grid_epoch_utc": str(state["GRID_EPOCH_UTC"]),
        "flights": [ac["acId"] for ac in state["aircraft_list"]],
        "milp_status": int(state["m"].Status), "milp_gap": float(state["m"].MIPGap),
        "milp_runtime_s": float(state["m"].Runtime),
        "milp_objective": float(state["m"].ObjVal), "nlp_info": info,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for acid, trajectory in results["optimized"].items():
        trajectory.to_csv(output / f"nlp_{acid}.csv", index=False)
    return metadata


def load_curves(output, metadata):
    wide = pd.read_csv(output / "df_wide.csv")
    epoch = pd.to_datetime(metadata["grid_epoch_utc"], utc=True)
    curves = {}
    for index, acid in enumerate(metadata["flights"], start=1):
        status = wide[f"f{index}_status"]
        active = pd.concat([wide[status == 1], wide[status == 2].head(1)])
        if len(active) < 2:
            raise ValueError(f"{acid}: fewer than two active MILP waypoints.")
        mt = active["t"].to_numpy(float)
        mx = active[[f"f{index}_{axis}" for axis in "xyz"]].to_numpy(float)
        nlp = pd.read_csv(output / f"nlp_{acid}.csv")
        nt = (pd.to_datetime(nlp["recTime"], utc=True) - epoch).dt.total_seconds().to_numpy()
        nx = nlp[["x", "y", "z"]].to_numpy(float)
        validate_curve(mt, mx)
        validate_curve(nt, nx)
        curves[acid] = (mt, mx, nt, nx)
    return curves, epoch


def analyze(output, metadata, step):
    failed = [row for row in metadata["nlp_info"]
              if row["status"] not in ("Solve_Succeeded", "Solved_To_Acceptable_Level")]
    if failed:
        raise RuntimeError(f"Refusing to label unconverged NLP trajectories as valid: {failed}")
    curves, epoch = load_curves(output, metadata)
    samples, coverages, peak_rows = [], [], []
    for acid, (mt, mx, nt_utc, nx) in curves.items():
        for alignment in ("utc", "start_aligned"):
            nt = nt_utc if alignment == "utc" else nt_utc - nt_utc[0]
            milp_t = mt if alignment == "utc" else mt - mt[0]
            frame, peaks, coverage = compare_curves(milp_t, mx, nt, nx, step)
            frame.insert(0, "acId", acid)
            frame.insert(0, "alignment", alignment)
            if alignment == "utc":
                frame["time_utc"] = epoch + pd.to_timedelta(frame["time_s"], unit="s")
            samples.append(frame)
            coverages.append({"alignment": alignment, "acId": acid, **coverage})
            for metric, peak in peaks.items():
                peak_rows.append({"alignment": alignment, "acId": acid, "metric": metric, **peak})
    data = pd.concat(samples, ignore_index=True)
    peaks = pd.DataFrame(peak_rows)
    summary = []
    for alignment, group in data.groupby("alignment", sort=False):
        groups = list(group.groupby("acId", sort=False))
        groups += [("overall_time_weighted", group), ("overall_equal_flight", group)]
        for acid, frame in groups:
            weights = frame["weight_s"].to_numpy().copy()
            if acid == "overall_equal_flight":
                weights /= frame.groupby("acId")["weight_s"].transform("sum").to_numpy()
            for metric in METRICS:
                candidates = peaks[(peaks.alignment == alignment) & (peaks.metric == metric)]
                if not acid.startswith("overall_"):
                    candidates = candidates[candidates.acId == acid]
                peak = candidates.loc[candidates.max_m.idxmax()]
                summary.append({
                    "alignment": alignment, "acId": acid, "metric": metric,
                    **statistics(frame[metric].to_numpy(), weights),
                    "max_m": peak.max_m, "max_flight": peak.acId, "max_time_s": peak.max_time_s,
                })
        plot_distributions(group, alignment, output)
    result = pd.DataFrame(summary)
    data.to_csv(output / "distance_samples.csv", index=False)
    result.to_csv(output / "distance_summary.csv", index=False)
    pd.DataFrame(coverages).to_csv(output / "time_coverage.csv", index=False)
    peaks.to_csv(output / "distance_peaks.csv", index=False)
    utc_coverage = [row for row in coverages if row["alignment"] == "utc"]
    offsets = [row["nlp_start_minus_milp_start_s"] for row in utc_coverage]
    interpretation = [
        f"NLP entry minus MILP entry ranges from {min(offsets):g} to {max(offsets):g} s.",
        "UTC discrepancies include different entry times, speed profiles, and routes.",
        "Start alignment removes entry offsets but preserves different speeds and durations;",
        "neither comparison measures nearest-path geometric distance.",
    ]
    for alignment in ("utc", "start_aligned"):
        rows = result[(result.alignment == alignment) & (result.metric == "distance_3d_m")]
        overall = rows[rows.acId == "overall_time_weighted"].iloc[0]
        individual = rows[~rows.acId.str.startswith("overall_")]
        largest = individual.loc[individual.mean_m.idxmax()]
        smallest = individual.loc[individual.mean_m.idxmin()]
        interpretation.append(
            f"{alignment}: overall mean {overall.mean_m / 1000:.3f} km, "
            f"median {overall.median_m / 1000:.3f} km, p95 {overall.p95_m / 1000:.3f} km; "
            f"{overall.pct_above_10000m:.2f}% of overlapping flight-time exceeds 10 km. "
            f"Largest flight mean: {largest.acId} ({largest.mean_m / 1000:.3f} km); "
            f"smallest: {smallest.acId} ({smallest.mean_m / 1000:.3f} km).")
    report = (
        __doc__ + f"\nRequested maximum sample interval: {step:g} s\n"
        f"Grid epoch: {epoch}\nMILP gap: {metadata['milp_gap']:.6%}\n"
        "UTC time_s is seconds since grid epoch; start_aligned time_s is elapsed flight time.\n"
        "Unmatched pre-entry/post-arrival intervals are excluded, not assigned zero distance.\n"
        "Overall_time_weighted pools flight-seconds (longer overlaps weigh more).\n"
        "Overall_equal_flight is a mixture giving every flight equal probability.\n"
        "Quantiles/minima are sampled approximations; maxima are exact for interpolated curves.\n"
        "These are same-flight model discrepancies, NOT inter-aircraft separation distances.\n\n"
        + "\n".join(interpretation) + "\n\n"
        + result[result.metric == "distance_3d_m"].to_string(index=False)
        + "\n\nCoverage:\n" + pd.DataFrame(coverages).to_string(index=False)
        + "\n\nNLP solver status:\n" + pd.DataFrame(metadata["nlp_info"]).to_string(index=False)
    )
    (output / "distance_analysis.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved analysis: {output}", flush=True)
    return result


def animate_comparison(output, metadata):
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from Functions.utilities import load_star_fixes_xyz

    curves, epoch = load_curves(output, metadata)
    fixes, _, _, _, _ = load_star_fixes_xyz(
        SCRIPT_DIR / "Input" / "star_fixes.csv", 42.2125, -83.3534)
    colors = ["#E42320", "#2171B5", "#2CA25F", "#F16913", "#7B2D8B", "#D4AC0D"]
    start = min(min(mt[0], nt[0]) for mt, _, nt, _ in curves.values())
    end = max(max(mt[-1], nt[-1]) for mt, _, nt, _ in curves.values())
    # Include exact entries/arrivals as well as regular frames to show every transition.
    events = [t for mt, _, nt, _ in curves.values() for t in (mt[0], mt[-1], nt[0], nt[-1])]
    frames = np.unique(np.concatenate((np.arange(start, end, 10.), events, [end])))
    fig, ax = plt.subplots(figsize=(10, 10))
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(333 * np.cos(theta), 333 * np.sin(theta), "g--", lw=.8, alpha=.35)
    order = ["CRAKN", "BONZZ", "FERRL", "LAYKS", "GRAYT", "RKCTY", "HAYLL", "HANBL"]
    polygon = np.array([fixes[name][:2] for name in order if name in fixes]) / 1000
    polygon = np.vstack((polygon, polygon[0]))
    ax.plot(polygon[:, 0], polygon[:, 1], color="#018B38", lw=1.5, alpha=.7)
    ax.scatter([0], [0], color="#018B38", s=120, marker="*", zorder=10)
    ax.text(3, 3, "DTW", color="#018B38", fontsize=9)
    for name, fix in fixes.items():
        ax.scatter(fix[0] / 1000, fix[1] / 1000, color="darkgreen", marker="^", s=30)
        ax.text(fix[0] / 1000 + 3, fix[1] / 1000 + 3, name, color="darkgreen", fontsize=7)
    artists = []
    handles = []
    for index, (acid, (mt, mx, nt, nx)) in enumerate(curves.items()):
        color = colors[index % len(colors)]
        label = acid.split("_")[0]
        handles.append(Line2D([], [], color=color, label=label))
        for model, times, xyz, style, marker in (
                ("MILP", mt, mx, "-", "o"), ("NLP", nt, nx, "--", "^")):
            ax.plot(xyz[:, 0] / 1000, xyz[:, 1] / 1000, style, color=color, alpha=.12, lw=1)
            trail, = ax.plot([], [], style, color=color, lw=1.8)
            point, = ax.plot([], [], marker, color=color, markersize=8,
                             markeredgecolor="white", markeredgewidth=.5)
            artists.append((model, times, xyz, trail, point))
    flight_legend = ax.legend(handles=handles, loc="lower left", fontsize=8)
    ax.add_artist(flight_legend)
    ax.legend(handles=[
        Line2D([], [], color="black", marker="o", linestyle="-", label="MILP"),
        Line2D([], [], color="black", marker="^", linestyle="--", label="NLP"),
    ], loc="lower right", fontsize=9)
    ax.set(xlabel="East (km)", ylabel="North (km)", aspect="equal")
    ax.grid(True, linestyle=":", alpha=.25)
    ax.margins(.07)
    ax.autoscale_view()
    ax.set_autoscale_on(False)
    status = ax.text(.02, .98, "", transform=ax.transAxes, va="top", fontsize=9,
                     bbox=dict(boxstyle="round", facecolor="wheat", alpha=.7))

    def draw(time_s):
        counts = {"MILP": [0, 0], "NLP": [0, 0]}
        for model, times, xyz, trail, point in artists:
            point.set_data([], [])
            if time_s < times[0]:
                trail.set_data([], [])
                continue
            until = min(time_s, times[-1])
            current = np.array([np.interp(until, times, xyz[:, i]) for i in range(2)])
            past = xyz[times < until, :2]
            path = np.vstack((past, current)) / 1000
            trail.set_data(path[:, 0], path[:, 1])
            if time_s <= times[-1]:
                point.set_data([current[0] / 1000], [current[1] / 1000])
                counts[model][0] += 1
            else:
                counts[model][1] += 1
        utc = epoch + pd.Timedelta(seconds=float(time_s))
        ax.set_title(f"MILP and NLP trajectories | {utc.strftime('%H:%M:%S')} UTC\n"
                     "Same UTC clock; no time stretching", fontsize=11, fontweight="bold")
        status.set_text("\n".join(f"{model}: flying {count[0]}, completed {count[1]}"
                                  for model, count in counts.items()))

    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    draw(float(frames[len(frames) // 2]))
    fig.tight_layout()
    fig.savefig(plots / "milp_nlp_animation_preview.png", dpi=100)
    animation = FuncAnimation(fig, draw, frames=frames, interval=100, repeat=True)
    animation.save(str(plots / "milp_nlp_animation.gif"), writer=PillowWriter(fps=10), dpi=90)
    plt.close(fig)
    print(f"Combined MILP/NLP animation saved ({len(frames)} frames).", flush=True)


def plot_distributions(data, alignment, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for acid, frame in data.groupby("acId", sort=False):
        label = acid.split("_")[0]
        values = frame["distance_3d_m"].to_numpy() / 1000
        order = np.argsort(values)
        cdf = np.cumsum(frame["weight_s"].to_numpy()[order]) / frame["weight_s"].sum()
        axes[0, 0].plot(frame.time_s, values, label=label)
        axes[0, 1].step(values[order], cdf, where="post", label=label)
        axes[1, 0].plot(frame.time_s, frame.horizontal_m / 1000, label=label)
        axes[1, 1].plot(frame.time_s, frame.vertical_abs_m, label=label)
    values = data.distance_3d_m.to_numpy() / 1000
    order = np.argsort(values)
    axes[0, 1].step(values[order], np.cumsum(data.weight_s.to_numpy()[order]) / data.weight_s.sum(),
                    color="black", linestyle="--", label="Overall (time-weighted)", where="post")
    time_label = "Seconds since grid epoch" if alignment == "utc" else "Seconds since flight start"
    axes[0, 0].set(xlabel=time_label, ylabel="3D distance (km)")
    axes[0, 1].set(xlabel="3D distance (km)", ylabel="Cumulative probability")
    axes[1, 0].set(xlabel=time_label, ylabel="Horizontal distance (km)")
    axes[1, 1].set(xlabel=time_label, ylabel="Absolute vertical distance (m)")
    for axis in axes.flat:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle(f"v16.2 MILP vs NLP: {alignment}; overlapping flight time only")
    fig.tight_layout()
    fig.savefig(output / f"distance_distribution_{alignment}.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", type=Path, help="Reanalyze saved raw trajectories without solving.")
    parser.add_argument("--step", type=float, default=1.0, help="Maximum sampling interval in seconds.")
    args = parser.parse_args()
    if not math.isfinite(args.step) or args.step <= 0:
        parser.error("--step must be finite and positive.")
    if args.reuse:
        output = args.reuse.resolve()
        metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    else:
        output = SCRIPT_DIR / "Output" / "v162_distance" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
        print(f"Run output: {output}", flush=True)
        metadata = run_model(output)
    analyze(output, metadata, args.step)
    animate_comparison(output, metadata)


if __name__ == "__main__":
    main()
