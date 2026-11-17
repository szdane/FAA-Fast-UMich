"""Single-aircraft/single-storm holding-cost sensitivity, with one shared OFF baseline."""

import argparse
import json
import math
from pathlib import Path

import gurobipy as gp
import pandas as pd

from main_v18 import SCRIPT_DIR, FuelFlowTable, load_scenario, solve_case
import matplotlib.pyplot as plt


def run_sweep(factors, time_limit=60.0, output_dir=None):
    factors = list(factors)
    if not factors or any(not math.isfinite(factor) or factor < 0 for factor in factors):
        raise ValueError("Supply at least one finite, nonnegative holding-cost factor.")
    if len(set(factors)) != len(factors):
        raise ValueError("Holding-cost factors must be distinct.")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("The per-case time limit must be finite and positive.")
    output = Path(output_dir) if output_dir is not None else (
        SCRIPT_DIR / "Output" / "v18_holding_penalty_sweep" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    scenario = load_scenario()
    (output / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    table = FuelFlowTable(scenario["acType"], scenario["mass_fraction"])
    rows = []
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        baseline, _, seed = solve_case(scenario, table, False, time_limit, output / "holding_off", env)
        for index, factor in enumerate(factors):
            scenario = load_scenario(holding_cost_factor=factor)
            folder = output / f"factor_{index:02d}_{factor:g}"
            print(f"\n=== OPTIONAL HOLDING: COST FACTOR {factor:g} ===", flush=True)
            result, _, _ = solve_case(scenario, table, True, time_limit, folder, env, seed)
            (folder / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
            intervals = pd.read_csv(folder / scenario["aircraft"][0]["acId"] / "holding_intervals.csv")
            row = dict(result)
            row.update({
                "output_directory": folder.name,
                "hold_intervals_s": "; ".join(f"{start:g}-{end:g}" for start, end in
                                             zip(intervals.start_t, intervals.end_t)),
                "objective_minus_off_incumbent": result["objective"] - baseline["objective"],
                "objective_minus_off_bound": result["objective"] - baseline["objective_bound"],
                "holding_beats_all_no_hold_solutions": bool(
                    result["hold_intervals"] > 0 and result["objective"] < baseline["objective_bound"] - 0.01),
            })
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "penalty_sweep.csv", index=False)
    frame = pd.DataFrame(rows).sort_values("holding_cost_factor")
    summary = {
        "baseline": baseline, "results": rows,
        "interpretation": (
            "Only the holding time/distance/fuel objective contributions are multiplied by the factor. "
            "Physical costs and constraints are unchanged. Every optional solve uses the same OFF seed, "
            "not the previous factor's incumbent. Selection thresholds are empirical unless bounded: "
            "a held objective below the OFF lower bound proves no zero-hold solution can be optimal at that factor. "
            "Overlapping bounds do not prove absence of a better holding or zero-hold solution."),
    }
    (output / "penalty_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(frame.holding_cost_factor, frame.hold_seconds, "o-")
    axes[0].set(ylabel="Selected holding (s)", title="Optional holding activation")
    axes[1].plot(frame.holding_cost_factor, frame.total_fuel_kg, "o-", label="Holding allowed")
    axes[1].axhline(baseline["total_fuel_kg"], color="gray", linestyle="--", label="OFF incumbent")
    axes[1].set(ylabel="Actual OpenAP fuel (kg)", title="Fuel is NOT discounted")
    axes[1].legend()
    axes[2].plot(frame.holding_cost_factor, frame.objective, "o-", label="Optional incumbent")
    axes[2].plot(frame.holding_cost_factor, frame.objective_bound, ":", label="Optional lower bound")
    axes[2].axhline(baseline["objective_bound"], color="gray", linestyle="--", label="OFF lower bound")
    axes[2].set(ylabel="Weighted objective", title="Solver evidence")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Holding cost factor (1 = original)")
        axis.grid(alpha=0.3)
    figure.suptitle("Same aircraft and weather (400-800 s); holding is never forced")
    figure.tight_layout()
    figure.savefig(output / "penalty_sweep.png", dpi=160)
    plt.close(figure)
    print(frame[["holding_cost_factor", "hold_seconds", "hold_intervals_s", "total_fuel_kg",
                 "objective", "mip_gap", "holding_beats_all_no_hold_solutions"]].to_string(index=False))
    print(f"Results: {output.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factors", type=float, nargs="+", default=[1, 0.75, 0.5, 0.25, 0])
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive.")
    if any(not math.isfinite(factor) or factor < 0 for factor in args.factors):
        parser.error("--factors must be finite and nonnegative.")
    if len(set(args.factors)) != len(args.factors):
        parser.error("--factors must be distinct.")
    run_sweep(args.factors, args.time_limit, args.output_dir)


if __name__ == "__main__":
    main()
