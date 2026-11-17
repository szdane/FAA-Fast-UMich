"""Sequential, objective-only distance/heading ablations of main_v16.2_test.py."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import gurobipy as gp


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_SCRIPT = SCRIPT_DIR / "main_v16.2_test.py"
BUILD_BOUNDARY = "# 2.6. Initiate optimization"
CASES = {
    "baseline": (True, True),
    "no_distance": (False, True),
    "no_heading": (True, False),
    "no_both": (False, False),
}
STATUS_NAMES = {
    2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
    6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    15: "USER_OBJ_LIMIT", 16: "WORK_LIMIT", 17: "MEM_LIMIT",
}


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model(model, folder):
    # Gurobi's compressed writer is unavailable in some Windows installations.
    path = folder / "model.mps"
    model.write(str(path))
    with path.open("rb") as source, gzip.open(folder / "model.mps.gz", "wb") as target:
        shutil.copyfileobj(source, target)
    path.unlink()


def source_hashes():
    files = [MODEL_SCRIPT, Path(__file__).resolve(), *sorted((SCRIPT_DIR / "Functions").glob("*.py")),
             SCRIPT_DIR / "Input" / "entry_exit_points.csv",
             SCRIPT_DIR / "Input" / "star_fixes.csv"]
    return {str(path.relative_to(SCRIPT_DIR)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


@contextmanager
def built_model():
    # Use the same explicit build boundary as the procedural v18.1 test fixtures.
    source = MODEL_SCRIPT.read_text(encoding="utf-8")
    if source.count(BUILD_BOUNDARY) != 1:
        raise RuntimeError("Expected exactly one model-building boundary; refusing a full script run.")
    namespace = {"__file__": str(MODEL_SCRIPT), "__name__": "v162_ablation_model"}
    try:
        exec(compile(source.partition(BUILD_BOUNDARY)[0], str(MODEL_SCRIPT), "exec"), namespace)
        model = namespace["m"]
        model.setObjective(namespace["obj"], gp.GRB.MINIMIZE)
        model.update()
        if model.NumQNZs or model.NumQConstrs or not model.IsMIP:
            raise RuntimeError("The experiment requires the original linear mixed-integer model.")
        yield namespace
    finally:
        if "m" in namespace:
            namespace["m"].dispose()
        if "env" in namespace:
            namespace["env"].dispose()


def objective_groups(namespace):
    distance, heading = [], []
    for i in range(namespace["N_flights"]):
        entry = int(namespace["flights"].iloc[i]["flight_entry_timestep"])
        for k in range(entry + 1, namespace["N_steps"]):
            distance.append(namespace["speed_2d"][i, k])
            if k >= entry + 2:
                heading.append(namespace["accel_cost"][i, k])
    return distance, heading


def zero_objective_fingerprint(model):
    objective = model.getObjective()
    model.setObjective(gp.LinExpr(), gp.GRB.MINIMIZE)
    model.update()
    fingerprint = model.Fingerprint
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.update()
    return f"{fingerprint & 0xffffffff:08x}"


def apply_case(namespace, case):
    groups = objective_groups(namespace)
    for variables, keep in zip(groups, CASES[case]):
        if not keep:
            namespace["m"].setAttr("Obj", variables, [0.0] * len(variables))
    namespace["m"].update()
    return groups


def finite(value):
    return float(value) if math.isfinite(value) and abs(value) < gp.GRB.INFINITY else None


def relative_gap(incumbent, bound):
    if incumbent is None or bound is None:
        return None
    if incumbent == bound:
        return 0.0
    return abs(incumbent - bound) / abs(incumbent) if incumbent else None


class SolverTrace:
    def __init__(self, stream):
        self.stream = stream
        self.writer = csv.DictWriter(stream, fieldnames=[
            "event", "runtime_s", "work", "incumbent", "bound", "relative_gap", "nodes",
        ])
        self.writer.writeheader()
        self.first_incumbent_s = None
        self.first_incumbent_objective = None
        self.last_improvement_s = None
        self.best = math.inf
        self.last_sample_s = -math.inf

    def __call__(self, model, where):
        if where not in (gp.GRB.Callback.MIP, gp.GRB.Callback.MIPSOL):
            return
        elapsed = model.cbGet(gp.GRB.Callback.RUNTIME)
        if where == gp.GRB.Callback.MIP:
            if elapsed - self.last_sample_s < 5.0:
                return
            incumbent = finite(model.cbGet(gp.GRB.Callback.MIP_OBJBST))
            bound = finite(model.cbGet(gp.GRB.Callback.MIP_OBJBND))
            nodes = model.cbGet(gp.GRB.Callback.MIP_NODCNT)
            event = "progress"
            self.last_sample_s = elapsed
        else:
            candidate = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            incumbent = finite(min(candidate, model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)))
            bound = finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))
            nodes = model.cbGet(gp.GRB.Callback.MIPSOL_NODCNT)
            event = "solution"
            if self.first_incumbent_s is None:
                self.first_incumbent_s = elapsed
                self.first_incumbent_objective = finite(candidate)
            if candidate < self.best - 1e-7:
                self.best = candidate
                self.last_improvement_s = elapsed
        self.writer.writerow({
            "event": event, "runtime_s": elapsed, "work": model.cbGet(gp.GRB.Callback.WORK),
            "incumbent": incumbent, "bound": bound,
            "relative_gap": relative_gap(incumbent, bound), "nodes": nodes,
        })
        self.stream.flush()


def outcome(status, solutions, gap, runtime, acceptable_seconds, quality_ok):
    if status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT):
        return STATUS_NAMES.get(status, f"STATUS_{status}")
    if not solutions:
        return "NO_INCUMBENT"
    if not quality_ok:
        return "SOLUTION_TOLERANCE_EXCEEDED"
    if gap is None or gap > 0.01 + 1e-9:
        return "FEASIBLE_GAP_OPEN"
    return "PASS" if runtime <= acceptable_seconds else "SOLVED_OVER_TARGET"


def solve_case(case, seed, folder, time_limit, acceptable_seconds, expected_sources,
               reference_signature=None, start_from=None):
    if source_hashes() != expected_sources:
        raise RuntimeError("Model source or input files changed during the experiment.")
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with (folder / "console.txt").open("w", encoding="utf-8") as console:
        with redirect_stdout(console), redirect_stderr(console), built_model() as namespace:
            model = namespace["m"]
            original_objective = model.getObjective()
            original_fingerprint = f"{model.Fingerprint & 0xffffffff:08x}"
            start_hash = hashlib.sha256(json.dumps(model.getAttr("Start")).encode()).hexdigest()
            signature = {
                "zero_objective_fingerprint": zero_objective_fingerprint(model),
                "original_fingerprint": original_fingerprint,
                "original_start_sha256": start_hash,
                "rows": model.NumConstrs, "columns": model.NumVars,
                "general_constraints": model.NumGenConstrs, "binaries": model.NumBinVars,
            }
            if reference_signature is not None and signature != reference_signature:
                raise RuntimeError("Constraints, original objective, or original MIP starts changed between cases.")
            distance, heading = apply_case(namespace, case)
            if zero_objective_fingerprint(model) != signature["zero_objective_fingerprint"]:
                raise RuntimeError("An ablation changed more than the objective.")
            model.Params.TimeLimit = time_limit
            model.Params.Seed = seed
            model.Params.MIPGap = 0.01
            model.Params.LogToConsole = 0
            model.Params.LogFile = str(folder / "gurobi.log")
            if start_from is not None:
                model.read(str(start_from))
                model.update()
            metadata = {
                "case": case, "seed": seed, **signature,
                "fingerprint": f"{model.Fingerprint & 0xffffffff:08x}",
                "threads": model.Params.Threads, "time_limit_s": time_limit,
                "target_gap": model.Params.MIPGap, "acceptable_seconds": acceptable_seconds,
                "start_from": str(start_from) if start_from else None,
                "distance_weight": namespace["w_dist"] if CASES[case][0] else 0.0,
                "heading_weight": namespace["w_accel"] if CASES[case][1] else 0.0,
                "time_weight": namespace["w_time"], "fuel_weight": namespace["w_fuel"],
                "descent_weight": namespace["w_descent"],
                "flight_count": namespace["N_flights"], "grid_points": namespace["N_steps"],
                "distance_terms": len(distance), "heading_terms": len(heading),
                "objective_constant": model.ObjCon,
            }
            write_json(folder / "metadata.json", metadata)
            write_model(model, folder)
            model.write(str(folder / "parameters.prm"))
            build_seconds = time.perf_counter() - started
            with (folder / "progress.csv").open("w", newline="", encoding="utf-8") as stream:
                trace = SolverTrace(stream)
                model.optimize(trace)
            has_solution = model.SolCount > 0
            gap = finite(model.MIPGap) if has_solution else None
            objective = finite(model.ObjVal) if has_solution else None
            bound = finite(model.ObjBound)
            quality = {key: None for key in ("constraint_violation", "bound_violation", "integer_violation")}
            quality_ok = False
            metrics = {}
            if has_solution:
                quality = {
                    "constraint_violation": model.ConstrVio, "bound_violation": model.BoundVio,
                    "integer_violation": model.IntVio,
                }
                quality_ok = (model.ConstrVio <= model.Params.FeasibilityTol
                              and model.BoundVio <= model.Params.FeasibilityTol
                              and model.IntVio <= model.Params.IntFeasTol)
                metrics = {
                    "original_objective_at_solution": original_objective.getValue(),
                    "horizontal_pwl_m": sum(variable.X for variable in distance),
                    "heading_l1_m_per_step": sum(variable.X for variable in heading),
                    "pwl_fuel_kg": sum(variable.X for variable in namespace["fuel_step_kg"].values()),
                    "active_flight_seconds": sum(
                        namespace["TIMESTEP_DT"] * (1 - namespace["fix_reached"][i, k - 1].X)
                        for i in range(namespace["N_flights"])
                        for k in range(int(namespace["flights"].iloc[i]["flight_entry_timestep"]) + 1,
                                       namespace["N_steps"])),
                }
                model.write(str(folder / "solution.sol"))
                write_csv(folder / "trajectory.csv", [
                    {"flight": namespace["flights"].iloc[i]["acId"], "step": k,
                     "x_m": namespace["f_x"][i][k].X, "y_m": namespace["f_y"][i][k].X,
                     "z_m": namespace["f_z"][i][k].X,
                     "fix_reached": namespace["fix_reached"][i, k].X}
                    for i in range(namespace["N_flights"]) for k in range(namespace["N_steps"])
                ])
            result = {
                "case": case, "seed": seed, "status": STATUS_NAMES.get(model.Status, str(model.Status)),
                "outcome": outcome(model.Status, model.SolCount, gap, model.Runtime,
                                   acceptable_seconds, quality_ok),
                "runtime_s": model.Runtime, "build_and_export_s": build_seconds, "work": model.Work,
                "nodes": model.NodeCount, "simplex_iterations": model.IterCount,
                "solutions": model.SolCount, "objective": objective, "bound": bound, "gap": gap,
                "absolute_gap": abs(objective - bound) if objective is not None and bound is not None else None,
                "first_incumbent_s": trace.first_incumbent_s,
                "first_incumbent_objective": trace.first_incumbent_objective,
                "last_improvement_s": trace.last_improvement_s, "quality_ok": quality_ok,
                **quality, **metrics,
            }
            write_json(folder / "result.json", result)
    return result, signature


def write_summary(output, rows):
    write_json(output / "summary.json", rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "summary.csv", [{key: row.get(key) for key in columns} for row in rows])
    lines = [
        "# v16.2 objective ablation results", "",
        "Objective-only changes; all physical constraints remain in place. "
        "OPTIMAL means within the configured 1% MIP gap, not exact optimality.", "",
        "| Case | Seed | Outcome | Solve (s) | First feasible (s) | Gap (%) | Nodes |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        first = "-" if row["first_incumbent_s"] is None else f'{row["first_incumbent_s"]:.3f}'
        gap = "-" if row["gap"] is None else f'{100 * row["gap"]:.4f}'
        lines.append(f'| {row["case"]} | {row["seed"]} | {row["outcome"]} | '
                     f'{row["runtime_s"]:.3f} | {first} | {gap} | {row["nodes"]:.0f} |')
    lines.extend(["", "A time limit with an incumbent is not infeasibility. Raw objective values "
                  "across different objectives are not comparable; use physical metrics or "
                  "original_objective_at_solution. Inspect each case's progress.csv and gurobi.log "
                  "to distinguish incumbent search from bound improvement.", ""])
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(cases, seeds, time_limit=1200.0, acceptable_seconds=900.0,
                   output_dir=None, start_from=None):
    if (not math.isfinite(time_limit) or not math.isfinite(acceptable_seconds)
            or not 0 < acceptable_seconds <= time_limit):
        raise ValueError("Require 0 < acceptable seconds <= finite time limit.")
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("Supply distinct, known experiment cases.")
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= seed <= 2000000000 for seed in seeds):
        raise ValueError("Supply distinct Gurobi seeds between 0 and 2000000000.")
    if start_from is not None:
        start_from = Path(start_from).resolve(strict=True)
    output = Path(output_dir).resolve() if output_dir is not None else (
        SCRIPT_DIR / "Experiments" / "v162_objective_ablation" /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    os.environ["MPLBACKEND"] = "Agg"
    sources = source_hashes()
    write_json(output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "cases": cases, "seeds": seeds,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "logical_cpus": os.cpu_count(), "gurobi": gp.gurobi.version(),
        "packages": {name: version(name) for name in ("gurobipy", "numpy", "pandas", "openap")},
        "source_sha256": sources, "time_limit_s": time_limit, "acceptable_seconds": acceptable_seconds,
        "target_gap": 0.01, "start_from": str(start_from) if start_from else None,
    })
    print(f"Experiment directory: {output}", flush=True)
    rows, reference = [], None
    for seed in seeds:
        for case in cases:
            print(f"Starting {case}, seed={seed} (sequential, limit={time_limit:g}s)", flush=True)
            result, reference = solve_case(
                case, seed, output / f"{case}_seed{seed}", time_limit, acceptable_seconds,
                sources, reference, start_from)
            rows.append(result)
            write_summary(output, rows)
            print(f'{case}: {result["outcome"]}, {result["runtime_s"]:.3f}s, '
                  f'gap={result["gap"]}, first feasible={result["first_incumbent_s"]}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--time-limit", type=float, default=1200.0)
    parser.add_argument("--acceptable-seconds", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-from", type=Path, help="Diagnostic only: use the same saved solution as a MIP start.")
    args = parser.parse_args()
    run_experiment(args.cases, args.seeds, args.time_limit, args.acceptable_seconds,
                   args.output_dir, args.start_from)


if __name__ == "__main__":
    main()
