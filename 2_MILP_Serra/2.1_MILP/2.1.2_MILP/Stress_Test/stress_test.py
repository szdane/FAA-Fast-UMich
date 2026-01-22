#!/usr/bin/env python3
import subprocess
import sys
import time
import os
import statistics as stats
from pathlib import Path
import csv
import random

# ======= CONFIG =======
PYTHON_EXE = sys.executable  # use current interpreter
SOLVER_SCRIPT = "milp_stress.py"  # <-- replace with your actual filename
SIZES = [10, 15, 20, 25, 30]
TRIALS_PER_SIZE = 10          # at least 10 as requested
WALL_TIMEOUT_SEC = 900        # 15 minutes
GRB_TIMELIMIT_SEC = 900       # also pass into Gurobi to match
SHUFFLE = True                # shuffle flight order each trial
RESULTS_CSV = "stress_results.csv"
SUMMARY_CSV = "stress_summary.csv"
# ======================

def run_one(size: int, trial: int) -> dict:
    """
    Run one trial in a subprocess, enforcing WALL_TIMEOUT_SEC.
    Returns a dict with runtime, status, rc, and stderr snippet.
    """
    env = os.environ.copy()
    env["NUM_FLIGHTS"] = str(size)
    env["TRIAL_ID"] = str(trial)
    env["GRB_TIMELIMIT_SEC"] = str(GRB_TIMELIMIT_SEC)
    if SHUFFLE:
        env["SHUFFLE"] = "1"
        env["SEED"] = str(random.randrange(1_000_000_000))
    else:
        env["SHUFFLE"] = "0"

    cmd = [PYTHON_EXE, SOLVER_SCRIPT]

    start = time.time()
    try:
        completed = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=WALL_TIMEOUT_SEC
        )
        end = time.time()
        runtime = end - start
        status = "ok" if completed.returncode == 0 else "error"
        # Detect if Gurobi hit its own time limit:
        if "Time limit reached" in completed.stdout or "Time limit reached" in completed.stderr:
            status = "timelimit"
        return {
            "size": size,
            "trial": trial,
            "status": status,
            "returncode": completed.returncode,
            "runtime_sec": round(runtime, 3),
            "stderr_tail": completed.stderr[-500:],  # last 500 chars
        }
    except subprocess.TimeoutExpired as e:
        # Hard wall timeout: kill process
        return {
            "size": size,
            "trial": trial,
            "status": "timeout",
            "returncode": None,
            "runtime_sec": WALL_TIMEOUT_SEC,
            "stderr_tail": (e.stderr or "")[-500:] if hasattr(e, "stderr") and e.stderr else "",
        }

def main():
    Path(".").mkdir(exist_ok=True, parents=True)

    all_rows = []
    for size in SIZES:
        print(f"\n=== Size {size} ===")
        for trial in range(1, TRIALS_PER_SIZE + 1):
            print(f"  -> Trial {trial}/{TRIALS_PER_SIZE} ...", end="", flush=True)
            row = run_one(size, trial)
            all_rows.append(row)
            print(f" {row['status']} ({row['runtime_sec']} s)")

    # Write detailed results
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size","trial","status","returncode","runtime_sec","stderr_tail"])
        w.writeheader()
        w.writerows(all_rows)

    # Summarize per size (successful runs only)
    summary_rows = []
    for size in SIZES:
        rows = [r for r in all_rows if r["size"] == size]
        ok_runs = [r for r in rows if r["status"] == "ok" or r["status"] == "timelimit"]
        # If you want averages only for fully successful solves (rc==0), use:
        # ok_runs = [r for r in rows if r["status"] == "ok"]

        runtimes = [r["runtime_sec"] for r in ok_runs]
        timeouts = sum(1 for r in rows if r["status"] == "timeout")
        errors = sum(1 for r in rows if r["status"] == "error")

        if runtimes:
            min_rt = min(runtimes)
            max_rt = max(runtimes)
            avg_rt = stats.mean(runtimes)
            std_rt = stats.pstdev(runtimes) if len(runtimes) > 1 else 0.0
        else:
            min_rt = max_rt = avg_rt = std_rt = float("nan")

        summary = {
            "size": size,
            "trials": len(rows),
            "completed": len(ok_runs),
            "timeouts": timeouts,
            "errors": errors,
            "min_sec": round(min_rt, 3) if runtimes else "",
            "max_sec": round(max_rt, 3) if runtimes else "",
            "avg_sec": round(avg_rt, 3) if runtimes else "",
            "std_sec": round(std_rt, 3) if runtimes else "",
        }
        summary_rows.append(summary)

    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["size","trials","completed","timeouts","errors","min_sec","max_sec","avg_sec","std_sec"])
        w.writeheader()
        w.writerows(summary_rows)

    # Pretty print summary
    print("\n=== Summary (runtimes on completed trials) ===")
    for s in summary_rows:
        print(
            f"Flights={s['size']:>2} | trials={s['trials']:>2} | "
            f"completed={s['completed']:>2} | timeouts={s['timeouts']:>2} | errors={s['errors']:>2} | "
            f"min={s['min_sec']}s | max={s['max_sec']}s | avg={s['avg_sec']}s | std={s['std_sec']}s"
        )
    print(f"\nWrote detailed results to ./{RESULTS_CSV} and summary to ./{SUMMARY_CSV}")

if __name__ == "__main__":
    main()
