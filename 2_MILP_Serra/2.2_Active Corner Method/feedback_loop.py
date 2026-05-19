# ============================================================
# FEEDBACK LOOP
# ============================================================

from dataclasses import dataclass
import os
import pandas as pd
from gurobipy import *
from run_acm_multi_aircraft import run_acm_multi_until_star
from milp_multiple_Debug import solve_milp


@dataclass
class FeedbackLoopConfig:

    initial_dt: float = 300.0
    beta: float = 0.5
    min_dt: float = 1.0
    max_iterations: int = 10
    output_dir: str = "feedback_loop_outputs"


@dataclass
class FeedbackLoopResult:

    outcome: str
    final_dt: float
    iterations: int
    final_trajectories: pd.DataFrame | None
    final_acm_results: pd.DataFrame | None
    final_3d_conflicts: pd.DataFrame | None
    details: str


# ============================================================
# EXTRACT MILP SOLUTION
# ============================================================

def extract_trajectory_dataframe(
    model,
    n,
    N,
    DT,
):

    pat = []

    for i in range(N):
        for j in range(n):

            pat.append(f"f{j+1}_lat[{i}]")
            pat.append(f"f{j+1}_lon[{i}]")
            pat.append(f"f{j+1}_alt_ft[{i}]")

    data = {
        "var": [
            v.VarName
            for v in model.getVars()
            if v.VarName in pat
        ],

        "value": [
            v.X
            for v in model.getVars()
            if v.VarName in pat
        ],
    }

    df = pd.DataFrame(data)

    df["root"] = (
        df["var"]
        .str.extract(r"^([^\[]+)", expand=False)
    )

    df["t"] = (
        df["var"]
        .str.extract(r"\[(\d+)\]", expand=False)
        .astype(int)
    ) * DT

    wide = (
        df.pivot(
            index="t",
            columns="root",
            values="value",
        )
        .sort_index()
        .reset_index()
    )

    ordered = ["t"]

    for i in range(n):

        ordered.extend([
            f"f{i+1}_lat",
            f"f{i+1}_lon",
            f"f{i+1}_alt_ft",
        ])

    wide = wide[
        ordered +
        [c for c in wide.columns if c not in ordered]
    ]

    return wide


# ============================================================
# RUN ACM
# ============================================================

def validate_with_acm(
    trajectories,
    dt,
    iteration,
    output_dir,
):

    os.makedirs(output_dir, exist_ok=True)

    traj_path = os.path.join(
        output_dir,
        f"trajectory_iter_{iteration}_dt_{dt:g}.csv",
    )

    trajectories.to_csv(
        traj_path,
        index=False,
    )

    acm_output = run_acm_multi_until_star(
        traj_path
    )

    if isinstance(acm_output, tuple):

        acm_results = acm_output[0]

    else:

        acm_results = acm_output

    conflicts = acm_results[
        acm_results["status"] == "UNSAFE"
    ]

    REQUIRED_PLANES = {"lat_lon", "lat_alt", "lon_alt"}

    conflicts_3d = (
        conflicts
        .groupby(["aircraft_1", "aircraft_2", "segment", "t_start", "t_end"])
        .filter(lambda g: REQUIRED_PLANES.issubset(set(g["plane"])))
    )   

    acm_results.to_csv(
        os.path.join(
            output_dir,
            f"acm_results_iter_{iteration}_dt_{dt:g}.csv",
        ),
        index=False,
    )

    conflicts_3d.to_csv(
        os.path.join(
            output_dir,
            f"acm_3d_conflicts_iter_{iteration}_dt_{dt:g}.csv",
        ),
        index=False,
    )

    return acm_results, conflicts_3d


# ============================================================
# FEEDBACK LOOP
# ============================================================

def feedback_loop(
    initial_dt=300.0,
    beta=0.5,
    min_dt=1.0,
    max_iterations=10,
):

    output_dir = "feedback_loop_outputs"

    os.makedirs(output_dir, exist_ok=True)

    dt = initial_dt

    loop_log = []

    for iteration in range(max_iterations):

        print("\n" + "="*70)
        print(f"ITERATION {iteration}")
        print(f"DT = {dt}")
        print("="*70)

        # ====================================================
        # IMPORTANT:
        # Replace ONLY the DT assignment in your MILP
        # with:
        #
        #     DT = dt
        #
        # Then wrap your whole MILP model construction
        # into:
        #
        #     def solve_milp(dt):
        #         ...
        #         return m
        #
        # ====================================================

        m = solve_milp(dt)

        if m.status != GRB.OPTIMAL:

            loop_log.append({
                "iteration": iteration,
                "dt": dt,
                "milp_status": m.status,
            })

            pd.DataFrame(loop_log).to_csv(
                os.path.join(
                    output_dir,
                    "feedback_loop_log.csv",
                ),
                index=False,
            )

            return FeedbackLoopResult(
                outcome="INFEASIBLE",
                final_dt=dt,
                iterations=iteration + 1,
                final_trajectories=None,
                final_acm_results=None,
                final_3d_conflicts=None,
                details=(
                    f"MILP not optimal. "
                    f"Status={m.status}"
                ),
            )

        print(f"Obj: {m.ObjVal:g}")

        trajectories = extract_trajectory_dataframe(
            model=m,
            n=10,
            N=209,
            DT=dt,
        )

        trajectories.to_csv(
            os.path.join(
                output_dir,
                f"milp_solution_iter_{iteration}_dt_{dt:g}.csv",
            ),
            index=False,
        )

        # ====================================================
        # ACM VALIDATION
        # ====================================================

        acm_results, conflicts_3d = validate_with_acm(
            trajectories=trajectories,
            dt=dt,
            iteration=iteration,
            output_dir=output_dir,
        )

        unsafe_projection_count = len(
            acm_results[
                acm_results["status"] == "UNSAFE"
            ]
        )

        error_count = len(
            acm_results[
                acm_results["status"] == "ERROR"
            ]
        )

        confirmed_3d_count = len(
            conflicts_3d
        )

        loop_log.append({
            "iteration": iteration,
            "dt": dt,
            "milp_status": m.status,
            "objective": m.ObjVal,
            "unsafe_projection_count": unsafe_projection_count,
            "confirmed_3d_conflicts": confirmed_3d_count,
            "acm_errors": error_count,
        })

        print(
            f"Unsafe projection rows: "
            f"{unsafe_projection_count}"
        )

        print(
            f"Confirmed 3D conflicts: "
            f"{confirmed_3d_count}"
        )

        print(
            f"ACM errors: "
            f"{error_count}"
        )

        # ====================================================
        # SAFE
        # ====================================================

        if (
            confirmed_3d_count == 0
            and error_count == 0
        ):

            pd.DataFrame(loop_log).to_csv(
                os.path.join(
                    output_dir,
                    "feedback_loop_log.csv",
                ),
                index=False,
            )

            return FeedbackLoopResult(
                outcome="SAFE_3D",
                final_dt=dt,
                iterations=iteration + 1,
                final_trajectories=trajectories,
                final_acm_results=acm_results,
                final_3d_conflicts=conflicts_3d,
                details=(
                    "No confirmed 3D conflicts."
                ),
            )

        # ====================================================
        # UNSAFE -> REFINE DT
        # ====================================================

        next_dt = beta * dt

        print(
            f"Refining DT: "
            f"{dt:g} -> {next_dt:g}"
        )

        if next_dt < min_dt:

            pd.DataFrame(loop_log).to_csv(
                os.path.join(
                    output_dir,
                    "feedback_loop_log.csv",
                ),
                index=False,
            )

            return FeedbackLoopResult(
                outcome="UNKNOWN_MIN_DT_REACHED",
                final_dt=dt,
                iterations=iteration + 1,
                final_trajectories=trajectories,
                final_acm_results=acm_results,
                final_3d_conflicts=conflicts_3d,
                details=(
                    f"next_dt={next_dt:g} "
                    f"below min_dt={min_dt:g}"
                ),
            )

        dt = next_dt

    # ========================================================
    # MAX ITERATIONS
    # ========================================================

    pd.DataFrame(loop_log).to_csv(
        os.path.join(
            output_dir,
            "feedback_loop_log.csv",
        ),
        index=False,
    )

    return FeedbackLoopResult(
        outcome="UNKNOWN_MAX_ITERATIONS_REACHED",
        final_dt=dt,
        iterations=max_iterations,
        final_trajectories=trajectories,
        final_acm_results=acm_results,
        final_3d_conflicts=conflicts_3d,
        details="Reached max iterations.",
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    result = feedback_loop(
        initial_dt=300.0,
        beta=0.5,
        min_dt=1.0,
        max_iterations=10,
    )

    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)

    print("Outcome:", result.outcome)
    print("Final DT:", result.final_dt)
    print("Iterations:", result.iterations)
    print("Details:", result.details)