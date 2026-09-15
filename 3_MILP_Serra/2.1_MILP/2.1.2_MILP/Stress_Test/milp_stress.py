#!/usr/bin/env python3
# solver_milp.py
from gurobipy import *
import numpy as np
import pandas as pd
import math
import os
import random
import sys
from datetime import datetime

# ---- Project-specific helpers (expected to exist) ----
# must provide compute_fuel_emission_flow(...)
from main_ver4_gurobi_debug import *

# ----------------- Constants -----------------
DT = 60.0 * 10  # 10 minutes in seconds (discrete step)
FT2NM = 1 / 6076.12

BIG_M = 1e5
V_MAX_X = 0.25 / 60.0    # grid units per second
V_MAX_Y = 0.072 / 60.0
V_MAX_Z = 1000.0 / 60.0

GLIDE_RATIO = 2
SEP_HOR_NM = 500.0 * FT2NM
SEP_VERT_FT = 100.0

CT = 1
CF = 1.5

STAR_FIXES = {
    "BONZZ": (41.7483, -82.7972, (21000, 15000)), "CRAKN": (41.6730, -82.9405, (26000, 12000)),
    "CUUGR": (42.3643, -83.0975, (11000, 10000)), "FERRL": (42.4165, -82.6093, (10000, 8000)),
    "GRAYT": (42.9150, -83.6020, (22000, 17000)), "HANBL": (41.7375, -84.1773, (21000, 17000)),
    "HAYLL": (41.9662, -84.2975, (11000, 11000)), "HTROD": (42.0278, -83.3442, (12000, 12000)),
    "KKISS": (42.5443, -83.7620, (15000, 12000)), "KLYNK": (41.8793, -82.9888, (10000, 9000)),
    "LAYKS": (42.8532, -83.5498, (10000, 10000)), "LECTR": (41.9183, -84.0217, (10000, 8000)),
    "RKCTY": (42.6869, -83.9603, (13000, 11000)), "VCTRZ": (41.9878, -84.0670, (15000, 12000))
}

# Example aircraft parameters (used by your fuel function)
S = 122.6
mtow = 70000
tsfc = 0.00003
cd0 = 0.02
k_induced = 0.045

# --------------- I/O Helpers ----------------
def load_flights(csv_path: str, num_flights: int, shuffle: bool, seed: int):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by='entry_rectime').reset_index(drop=True)
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df[:num_flights]

    df['entry_rectime'] = pd.to_datetime(df['entry_rectime'])
    df['exit_rectime'] = pd.to_datetime(df['exit_rectime'])
    min_time = df['entry_rectime'].min()

    df['entry_time_sec'] = (df['entry_rectime'] - min_time).dt.total_seconds()
    df['landing_time_sec'] = (df['exit_rectime'] - min_time).dt.total_seconds()

    cols = [
        'acId',
        'entry_lat', 'entry_lon', 'entry_alt',
        'entry_time_sec',
        'exit_lat', 'exit_lon', 'exit_alt',
        'landing_time_sec'
    ]
    flights = df[cols].values.tolist()
    return flights, df, min_time

def save_solution_csv(m, N, n, out_path_prefix: str, dt_seconds: int = 60 * 5):
    # Collect only trajectory vars to mimic your original post-processing
    pat = []
    for i in range(N):
        for j in range(n):
            pat.append(f"f{j+1}_lat[{i}]")
            pat.append(f"f{j+1}_lon[{i}]")
            pat.append(f"f{j+1}_alt_ft[{i}]")

    data = {
        "var": [v.VarName for v in m.getVars() if v.VarName in pat],
        "value": [v.X for v in m.getVars() if v.VarName in pat],
    }
    if not data["var"]:
        return None  # nothing to write

    df = pd.DataFrame(data)
    df["root"] = df["var"].str.extract(r"^([^\[]+)", expand=False)
    df["t"] = (df["var"].str.extract(r"\[(\d+)\]", expand=False).astype(int)) * dt_seconds

    wide = (df.pivot(index="t", columns="root", values="value")
              .sort_index()
              .reset_index())

    ordered = ['t'] + [name for j in range(n) for name in (f'f{j+1}_lat', f'f{j+1}_lon', f'f{j+1}_alt_ft')]
    wide = wide[[c for c in ordered if c in wide.columns] + [c for c in wide.columns if c not in ordered]]

    out_csv = f"{out_path_prefix}.csv"
    wide.to_csv(out_csv, index=False)
    return out_csv

# --------------- Core MILP ----------------
def build_and_solve(csv_path: str,
                    num_flights: int = 20,
                    shuffle: bool = False,
                    seed: int = 0,
                    trial_id: str = "0",
                    grb_timelimit_sec: int = 900):
    print(f"[INFO] Loading flights from {csv_path}")
    flights, flights_df, ref_start = load_flights(csv_path, num_flights, shuffle, seed)

    if len(flights) == 0:
        print("[ERROR] No flights loaded.")
        return 1

    print(f"[INFO] Using {len(flights)} flights (NUM_FLIGHTS={num_flights}, SHUFFLE={int(shuffle)}, SEED={seed}, TRIAL_ID={trial_id})")
    print(f"[INFO] Reference start time (t=0): {ref_start}")

    # Determine horizon
    max_time = max(f[8] for f in flights)  # landing_time_sec
    if max_time > 210000:
        t0, tN = 0, max_time
    else:
        t0, tN = 0, 210000

    print(f"[INFO] Actual number of time steps: {((tN - t0) / DT) + 1}")
    N = int((tN - t0) / DT) + 1
    print(f"[INFO] Number of time steps (N): {N}")
    times = np.linspace(t0, tN, N, dtype=int)

    entry_indices = [int(f[4] / DT) for f in flights]  # entry_time_sec / DT
    print(f"[INFO] Entry time indices: {entry_indices}")

    # ----------------- Model -----------------
    m = Model("mip_arrivals")
    m.Params.TimeLimit = float(grb_timelimit_sec)

    n = len(flights)
    x, y, z = [], [], []
    ux, uy, uz = [], [], []

    # Decision variables for trajectories and increments
    for i in range(1, n + 1):
        x.append(m.addVars(range(N), name=f"f{i}_lat", lb=-1e5))
        y.append(m.addVars(range(N), name=f"f{i}_lon", lb=-1e5))
        z.append(m.addVars(range(N), name=f"f{i}_alt_ft"))
        ux.append(m.addVars(range(N), name=f"uf{i}_x"))
        uy.append(m.addVars(range(N), name=f"uf{i}_y"))
        uz.append(m.addVars(range(N), name=f"uf{i}_z"))

    # Fix positions before entry to the entry point (inclusive)
    for i in range(n):
        entry_k = entry_indices[i]
        for k in range(entry_k + 1):
            m.addConstr(x[i][k] == flights[i][1], f"c_pre_entry_x_{i}_t{k}")
            m.addConstr(y[i][k] == flights[i][2], f"c_pre_entry_y_{i}_t{k}")
            m.addConstr(z[i][k] == flights[i][3], f"c_pre_entry_z_{i}_t{k}")

    # STAR fix choices (one per flight)
    fix_names = list(STAR_FIXES)
    lat_vals = [STAR_FIXES[k][0] for k in fix_names]
    lon_vals = [STAR_FIXES[k][1] for k in fix_names]
    alt_vals_min = [STAR_FIXES[k][2][1] for k in fix_names]
    alt_vals_max = [STAR_FIXES[k][2][0] for k in fix_names]
    k_fixes = len(fix_names)

    b = [m.addVars(k_fixes, vtype=GRB.BINARY, name=f"y{i+1}") for i in range(n)]

    for j in range(n):
        m.addConstr(quicksum(b[j][i] for i in range(k_fixes)) == 1, f"one_fix{j+1}")
        m.addConstr(x[j][N - 1] == LinExpr(lat_vals, b[j].values()), f"lat_choice{j+1}")
        m.addConstr(y[j][N - 1] == LinExpr(lon_vals, b[j].values()), f"lon_choice{j+1}")
        m.addConstr(z[j][N - 1] <= LinExpr(alt_vals_max, b[j].values()), f"alt_choice_max{j+1}")
        m.addConstr(z[j][N - 1] >= LinExpr(alt_vals_min, b[j].values()), f"alt_choice_min{j+1}")

    # Bookkeeping binaries (landed flags per k; user’s code references them in safety big-M)
    landed = [[m.addVar(vtype=GRB.BINARY, name=f'landed_{i}_{k}') for k in range(N)] for i in range(n)]
    # End-of-flight indicator per k (to zero out fuel after "end")
    is_end_flag = [[m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}') for k in range(N)] for i in range(n)]

    obj = LinExpr()

    # Dynamics, effort, and fuel cost
    for i in range(n):
        entry_k = entry_indices[i]
        for k in range(entry_k + 1, N):
            # Velocity / increment bounds
            m.addConstr(x[i][k] - x[i][k - 1] <= V_MAX_X * DT)
            m.addConstr(y[i][k] - y[i][k - 1] <= V_MAX_Y * DT)
            m.addConstr(z[i][k] - z[i][k - 1] <= V_MAX_Z * DT)

            m.addConstr(x[i][k - 1] - x[i][k] <= V_MAX_X * DT)
            m.addConstr(y[i][k - 1] - y[i][k] <= V_MAX_Y * DT)
            m.addConstr(z[i][k - 1] - z[i][k] <= V_MAX_Z * DT)

            # Absolute increments for objective terms
            dx = m.addVar(lb=-GRB.INFINITY, name=f'dx{i}_{k}')
            dy = m.addVar(lb=-GRB.INFINITY, name=f'dy{i}_{k}')
            dz = m.addVar(lb=-GRB.INFINITY, name=f'dz{i}_{k}')
            m.addConstr(dx == x[i][k] - x[i][k - 1])
            m.addConstr(dy == y[i][k] - y[i][k - 1])
            m.addConstr(dz == z[i][k] - z[i][k - 1])

            m.addConstr(ux[i][k - 1] == abs_(dx))
            m.addConstr(uy[i][k - 1] == abs_(dy))
            m.addConstr(uz[i][k - 1] == abs_(dz))

            # Mark "end" (if at STAR value), then clamp positions
            m.addConstr((is_end_flag[i][k] == 1) >> (x[i][k] == x[i][N - 1]))
            m.addConstr((is_end_flag[i][k] == 1) >> (y[i][k] == y[i][N - 1]))
            m.addConstr((is_end_flag[i][k] == 1) >> (z[i][k] == z[i][N - 1]))

            # Horizontal speed magnitude (for fuel)
            speed = m.addVar(name=f"speed_{i}_{k}")
            m.addConstr(speed * speed == dx * dx + dy * dy)

            # (Your code defines a PWL arctan but does not use it; we keep it for parity.)
            def f(u): return math.atan(u)
            lbx, ubx, npts = -2, 2, 101
            x_pts = [lbx + (ubx - lbx) * p / (npts - 1) for p in range(npts)]
            y_pts = [f(val) for val in x_pts]
            gamma = m.addVar(name=f"gamma_{i}_{k}")
            lx = m.addVar(lb=lbx, ub=ubx, vtype=GRB.CONTINUOUS, name=f"lx_{i}_{k}")
            m.addGenConstrPWL(lx, gamma, x_pts, y_pts, f"PWLarctan_{i}_{k}")

            # Fuel term "t" active only before end
            t = m.addVar(name=f"fuel_{i}_{k}")

            # We follow your original structure: use indicators to switch fuel off after "end".
            # Note: compute_fuel_emission_flow is assumed to add internal constraints and
            # return an affine expression / variable usable here.
            fuel_expr = compute_fuel_emission_flow(
                speed, z[i][k], dz,
                0.8 * mtow, S, cd0, k_induced, tsfc, m,
                limit=True, cal_emission=False, mode="full"
            )

            # active branch: is_end_flag == 0 -> t == fuel_expr
            m.addGenConstrIndicator(is_end_flag[i][k], 0, t == fuel_expr, name=f"fuel_active_{i}_{k}")
            # ended branch: is_end_flag == 1 -> t == 0
            m.addGenConstrIndicator(is_end_flag[i][k], 1, t == 0, name=f"fuel_zero_{i}_{k}")

            obj += t
            obj += (CT / CF) * (1 - is_end_flag[i][k])

    # Safety separations (only after both have entered)
    for k in range(N):
        for i in range(n - 1):
            for j in range(i + 1, n):
                if k >= entry_indices[i] and k >= entry_indices[j]:
                    bins = m.addVars(range(6), vtype=GRB.BINARY, name=f"bin_{i}_{j}_{k}")
                    m.addConstr(quicksum(bins[p] for p in range(6)) >= 1)

                    m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM - BIG_M * (1 - bins[0]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])
                    m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM - BIG_M * (1 - bins[1]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])
                    m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M * (1 - bins[2]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])

                    m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM - BIG_M * (1 - bins[3]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])
                    m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM - BIG_M * (1 - bins[4]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])
                    m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M * (1 - bins[5]) - BIG_M * landed[i][k] - BIG_M * landed[j][k])

    m.setObjective(obj, GRB.MINIMIZE)

    print(f"[INFO] Optimizing with TimeLimit={grb_timelimit_sec}s ...")
    m.optimize()

    status = m.Status
    print(f"[INFO] Solver status: {status} ({m.Status})")
    if status == GRB.OPTIMAL:
        print(f"[INFO] Optimal objective: {m.ObjVal}")
    elif status == GRB.TIME_LIMIT:
        print("[WARN] Time limit reached.")
        if m.SolCount > 0:
            try:
                print(f"[INFO] Incumbent objective: {m.ObjVal}")
            except Exception:
                pass
    elif status in (GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        print("[WARN] Solver ended without proven optimality.")
    else:
        print(f"[ERROR] Solver ended with status={status}")

    # Save solution trajectories if any
    out_prefix = f"out_{num_flights}flights_trial{trial_id}"
    wrote = None
    if m.SolCount > 0:
        wrote = save_solution_csv(m, N, n, out_prefix, dt_seconds=int(DT))
        if wrote:
            print(f"[INFO] Wrote solution to {wrote}")
    else:
        print("[INFO] No feasible solution available to save.")

    # Exit code policy:
    # - return 0 if optimal OR time limit with an incumbent solution
    # - return 1 otherwise
    if status == GRB.OPTIMAL:
        return 0
    if status == GRB.TIME_LIMIT and m.SolCount > 0:
        return 0
    return 1

# --------------- CLI / Env entrypoint ---------------
if __name__ == "__main__":
    csv_path = os.getenv("FLIGHTS_CSV", "entry_exit_points.csv")
    num_flights = int(os.getenv("NUM_FLIGHTS", "10"))
    shuffle = os.getenv("SHUFFLE", "0") == "1"
    seed = int(os.getenv("SEED", "0"))
    trial_id = os.getenv("TRIAL_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    grb_timelimit_sec = int(float(os.getenv("GRB_TIMELIMIT_SEC", "900")))

    rc = build_and_solve(
        csv_path=csv_path,
        num_flights=num_flights,
        shuffle=shuffle,
        seed=seed,
        trial_id=trial_id,
        grb_timelimit_sec=grb_timelimit_sec,
    )
    sys.exit(rc)
