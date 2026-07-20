# need to check 1) everything in xyz in meters and 2) grid point MILP formulation

from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
from Functions.fuel_model import *
from Functions.fuel_emission_analysis_main import analyze_optimized_trajectory_xyz
from Functions.utilities import (
    load_flights_utc_xyz, load_star_fixes_xyz, compute_time_grid,
    print_chosen_star_fixes_xyz, print_exit_times,
    save_trajectory_csv_xyz, print_waypoint_table_xyz,
    print_separation_check_xyz, save_decision_variables_csv_xyz,
)
import os
# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

############
# 1. SETUP #
############
print(" === PROBLEM SETUP ===")
## 1.1 Set parameters
# Global parameters — all positions in SI metres from DTW origin
DTW_LAT0, DTW_LON0 = 42.2125, -83.3534     # Projection origin (DTW airport)
FT2M               = 0.3048                 # Feet → metres
NM2M               = 1852.0                 # Nautical miles → metres

flights_to_optimize = [
    # "DAL1066_KTPAtoKDTW",   # enters 07:56 (from SE)
    # "EDV5018_CYULtoKDTW",   # enters 08:12 (from NE)
    # "DAL2140_KSTLtoKDTW",   # enters 08:09 (from W)
    # "DAL1120_KMSNtoKDTW",   # enters 08:10 (from W)
    "AAL419_KCLTtoKDTW",   # enters 08:11 (from SE)
]

# Model parameters
TIMESTEP_DT     = 300.0   # Time step (s)
N_STEPS_HORIZON = 6       # Extra steps beyond last entry step

# Cost parameters — NOTE: w_smooth / w_accel scaled for metre-based displacements (~1e4 m/step)
w_time      = 1.0         # Time cost weight (per active step × dt)
w_smooth    = 5e-7        # L1 displacement smoothness weight (m⁻¹)
w_accel     = 1e-5        # L1 heading-rate weight (second difference, m⁻¹)
w_z         = 0.1         # Relative vertical vs horizontal penalty
w_descent   = 0           # Descent reward per metre (0 = disabled)
w_alt_final = 0           # Final-altitude penalty per metre (0 = disabled)

# Constraint parameters — all in SI (metres, m/step)
BIG_M      = 1e7                                   # Disjunction constant (≫ max pos diff ~5×10⁵ m)
VMAX_XY    = 300.0 * TIMESTEP_DT                   # Max horiz speed/step (300 m/s → 90 000 m/step)
VMAX_Z     = (1000.0 / 60.0) * FT2M * TIMESTEP_DT  # Max vert speed/step (1000 ft/min → ~1524 m/step)
VMIN_2D    = 120.0 * (NM2M / 3600.0) * TIMESTEP_DT # Min 2D speed/step  (120 kts  → ~18 520 m/step)
SEP_HOR_M  = 3.0 * NM2M                             # Horizontal separation (3 NM → 5556 m)
SEP_VERT_M = 1000.0 * FT2M                         # Vertical separation   (1000 ft → 304.8 m)

# Aircraft parameters (used by downstream fuel analysis only)
S    = 122.6    # Wing area (m²)
mtow = 70000    # Max takeoff weight (kg)
tsfc = 0.00003  # Thrust specific fuel consumption
cd0  = 0.02     # Zero-lift drag coefficient

## 1.2. Define STAR fixes — projected to (x, y) metres from DTW, altitude in metres
script_dir = Path(__file__).parent
star_fixes_xyz, x_vals, y_vals, z_vals_max, z_vals_min = load_star_fixes_xyz(
    script_dir / "Input" / "star_fixes.csv", DTW_LAT0, DTW_LON0)

## 1.3. Load flight data — positions converted to metres, altitude to metres
csv_path = script_dir / "Input" / "entry_exit_points.csv"
flights, GRID_EPOCH_UTC = load_flights_utc_xyz(
    csv_path, flights_to_optimize, TIMESTEP_DT, DTW_LAT0, DTW_LON0)

## 1.4. Build 3D pre-TRACON waypoint grid (x, y, z all discrete)
# i) Regular 3D grid: horizontal spacing 30 km, vertical spacing 1500 m, z ∈ [4000, 12000] m
fix_names = list(star_fixes_xyz)
n_fixes   = len(fix_names)

GRID_SPACING_XY = 30_000    # Horizontal grid spacing (m) — 30 km
GRID_X_RANGE    = 250_000   # Grid x extent: ±250 km from DTW
GRID_Y_RANGE    = 250_000   # Grid y extent: ±250 km from DTW
GRID_Z_MIN      =  4_000    # Minimum altitude (m MSL) — ~13,100 ft
GRID_Z_MAX      = 12_000    # Maximum altitude (m MSL) — ~39,400 ft
GRID_SPACING_Z  =  1_500    # Vertical grid spacing (m) — ~4,920 ft per level

_gx = np.arange(-GRID_X_RANGE, GRID_X_RANGE + GRID_SPACING_XY, GRID_SPACING_XY)
_gy = np.arange(-GRID_Y_RANGE, GRID_Y_RANGE + GRID_SPACING_XY, GRID_SPACING_XY)
_gz = np.arange(GRID_Z_MIN,    GRID_Z_MAX    + 1,               GRID_SPACING_Z)
WP_xyz = np.array([(x, y, z) for x in _gx for y in _gy for z in _gz], dtype=float)
print(f"Regular 3D grid: {len(_gx)}×{len(_gy)}×{len(_gz)} = {len(WP_xyz)} waypoints")

# ii) Append STAR fix (x,y) positions at every grid altitude within each fix's allowed z range
# " need to create grid points at the STAR fix (x,y) locations, at each valid altitude."
_star_xy    = np.array([(star_fixes_xyz[nm][0], star_fixes_xyz[nm][1]) for nm in fix_names])
_star_z_min = np.array(z_vals_min)
_star_z_max = np.array(z_vals_max)
_fix_extra  = []
for f in range(n_fixes):
    _z_ok = _gz[(_gz >= _star_z_min[f] - 1.0) & (_gz <= _star_z_max[f] + 1.0)]
    if len(_z_ok) == 0:             # no grid level in range → use midpoint of STAR fix's z range
        _z_ok = [0.5 * (_star_z_min[f] + _star_z_max[f])]
    for zl in _z_ok:
        _fix_extra.append((_star_xy[f, 0], _star_xy[f, 1], float(zl)))
if _fix_extra:
    WP_xyz = np.vstack([WP_xyz, np.array(_fix_extra)])

# iii) Append flight entry positions at their exact (x, y, z)
_entry_xyz = flights[['entry_x', 'entry_y', 'entry_z']].values
WP_xyz     = np.vstack([WP_xyz, _entry_xyz])

# iv) Deduplicate (round to 1 m) and keep sorted order
_, _uniq = np.unique(np.round(WP_xyz, 0), axis=0, return_index=True)
WP_xyz   = WP_xyz[np.sort(_uniq)]
WP_x     = WP_xyz[:, 0].tolist()
WP_y     = WP_xyz[:, 1].tolist()
WP_z     = WP_xyz[:, 2].tolist()
N_WP     = len(WP_xyz)
print(f"3D waypoint grid: {len(_gx)}×{len(_gy)}×{len(_gz)} regular + {n_fixes} STAR fixes + "
      f"{len(flights)} entry pts → {N_WP} unique waypoints")

# v) Map STAR fix f → list of waypoint indices
# "Now that the final WP_xyz array is frozen (after dedup + sort), which indices n correspond to each STAR fix?"
wp_fix_indices = []
for f in range(n_fixes):
    _cand = np.where(
        (np.abs(WP_xyz[:, 0] - _star_xy[f, 0]) < 1.0) &
        (np.abs(WP_xyz[:, 1] - _star_xy[f, 1]) < 1.0) &
        (WP_xyz[:, 2] >= _star_z_min[f] - 1.0) &
        (WP_xyz[:, 2] <= _star_z_max[f] + 1.0)
    )[0].tolist()
    if not _cand:
        raise ValueError(f"STAR fix '{fix_names[f]}' has no 3D grid waypoints — widen GRID_Z range")
    wp_fix_indices.append(_cand)
    print(f"  Fix {fix_names[f]}: {len(_cand)} waypoint(s) in z-range [{_star_z_min[f]:.0f}, {_star_z_max[f]:.0f}] m")

# Union of all valid STAR-fix waypoint indices (aircraft must end at one of these)
_all_fix_wp_idx = sorted(set(n for idx in wp_fix_indices for n in idx))

# vi) Map flight i → entry waypoint index (exact match in 3D)
wp_entry_idx = [int(np.argmin(np.linalg.norm(WP_xyz - _entry_xyz[i], axis=1)))
                for i in range(len(flights))]

# Pre-compute 3D transition graph: reachable_from[n] = indices reachable in one step
# Both conditions must hold: 2D dist ≤ VMAX_XY  AND  |Δz| ≤ VMAX_Z
# This pre-computes for each waypoint: "which other waypoints can the aircraft fly to in a single time step?"
_dist_xy       = np.linalg.norm(WP_xyz[:, None, :2] - WP_xyz[None, :, :2], axis=-1)  # (N_WP, N_WP)
_dist_z        = np.abs(WP_xyz[:, None, 2]  - WP_xyz[None, :, 2])                    # (N_WP, N_WP)
reachable_from = [np.where((_dist_xy[n] <= VMAX_XY) & (_dist_z[n] <= VMAX_Z))[0].tolist()
                  for n in range(N_WP)]
print(f"Avg reachable neighbours per waypoint: {np.mean([len(r) for r in reachable_from]):.1f}")

## 1.5. Determine time steps
N_steps, max_entry_k = compute_time_grid(flights, N_STEPS_HORIZON)
_epoch_end_utc = GRID_EPOCH_UTC + pd.Timedelta(seconds=(N_steps - 1) * TIMESTEP_DT)
print(f"UTC grid epoch  : {GRID_EPOCH_UTC.strftime('%Y-%m-%d %H:%M:%S')} UTC (k=0)")
print(f"Grid step size  : {int(TIMESTEP_DT)}s  ({TIMESTEP_DT / 60:.0f} min per step)")
print(f"Time steps      : {N_steps}  (k=0 → {GRID_EPOCH_UTC.strftime('%H:%M')} UTC, k={N_steps-1} → {_epoch_end_utc.strftime('%H:%M')} UTC)")
print(f"max_entry_k={max_entry_k}, horizon={N_STEPS_HORIZON} extra steps...")
print(f"\nFinal flight data for optimization:\n{flights}\n")

########################
# 2. MILP Optimization #
########################
print(" === MILP OPTIMIZATION ===")
# 2.1. Create model
env = Env(empty=True)
env.setParam("OutputFlag", 0)   # disable all output from Gurobi
# env.setParam("Seed", 42)        # fix random seed for reproducibility
# env.setParam("Threads", 1)      # single thread eliminates parallel non-determinism
env.start()

m = Model("mip1", env=env)
print("Model created...")

# 2.2. Create decision variables — all positions and displacements in SI metres
# i) Position variables: x = east (m), y = north (m) from DTW; z = altitude (m MSL)
N_flights = len(flights)
f_x = [m.addVars(range(N_steps), lb=-GRB.INFINITY, name=f"f{i}_x") for i in range(1, N_flights+1)] # East  position (m)
f_y = [m.addVars(range(N_steps), lb=-GRB.INFINITY, name=f"f{i}_y") for i in range(1, N_flights+1)] # North position (m)
f_z = [m.addVars(range(N_steps),                   name=f"f{i}_z") for i in range(1, N_flights+1)] # Altitude (m MSL)
u_x = [m.addVars(range(N_steps),                   name=f"uf{i}_x") for i in range(1, N_flights+1)] # |Δeast|  per step (m)
u_y = [m.addVars(range(N_steps),                   name=f"uf{i}_y") for i in range(1, N_flights+1)] # |Δnorth| per step (m)
u_z = [m.addVars(range(N_steps),                   name=f"uf{i}_z") for i in range(1, N_flights+1)] # |Δalt|   per step (m)

# ii) STAR fix selection (binary) — fix_names / n_fixes defined in Section 1.4
fix_sel = [m.addVars(n_fixes, vtype=GRB.BINARY, name=f"y{i+1}") for i in range(N_flights)]

# iii) Auxiliary variables — all in metres
d_x, d_y, d_z   = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("dx",  "dy", "dz")]  # Δx, Δy, Δz per step (m)
dd_x,   dd_y     = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("ddx", "ddy"     )]  # Δ(Δx), Δ(Δy) — heading-rate proxy (m)
u_dd_x, u_dd_y   = [m.addVars(N_flights, N_steps, lb=0.0,           vtype=GRB.CONTINUOUS, name=n) for n in ("udx", "udy"     )]  # |ddx|, |ddy| (m)
accel_cost, speed_2d = [m.addVars(N_flights, N_steps, lb=0.0,       vtype=GRB.CONTINUOUS, name=n) for n in ("accel_xy", "speed")]  # heading-rate cost, 2D speed (m/step)
fix_reached, sep_bypass, fix_enters = [m.addVars(N_flights, N_steps, lb=0.0, vtype=GRB.BINARY, name=n) for n in ("is_end", "landed", "delta_end")]
k_arrive = m.addVars(N_flights, lb=0, vtype=GRB.INTEGER, name="exit_k", ub=N_steps-1)

# iv) Waypoint assignment: wp[i][k, n] = 1 if aircraft i is at grid waypoint n at step k
wp = [m.addVars(N_steps, N_WP, vtype=GRB.BINARY, name=f"wp{i+1}") for i in range(N_flights)]
print("Decision variables created...")

# 2.3. Define Objective Function
obj = LinExpr()

for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    
    for k in range(k_entry + 1, N_steps):
        active = 1 - fix_reached[i, k]                  # 1 while airborne, 0 after arrival
        obj += w_time * TIMESTEP_DT * active            # penalize each active step → minimizes flight time

        # Step differences and L1 absolutes — all in metres
        for diff, u, cur, prv in zip(
            (d_x[i,k],      d_y[i,k],      d_z[i,k]   ),
            (u_x[i][k-1],   u_y[i][k-1],   u_z[i][k-1]),
            (f_x[i][k],     f_y[i][k],     f_z[i][k]  ),
            (f_x[i][k-1],   f_y[i][k-1],   f_z[i][k-1]),
        ):
            m.addConstr(diff == cur - prv)
            m.addConstr(u    == abs_(diff))

        # speed_2d QCP constraint omitted — waypoint transition graph bounds horizontal speed
        obj += w_smooth * (u_x[i][k-1] + u_y[i][k-1] + w_z * u_z[i][k-1]) # Smoothness penalty (L1, m)
        obj += w_descent * d_z[i, k]  # descent reward (d_z < 0 descending → reduces objective)

        # Heading-rate penalty (k_entry+2 onward)
        if k >= k_entry + 2:
            for dd, ud, df, dfp in zip(
                (dd_x[i,k],    dd_y[i,k]  ),
                (u_dd_x[i,k],  u_dd_y[i,k]),
                (d_x[i,k],     d_y[i,k]   ),
                (d_x[i,k-1],   d_y[i,k-1] ),
            ):
                m.addConstr(dd == df - dfp)
                m.addConstr(ud == abs_(dd))
            m.addGenConstrIndicator(fix_reached[i,k], 0, accel_cost[i,k] == u_dd_x[i,k] + u_dd_y[i,k])
            m.addGenConstrIndicator(fix_reached[i,k], 1, accel_cost[i,k] == u_dd_x[i,k] + u_dd_y[i,k])
            obj += w_accel * accel_cost[i,k] # Heading-rate penalty (m/step)

m.setObjective(obj, GRB.MINIMIZE)
print("Objective function created...")


# 2.4. Define constraints
# 0) Waypoint grid constraints
for i in range(N_flights): # for each aircraft
    k_entry = flights.iloc[i]['flight_entry_timestep'] # Get the time step when it enters the region.
    for k in range(N_steps):
        # Exactly one waypoint per aircraft per step
        m.addConstr(quicksum(wp[i][k, n] for n in range(N_WP)) == 1,
                    f"wp_one_{i}_{k}")
        # f_x, f_y, f_z are all determined by the selected 3D waypoint
        m.addConstr(f_x[i][k] == quicksum(WP_x[n] * wp[i][k, n] for n in range(N_WP)),
                    f"wp_fx_{i}_{k}")
        m.addConstr(f_y[i][k] == quicksum(WP_y[n] * wp[i][k, n] for n in range(N_WP)),
                    f"wp_fy_{i}_{k}")
        m.addConstr(f_z[i][k] == quicksum(WP_z[n] * wp[i][k, n] for n in range(N_WP)),
                    f"wp_fz_{i}_{k}")
    # Transition feasibility: aircraft can only move to waypoints reachable in one step
    # (2D Euclidean distance ≤ VMAX_XY; includes self-loops so aircraft may idle)
    for k in range(k_entry, N_steps - 1):
        for n2 in range(N_WP):
            m.addConstr(
                wp[i][k+1, n2] <= quicksum(wp[i][k, n1] for n1 in reachable_from[n2]),
                f"wp_trans_{i}_{k}_{n2}"
            )
print("Waypoint grid constraints created...")

# i) Entry constraints — freeze each aircraft at its entry waypoint until k_entry
for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    for k in range(k_entry + 1):
        # Pin waypoint assignment to entry node (f_x, f_y, f_z all follow via 3D linkage)
        m.addConstr(wp[i][k, wp_entry_idx[i]] == 1, f"c_pre_entry_wp_{i}_t{k}")
    print(f"Entry point constraints created, k_entry={k_entry}...")

# ii) STAR fix constraints — link fix_sel to fix's 3D waypoints at the final step
# z is discrete and within each fix's allowed range by grid construction; no separate z bounds needed
for j in range(N_flights):
    # Final step must land on one of the valid STAR fix waypoints
    m.addConstr(quicksum(wp[j][N_steps-1, n] for n in _all_fix_wp_idx) == 1, f"must_reach_fix{j+1}")
    # fix_sel[j][f] = 1 iff one of fix f's 3D waypoints is selected at the final step
    for f in range(n_fixes):
        m.addConstr(
            fix_sel[j][f] == quicksum(wp[j][N_steps-1, n] for n in wp_fix_indices[f]),
            f"fix_wp_{j}_{f}"
        )
    m.addConstr(quicksum(fix_sel[j][f] for f in range(n_fixes)) == 1, f"one_fix{j+1}")
    obj += w_alt_final * f_z[j][N_steps-1]
print("STAR fix constraints created...")

# iii) Arrival flag (fix_reached / fix_enters) logic — unchanged
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']

    # (1) Before entry: fix_reached must be 0
    for k in range(k_entry + 1):
        m.addConstr(fix_reached[i, k] == 0, f"fix_reached_pre_entry_{i}_{k}")
        m.addConstr(fix_enters[i, k] == 0, f"fix_enters_pre_entry_{i}_{k}")

    # (2) After entry: fix_reached is monotone (0→1); position freezes once reached
    for k in range(k_entry + 2, N_steps):
        m.addConstr(fix_reached[i, k] >= fix_reached[i, k-1], f"fix_reached_monotone_{i}_{k}")
        m.addConstr((fix_reached[i, k] == 1) >> (f_x[i][k] == f_x[i][N_steps-1]))
        m.addConstr((fix_reached[i, k] == 1) >> (f_y[i][k] == f_y[i][N_steps-1]))
        m.addConstr((fix_reached[i, k] == 1) >> (f_z[i][k] == f_z[i][N_steps-1]))

    # (3) Encode 0→1 transition for k_arrive computation
    for k in range(k_entry + 1, N_steps):
        m.addConstr(fix_enters[i, k] >= fix_reached[i, k] - fix_reached[i, k-1], f"fix_enters_lb_{i}_{k}")
        m.addConstr(fix_enters[i, k] <= fix_reached[i, k],                        f"fix_enters_ub_{i}_{k}")
        m.addConstr(fix_enters[i, k] <= 1 - fix_reached[i, k-1],                  f"fix_enters_ub_prev_{i}_{k}")

    m.addConstr(
        k_arrive[i] == quicksum(k * fix_enters[i, k] for k in range(k_entry + 1, N_steps)),
        f"k_arrive_def_{i}"
    )

# (4) Every flight must arrive by final step
for i in range(N_flights):
    m.addConstr(fix_reached[i, N_steps-1] == 1, f"fix_reached_at_final_{i}")

# iv) Speed constraints — all axes (x, y, z) governed by the 3D waypoint transition graph
# (reachable_from enforces dist_xy ≤ VMAX_XY AND |Δz| ≤ VMAX_Z per step simultaneously)
print("Speed constraints: governed entirely by 3D waypoint transition graph...")

# v) Separation constraints — all thresholds in metres (unchanged)
for k in range(N_steps):
    for i in range(N_flights - 1):
        for j in range(i + 1, N_flights):
            if k >= flights.iloc[i]['flight_entry_timestep'] and k >= flights.iloc[j]['flight_entry_timestep']:
                bin_vars = m.addVars(range(6), name=f'bin_{i}_{j}_{k}', vtype=GRB.BINARY)

                m.addConstr(f_x[i][k] - f_x[j][k] >= SEP_HOR_M  - BIG_M*(1 - bin_vars[0]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_y[i][k] - f_y[j][k] >= SEP_HOR_M  - BIG_M*(1 - bin_vars[1]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_z[i][k] - f_z[j][k] >= SEP_VERT_M - BIG_M*(1 - bin_vars[2]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_x[j][k] - f_x[i][k] >= SEP_HOR_M  - BIG_M*(1 - bin_vars[3]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_y[j][k] - f_y[i][k] >= SEP_HOR_M  - BIG_M*(1 - bin_vars[4]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_z[j][k] - f_z[i][k] >= SEP_VERT_M - BIG_M*(1 - bin_vars[5]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])

                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5] >= 1)
print("Separation constraints created...")

# 2.5. Initiate optimization
print("Starting optimization...")
m.optimize()
print("Optimization completed.\n")


# ###########
# 3. OUTPUT #
# ###########
print(" === OUTPUT RESULTS ===")
if m.status == GRB.OPTIMAL: # Only extract results if Gurobi found a valid optimal solution.
    # 3.1. Print objective value and chosen STAR fixes
    print(f'Optimization success! Obj= {m.ObjVal:.4g}\n')
    print_chosen_star_fixes_xyz(flights, fix_sel, fix_names, star_fixes_xyz, f_z, N_steps)
    print_exit_times(flights, k_arrive, GRID_EPOCH_UTC, TIMESTEP_DT)

    # 3.2. Extract and save optimized trajectories
    # Primary columns: f{i}_x/y/z (metres). lat/lon/alt_ft also written for human readability.
    output_dir = script_dir / "Output"
    df_wide = save_trajectory_csv_xyz(
        f_x, f_y, f_z, N_flights, N_steps, GRID_EPOCH_UTC, TIMESTEP_DT,
        output_dir, DTW_LAT0, DTW_LON0, flights=flights, k_arrive=k_arrive)

    # 3.3. Print waypoint table per flight (metres)
    print_waypoint_table_xyz(flights, f_x, f_y, f_z, fix_reached, sep_bypass, N_steps, GRID_EPOCH_UTC, TIMESTEP_DT)

    # Diagnostic: pairwise separation check (metres)
    print_separation_check_xyz(flights, f_x, f_y, f_z, sep_bypass, N_flights, N_steps, SEP_HOR_M, SEP_VERT_M)

    # 3.3b. Verify wp-linkage: every trajectory f_z must equal an exact WP_z value
    # (mathematically guaranteed by f_z == quicksum(WP_z[n]*wp[i][k,n]) + binary sum-to-one)
    print("\n--- wp linkage verification (f_z vs WP_z) ---")
    _WP_z_arr = np.array(WP_z)
    for i in range(N_flights):
        k_entry = int(flights.iloc[i]['flight_entry_timestep'])
        for k in range(k_entry, N_steps):
            fz_val   = f_z[i][k].X
            nn       = int(np.argmin(np.abs(_WP_z_arr - fz_val)))
            err_m    = abs(fz_val - WP_z[nn])
            status   = "OK" if err_m < 1.0 else "MISMATCH!"
            print(f"  flight {i}  k={k:2d}:  f_z={fz_val:8.1f} m   WP_z[{nn:4d}]={WP_z[nn]:8.1f} m   err={err_m:.4f} m  {status}")
    print("--- end verification ---\n")

    # 3.4. Save decision variables and cost breakdown to CSV (SI units)
    save_decision_variables_csv_xyz(
        flights, fix_sel, fix_names, k_arrive, N_steps,
        f_x, f_y, f_z, fix_reached, fix_enters, sep_bypass,
        d_x, d_y, d_z, u_x, u_y, u_z,
        speed_2d, dd_x, dd_y, u_dd_x, u_dd_y, accel_cost,
        w_smooth, w_accel, w_z, GRID_EPOCH_UTC, TIMESTEP_DT, output_dir,
    )

    # 3.5. Analyze and visualize optimized trajectory
    print(" === ANALYZING OPTIMIZED TRAJECTORY ===")
    aircraft_list = [
        {"acId": flights.iloc[i]['acId'], "acType": "B737"}
        for i in range(N_flights)
    ]
    results = analyze_optimized_trajectory_xyz(df_wide, aircraft_list, DTW_LAT0, DTW_LON0)
    print("Analysis and visualization complete!")

    # 3.6. 3D visualization — waypoint grid, TRACON regions, and optimized trajectories
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    TRACON_R_M = 40.0 * NM2M                               # TRACON boundary radius (~40 NM)
    PRE_R_M    = max(GRID_X_RANGE, GRID_Y_RANGE) * 1.02    # pre-TRACON outer extent (≈ grid boundary)
    _th        = np.linspace(0, 2 * np.pi, 361)            # 361 pts → closed loop for line plots
    _th_poly   = np.linspace(0, 2 * np.pi, 180, endpoint=False)  # polygon fill vertices

    fig = plt.figure(figsize=(14, 11))
    ax  = fig.add_subplot(111, projection='3d')

    # —— xy-plane: green filled regions ——
    # pre-TRACON outer circle (pale green fill, drawn first so TRACON circle overlaps it)
    _pc = [(PRE_R_M * np.cos(t), PRE_R_M * np.sin(t), 0.0) for t in _th_poly]
    ax.add_collection3d(Poly3DCollection([_pc], facecolor='palegreen', edgecolor='none', alpha=0.18))

    # TRACON inner circle (light green fill)
    _tc = [(TRACON_R_M * np.cos(t), TRACON_R_M * np.sin(t), 0.0) for t in _th_poly]
    ax.add_collection3d(Poly3DCollection([_tc], facecolor='lightgreen', edgecolor='none', alpha=0.35))

    # Boundary circles (solid/dashed lines)
    ax.plot(TRACON_R_M * np.cos(_th), TRACON_R_M * np.sin(_th), np.zeros_like(_th),
            color='green', ls='-',  lw=1.5, alpha=0.85, label='TRACON boundary (~40 NM)')
    ax.plot(PRE_R_M    * np.cos(_th), PRE_R_M    * np.sin(_th), np.zeros_like(_th),
            color='green', ls='--', lw=1.0, alpha=0.65, label='pre-TRACON boundary')

    # DTW at origin (green star)
    ax.scatter([0], [0], [0], color='green', s=260, marker='*', zorder=10, label='DTW')
    ax.text(4000, 4000, 500, 'DTW', color='green', fontsize=9, fontweight='bold')

    # STAR fixes on the xy plane (green triangles)
    _sf_px = [star_fixes_xyz[nm][0] for nm in fix_names]
    _sf_py = [star_fixes_xyz[nm][1] for nm in fix_names]
    ax.scatter(_sf_px, _sf_py, [0] * n_fixes, color='green', s=65,
               marker='^', zorder=7, label='STAR fixes')
    for nm in fix_names:
        ax.text(star_fixes_xyz[nm][0], star_fixes_xyz[nm][1], 400,
                f' {nm}', color='darkgreen', fontsize=7)

    # —— All 3D grid waypoints at their actual altitude (blue cloud) ——
    ax.scatter(WP_x, WP_y, WP_z,
               color='steelblue', s=5, alpha=0.18, zorder=3,
               label=f'Grid waypoints ({N_WP} pts)')

    # —— Optimized trajectories per flight ——
    _WP_z_arr = np.array(WP_z)
    for i in range(N_flights):
        k_entry   = int(flights.iloc[i]['flight_entry_timestep'])
        traj_x    = [f_x[i][k].X for k in range(N_steps)]
        traj_y    = [f_y[i][k].X for k in range(N_steps)]
        traj_z    = [f_z[i][k].X for k in range(N_steps)]
        act_x     = traj_x[k_entry:]
        act_y     = traj_y[k_entry:]
        act_z     = traj_z[k_entry:]
        flight_id = flights.iloc[i]['acId']

        # Find the exact grid waypoint index selected at each active step (argmax over binary wp)
        _vis_idx = []
        for k in range(k_entry, N_steps):
            _wp_vals = np.array([wp[i][k, n].X for n in range(N_WP)])
            _vis_idx.append(int(np.argmax(_wp_vals)))

        # Highlight visited grid waypoints as large blue circles — proves each orange dot
        # sits exactly on a blue grid waypoint (same x, y, z coordinates)
        _vx = [WP_x[n] for n in _vis_idx]
        _vy = [WP_y[n] for n in _vis_idx]
        _vz = [WP_z[n] for n in _vis_idx]
        ax.scatter(_vx, _vy, _vz, color='blue', s=120, edgecolors='navy',
                   linewidths=0.8, alpha=0.85, zorder=7,
                   label='Visited grid waypoints (exact)' if i == 0 else '')

        # Vertical drop lines from each trajectory point to the xy plane (z=0)
        # makes the altitude of each waypoint unambiguous
        for xi, yi, zi in zip(act_x, act_y, act_z):
            ax.plot([xi, xi], [yi, yi], [0.0, zi],
                    color='gray', linewidth=0.6, alpha=0.35, linestyle=':')

        # Selected (visited) waypoints — orange (plotted on top of the blue circles)
        ax.scatter(act_x, act_y, act_z, color='darkorange', s=60, zorder=9,
                   label='Selected waypoints' if i == 0 else '')

        # Optimized trajectory line — red
        ax.plot(act_x, act_y, act_z, color='red', linewidth=2.5, zorder=8,
                label=f'Trajectory ({flight_id})')

    ax.set_xlabel('East (m from DTW)',  labelpad=10)
    ax.set_ylabel('North (m from DTW)', labelpad=10)
    ax.set_zlabel('Altitude (m MSL)',   labelpad=10)
    ax.set_title(
        'Waypoint-Grid MILP — Optimized Trajectories (v9)\n'
        'Green: DTW / TRACON / pre-TRACON  ·  Blue: grid  ·  Orange: selected  ·  Red: trajectory',
        pad=15,
    )
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    _plot_path = str(output_dir / 'trajectory_3d_v9.png')
    plt.savefig(_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"3D trajectory plot saved → {_plot_path}")