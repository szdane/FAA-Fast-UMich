"""V16.2-style paired weather/airborne-holding experiment.

Defaults: three aircraft; W1 at 400-800 s, W2 at 800-1200 s, W3 at 1000-1500 s
since first entry; and a 1200-second solve limit per case.
--holding-cost-factor scales only the
holding time/distance/fuel objective terms: 1 is original, 0.75 discounts 25%.
Physical reporting is never discounted; holding ON remains optional.

Six local intervals per aircraft preserve exact entry-time offsets and leave
room for holding. STAR selection and the disabled smoothness term are retained.
Any positive overlap with a region's window protects the entire 400 s interval,
including the holding envelope. weather_start_s/weather_end_s bound all windows;
weather_windows_s supplies the authoritative per-region timing.
Use --w3-window 300 700 --w3-match-w1-size for an earlier, larger W3:
its center/orientation stay fixed while its dimensions match W1.
Functions.holding_output writes v18-style results into a new timestamped folder
under Output/v18_holding_comparison; existing results are never deleted.
"""

from gurobipy import *
import argparse
from copy import deepcopy
import json
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
os.environ.setdefault("MPLBACKEND", "Agg")
from Functions.fuel_pwl_v1 import (  # Use the preserved fuel warm-start support.
    FuelFlowTable, add_horizontal_speed, horizontal_speed_pwl, NORM_COSINE,
)  # Finish the PWL imports.
from Functions.utilities import (
    load_flights_utc_xyz, load_star_fixes_xyz, compute_time_grid, select_cluster,
)
from Functions.free_holding import (
    add_hold_interval, add_weather_interval, add_separation, seed_geometric_start, active_weather_cells,
)
from Functions.holding_output import write_case_output, write_comparison_output

parser = argparse.ArgumentParser(description="V16.2-style model with shared weather and optional airborne holding.")
parser.add_argument("--flight-count", type=int, choices=(1, 3, 6), default=3)
parser.add_argument("--weather-regions", type=int, choices=(1, 3), default=None,
                    help="Default: three regions for multiple aircraft, one for a single aircraft.")
parser.add_argument("--time-limit", type=float, default=1200.0)
parser.add_argument("--holding-cost-factor", type=float, default=1.0,
                    help="Holding objective multiplier: 1=original, 0.75=25%% discount, 0.5=50%% discount.")
parser.add_argument("--w3-window", type=float, nargs=2, metavar=("START", "END"),
                    help="W3 active seconds since first entry; default: 1000 1500.")
parser.add_argument("--w3-match-w1-size", action="store_true",
                    help="Give W3 W1's width and height without moving its center.")
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
if args.weather_regions is None:
    args.weather_regions = 3 if args.flight_count >= 3 else 1
if not math.isfinite(args.time_limit) or args.time_limit <= 0:
    parser.error("--time-limit must be finite and positive.")
if not math.isfinite(args.holding_cost_factor) or args.holding_cost_factor < 0:
    parser.error("--holding-cost-factor must be finite and nonnegative.")
if args.weather_regions > args.flight_count:
    parser.error("Three weather regions require at least three aircraft.")
if (args.w3_window is not None or args.w3_match_w1_size) and args.weather_regions != 3:
    parser.error("W3 options require three weather regions.")
if args.w3_window is not None:
    start, end = args.w3_window
    if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
        parser.error("--w3-window requires finite seconds with 0 <= START < END.")

############
# 1. SETUP #
############
## 1.1 Set parameters
# Global parameters
DTW_LAT0, DTW_LON0 = 42.2125, -83.3534     # Projection origin (DTW airport) (Degrees)
FT2M               = 0.3048                 # Feet → metres
NM2M               = 1852.0                 # Nautical miles → metres

# Model parameters
TIMESTEP_DT     = 400.0   # Time step (s) — 400 s ≈ 6.7 min → gives 2-3 waypoints for typical ~15 min TRACON approach
N_STEPS_HORIZON = 6       # Leave time to hold and transit after the storm.
TIME_LIMIT = args.time_limit  # Preserve the copied 1200-second default.
MIP_GAP = 0.01  # Accept a solution within a 1% gap.
HOLD_SPEED_MPS = 210.0 * NM2M / 3600.0
HOLD_COST_FACTOR = args.holding_cost_factor
WEATHER_WINDOWS_S = [(400.0, 800.0), (800.0, 1200.0), (1000.0, 1500.0)][:args.weather_regions]
if args.w3_window is not None:
    WEATHER_WINDOWS_S[2] = tuple(args.w3_window)

# Cost parameters
w_time      = 1.0         # Time cost weight (per active step × dt)
w_smooth    = 5e-7        # L1 displacement smoothness weight (m⁻¹)
w_accel     = 2e-3        # L1 heading-rate weight (second difference, m⁻¹) — high to keep MILP path smooth
w_dist      = 0.1         # Euclidean distance cost weight (m⁻¹ per step) — penalizes longer paths # NOTE: change to 0.2 for better performance
w_z         = 0.1         # Relative vertical vs horizontal penalty
w_descent   = 0           # Descent reward per metre (0 = disabled)
w_alt_final = 0           # Final-altitude penalty per metre (0 = disabled)
w_fuel = 25.0  # Set the fuel cost weight.

# Constraint parameters
BIG_M      = 1e7                                   # Disjunction constant (≫ max pos diff ~5×10⁵ m)
SEP_HOR_M  = 500.0 * FT2M                          # Horizontal separation (500 ft → 152.4 m)
SEP_VERT_M = 100.0 * FT2M                          # Vertical separation   (100 ft →  30.5 m)
VMAX_XY    = 300.0 * TIMESTEP_DT                   # Max horiz speed/step (300 m/s → 120 000 m/step)
VMAX_Z     = (1000.0 / 60.0) * FT2M * TIMESTEP_DT  # Max vert speed/step (1000 ft/min → 5.08 m/s → ~2032 m/step)
VMIN_2D    = 120.0 * (NM2M / 3600.0) * TIMESTEP_DT # Min 2D speed/step  (120 kts  → ~18 520 m/step)
MAX_ACCEL_XY = 0.25 * 300.0 * TIMESTEP_DT          # Max heading-change per step (m/step) (scales with timestep)

# Simplified flight physics parameters
GAMMA_MAX_TAN   = float(np.tan(5 * np.pi / 180))    # tan(5°) ≈ 0.0875 — max flight path angle
DELTA_SPEED_MAX = 0.5 * (300.0 * TIMESTEP_DT - 120.0 * (NM2M/3600.0) * TIMESTEP_DT)  # max speed Δ/step

# Aircraft parameters (used by downstream fuel analysis only)
S    = 122.6    # Wing area (m²)
mtow = 70000    # Max takeoff weight (kg)
tsfc = 0.00003  # Thrust specific fuel consumption
cd0  = 0.02     # Zero-lift drag coefficient
FUEL_AIRCRAFT = "B737"  # Use B737 fuel data for every flight.
FUEL_MASS_FRACTION = 0.8  # Use 80% of maximum takeoff mass.
fuel_table = FuelFlowTable(FUEL_AIRCRAFT, FUEL_MASS_FRACTION)  # Build the PWL fuel table.

## 1.2. Define STAR fixes — projected to (x, y) metres from DTW, altitude in metres
script_dir = Path(__file__).parent
star_fixes_xyz, x_vals, y_vals, z_vals_max, z_vals_min = load_star_fixes_xyz(
    script_dir / "Input" / "star_fixes.csv", DTW_LAT0, DTW_LON0)

## 1.3. Cluster-based flight selection
QUERY_TIME         = "08:11"   # Set QUERY_TIME ("HH:MM" UTC) to automatically select all flights entering
CLUSTER_WINDOW_MIN = 15        # within CLUSTER_WINDOW_MIN of that time from entry_exit_points.csv.
_eep_csv  = Path(__file__).parent / "Input" / "entry_exit_points.csv"

flights_to_optimize, _cluster, _t_start, _t_end, _span = select_cluster(
    _eep_csv, QUERY_TIME, CLUSTER_WINDOW_MIN)

# Override: restrict to a manual subset (set to None to use the full cluster)
FLIGHTS_SUBSET = ["DAL1120_KMSNtoKDTW", "DAL1655_KPHLtoKDTW", "EDV5018_CYULtoKDTW", "EDV5043_KOMAtoKDTW", "EDV5455_KGRBtoKDTW", "NKS1449_KFLLtoKDTW"]
FLIGHTS_SUBSET = FLIGHTS_SUBSET[:args.flight_count]
if FLIGHTS_SUBSET:
    _cluster           = [(ac, t) for ac, t in _cluster if ac in FLIGHTS_SUBSET]
    flights_to_optimize = [ac for ac in flights_to_optimize if ac in FLIGHTS_SUBSET]
    if len(flights_to_optimize) != args.flight_count:
        raise ValueError("The requested flight subset is not present in the selected cluster.")
    _t_start, _t_end   = _cluster[0][1], _cluster[-1][1]
    _span              = (_t_end - _t_start).total_seconds() / 60

## 1.4. Load flight data — positions converted to metres, altitude to metres
csv_path = script_dir / "Input" / "entry_exit_points.csv"
grid_start = _t_start
flights, GRID_EPOCH_UTC = load_flights_utc_xyz(
    csv_path, flights_to_optimize, TIMESTEP_DT, DTW_LAT0, DTW_LON0, grid_epoch=grid_start)
# Indices are local to each flight; entry_epoch_s retains its exact shared-clock offset.
flights["flight_entry_timestep"] = 0

## 1.5. Determine time steps
N_steps, max_entry_k = compute_time_grid(flights, N_STEPS_HORIZON)
_epoch_end_utc = GRID_EPOCH_UTC + pd.Timedelta(seconds=flights.entry_epoch_s.max() + (N_steps - 1) * TIMESTEP_DT)
N_flights = len(flights)

## Shared weather scenario; STAR selection below remains an optimization decision.
aircraft = []
for _, flight in flights.iterrows():
    nearest = int(np.argmin(np.hypot(np.asarray(x_vals) - flight.entry_x, np.asarray(y_vals) - flight.entry_y)))
    aircraft.append({
        "acId": flight.acId, "acType": FUEL_AIRCRAFT, "entry_seconds": float(flight.entry_epoch_s),
        "entry_xyz_m": [float(flight.entry_x), float(flight.entry_y), float(flight.entry_z)],
        "star_fix": list(star_fixes_xyz)[nearest], "star_xy_m": [x_vals[nearest], y_vals[nearest]],
        "star_altitude_bounds_m": [z_vals_min[nearest], z_vals_max[nearest]],
    })
weather_origin = np.asarray(aircraft[0]["entry_xyz_m"][:2])
weather_direction = np.asarray(aircraft[0]["star_xy_m"]) - weather_origin
route_length = float(np.linalg.norm(weather_direction))
if route_length <= 0:
    raise ValueError("The weather reference route must have nonzero length.")
along = weather_direction / route_length
weather_basis = np.column_stack([along, [-along[1], along[0]]])
weather_boxes = [[0.25 * route_length, -70000.0, 0.85 * route_length, 70000.0]]
for flight in aircraft[1:args.weather_regions]:
    center = (0.5 * (np.asarray(flight["entry_xyz_m"][:2]) + np.asarray(flight["star_xy_m"]))
              - weather_origin) @ weather_basis
    weather_boxes.append([float(center[0] - 25000), float(center[1] - 20000),
                          float(center[0] + 25000), float(center[1] + 20000)])
if args.w3_match_w1_size:
    w1, w3 = np.asarray(weather_boxes[0]), np.asarray(weather_boxes[2])
    center = (w3[:2] + w3[2:]) / 2
    half_size = (w1[2:] - w1[:2]) / 2
    weather_boxes[2] = np.r_[center - half_size, center + half_size].tolist()
scenario = {
    "script": Path(__file__).name, "aircraft": aircraft, "acType": FUEL_AIRCRAFT,
    "mass_fraction": FUEL_MASS_FRACTION, "entry_utc": GRID_EPOCH_UTC.isoformat(),
    "time_step_seconds": TIMESTEP_DT, "horizon_intervals": N_steps - 1,
    "weather_enabled": True, "weather_windows_s": WEATHER_WINDOWS_S,
    "weather_start_s": min(start for start, _ in WEATHER_WINDOWS_S),
    "weather_end_s": max(end for _, end in WEATHER_WINDOWS_S),
    "weather_boxes_uv_m": weather_boxes, "origin_xy_m": weather_origin.tolist(),
    "basis": weather_basis.tolist(), "clearance_m": 1000.0, "hold_speed_mps": HOLD_SPEED_MPS,
    "hold_location_free": True, "hold_altitude_free": True, "entry_holding_allowed": True,
    "holding_forced": False, "holding_cost_factor": HOLD_COST_FACTOR,
    "horizontal_separation_m": SEP_HOR_M, "vertical_separation_m": SEP_VERT_M,
    "objective_weights": {"time": w_time, "distance": w_dist, "fuel": w_fuel, "smoothness": 0.0,
                          "vertical": w_z, "heading": w_accel, "descent": w_descent, "final_altitude": w_alt_final},
    "destination_policy": "Optimizer selects any STAR; nearest STAR only defines weather and initial hints.",
    "star_candidates": star_fixes_xyz,
    "timing": "Local grid per flight, with exact shared-clock entry offsets.",
}
output_dir = args.output_dir or script_dir / "Output" / "v18_holding_comparison" / (
    "v18.1_" + pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f"))
output_dir.mkdir(parents=True, exist_ok=False)
(output_dir / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")

## 1.6. Print problem setup summary
print(" === PROBLEM SETUP ===")

print(f"\n{'='*52}")
print(f"  CLUSTER SELECTION  (query: {QUERY_TIME} UTC, window: {CLUSTER_WINDOW_MIN} min)")
print(f"{'='*52}")
print(f"  Time span : {_t_start.strftime('%Y-%m-%d %H:%M:%S')} → {_t_end.strftime('%H:%M:%S')} UTC  ({_span:.1f} min)")
print(f"  Flights   : {N_flights}")

_utc_map = {ac: t.strftime('%H:%M:%S UTC') for ac, t in _cluster} # Map acId → entry time for display
_flights_display = flights.copy() # Add entry_utc column for display
_flights_display.insert(0, 'entry_utc', _flights_display['acId'].map(_utc_map)) # Add entry_utc column for display
print(f"{_flights_display.to_string(index=False)}")

print(f"{'='*52}\n")

print(f"Shared UTC epoch: {GRID_EPOCH_UTC.strftime('%Y-%m-%d %H:%M:%S')} UTC (first entry)")
print(f"Grid step size  : {int(TIMESTEP_DT)}s  ({TIMESTEP_DT / 60:.0f} min per step)")
print(f"Time steps      : {N_steps} per flight; k=0 is each flight's own entry. Latest horizon end: {_epoch_end_utc}")
print(f"max_entry_k={max_entry_k}, horizon={N_STEPS_HORIZON} extra steps...\n")


########################
# 2. MILP Optimization #
########################
print(" === MILP OPTIMIZATION ===")
# 2.1. Create model
env = Env(empty=True)
env.setParam("OutputFlag", 0)   # disable all output from Gurobi
env.start()

m = Model("mip1", env=env)
m.Params.OutputFlag = 1
m.Params.NonConvex = 0  # Require a linear MILP model.
m.Params.TimeLimit = TIME_LIMIT  # Apply the solver time limit.
m.Params.MIPGap = MIP_GAP  # Apply the target optimality gap.
m.Params.Threads = min(4, os.cpu_count() or 1)  # Use at most four CPU threads.
m.Params.OutputFlag = 1  # Enable Gurobi logging.
m.Params.LogToConsole = 1  # Print the solver log in the terminal.
#print("Model created...")

# 2.2. Create decision variables
# i) Position variables: x = east (m), y = north (m) from DTW; z = altitude (m MSL)
f_x = [m.addVars(range(N_steps), lb=-GRB.INFINITY, name=f"f{i}_x") for i in range(1, N_flights+1)] # East  position (m)
f_y = [m.addVars(range(N_steps), lb=-GRB.INFINITY, name=f"f{i}_y") for i in range(1, N_flights+1)] # North position (m)
f_z = [m.addVars(range(N_steps), ub=fuel_table.axes[1][-1], name=f"f{i}_z") for i in range(1, N_flights+1)] # Keep altitude inside the fuel table.
u_x = [m.addVars(range(N_steps),                   name=f"uf{i}_x") for i in range(1, N_flights+1)] # |Δeast|  per step (m)
u_y = [m.addVars(range(N_steps),                   name=f"uf{i}_y") for i in range(1, N_flights+1)] # |Δnorth| per step (m)
u_z = [m.addVars(range(N_steps),                   name=f"uf{i}_z") for i in range(1, N_flights+1)] # |Δalt|   per step (m)

# ii) STAR fix selection (binary)
fix_names = list(star_fixes_xyz)
n_fixes   = len(fix_names)
fix_sel = [m.addVars(n_fixes, vtype=GRB.BINARY, name=f"y{i+1}") for i in range(N_flights)]

# iii) Auxiliary variables
d_x, d_y, d_z   = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("dx",  "dy", "dz")]  # Δx, Δy, Δz per step (m)
dd_x,   dd_y     = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("ddx", "ddy"     )]  # Δ(Δx), Δ(Δy) — heading-rate proxy (m)
u_dd_x, u_dd_y   = [m.addVars(N_flights, N_steps, lb=0.0,           vtype=GRB.CONTINUOUS, name=n) for n in ("udx", "udy"     )]  # |ddx|, |ddy| (m)
accel_cost, speed_2d = [m.addVars(N_flights, N_steps, lb=0.0,       vtype=GRB.CONTINUOUS, name=n) for n in ("accel_xy", "speed")]  # heading-rate cost, 2D speed (m/step)
fix_reached, sep_bypass, fix_enters = [m.addVars(N_flights, N_steps, lb=0.0, vtype=GRB.BINARY, name=n) for n in ("is_end", "landed", "delta_end")] 
    # Binary flags: fix_reached=1 if flight has reached its STAR fix, sep_bypass=1 if separation constraint is bypassed, fix_enters=1 if flight enters fix at that step
k_arrive = m.addVars(N_flights, lb=0, vtype=GRB.INTEGER, name="exit_k", ub=N_steps-1) # Integer variable for the timestep at which each flight reaches its STAR fix (k_arrive)
hold = m.addVars(N_flights, N_steps, vtype=GRB.BINARY, name="hold_interval")
route_inactive = m.addVars(N_flights, N_steps, vtype=GRB.BINARY, name="route_inactive")
hold_step_kg, hold_budgets = {}, []
for i in range(N_flights):
    m.addConstr(hold[i, 0] == 0)
    m.addConstr(route_inactive[i, 0] == 1)
    hold_budgets.append(m.addConstr(quicksum(hold[i, k] for k in range(N_steps)) <= 0,
                                   name=f"holding_budget_{i}"))
    for k in range(1, N_steps):
        hold[i, k].Start = 0
        m.addConstr(hold[i, k] + fix_reached[i, k] <= 1)
        m.addConstr(route_inactive[i, k] == fix_reached[i, k-1] + hold[i, k])
        hold_step_kg[i, k] = add_hold_interval(
            m, hold[i, k], (f_x[i][k-1], f_y[i][k-1], f_z[i][k-1]),
            (f_x[i][k], f_y[i][k], f_z[i][k]), fuel_table,
            HOLD_SPEED_MPS, TIMESTEP_DT, f"hold_{i}_{k}")
#print("Decision variables created...")

# 2.3. Warm start Gurobi with .Start for each parameter
start_routes = {}  # Store simple routes used as solver hints.
for flight_index, flight in flights.iterrows():  # Create one hint for each flight.
    entry_step = int(flight['flight_entry_timestep'])  # Read when the flight enters.
    start = np.array([flight['entry_x'], flight['entry_y'], flight['entry_z']])  # Build the entry point.
    chosen = int(np.argmin(np.hypot(np.asarray(x_vals) - start[0], np.asarray(y_vals) - start[1])))  # Pick the nearest STAR fix.
    start_altitude = z_vals_min[chosen] + (z_vals_max[chosen] - z_vals_min[chosen]) * (flight_index + 1) / (N_flights + 1)  # Give flights different final altitudes.
    finish = np.array([x_vals[chosen], y_vals[chosen], start_altitude])  # Build the hinted final point.
    route = np.array([start + np.clip((step - entry_step) / (N_steps - 1 - entry_step), 0, 1) * (finish - start)  # Interpolate a straight route.
                      for step in range(N_steps)])  # Create a point at every time step.
    start_routes[flight_index] = route  # Save the route for fuel hints.
    for step, point in enumerate(route):  # Assign every hinted point.
        f_x[flight_index][step].Start, f_y[flight_index][step].Start, f_z[flight_index][step].Start = point  # Hint the position.
        fix_reached[flight_index, step].Start = float(step == N_steps - 1)  # Hint arrival at the final step.
        fix_enters[flight_index, step].Start = float(step == N_steps - 1)  # Hint the arrival transition.
    for fix_index in range(n_fixes):  # Hint the selected STAR fix.
        fix_sel[flight_index][fix_index].Start = float(fix_index == chosen)  # Select only the nearest fix.
    k_arrive[flight_index].Start = N_steps - 1  # Hint the final arrival step.

# 2.4. Define Objective Function
obj = LinExpr()
fuel_step_kg = {}  # Store fuel used by each flight segment.

for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    
    for k in range(k_entry + 1, N_steps):
        active = 1 - fix_reached[i, k-1]  # Count the interval that reaches the fix.
        obj += w_time * TIMESTEP_DT * (active - (1 - HOLD_COST_FACTOR) * hold[i, k])
        obj += HOLD_COST_FACTOR * (
            w_dist * HOLD_SPEED_MPS * TIMESTEP_DT * hold[i, k] + w_fuel * hold_step_kg[i, k])

        # Step differences and L1 absolutes — all in metres
        for diff, u, cur, prv in zip(
            (d_x[i,k],      d_y[i,k],      d_z[i,k]   ),
            (u_x[i][k-1],   u_y[i][k-1],   u_z[i][k-1]),
            (f_x[i][k],     f_y[i][k],     f_z[i][k]  ),
            (f_x[i][k-1],   f_y[i][k-1],   f_z[i][k-1]),
        ):
            m.addConstr(diff == cur - prv)
            m.addConstr(u    == abs_(diff))

        segment_speed = add_horizontal_speed(m, d_x[i,k] / TIMESTEP_DT, d_y[i,k] / TIMESTEP_DT, f"speed_{i}_{k}")  # Approximate horizontal speed linearly.
        m.addConstr(speed_2d[i,k] == TIMESTEP_DT * segment_speed)  # Convert speed back to distance per step.
        # obj += w_smooth * (u_x[i][k-1] + u_y[i][k-1] + w_z * u_z[i][k-1])    # Smoothness penalty (L1, m)
        obj += w_dist  * speed_2d[i,k]                                       # Distance penalty (Euclidean m/step)
        obj += w_descent * d_z[i, k]                                         # descent reward (d_z < 0 descending → reduces objective)

        reachable_change = (k - k_entry - 0.5) * VMAX_Z  # Find the reachable altitude change.
        altitude_bounds = (  # Limit this segment's fuel-table altitude.
            max(0.0, flights.iloc[i]['entry_z'] - reachable_change),  # Set its lowest reachable altitude.
            min(fuel_table.axes[1][-1], flights.iloc[i]['entry_z'] + reachable_change),  # Set its highest reachable altitude.
        )  # Finish the altitude bounds.
        before_start, after_start = start_routes[i][k-1], start_routes[i][k]
        start_velocity = (after_start - before_start) / TIMESTEP_DT
        start_point = (
            horizontal_speed_pwl(*start_velocity[:2]),
            (before_start[2] + after_start[2]) / 2,
            start_velocity[2],
        )
        fuel_step_kg[i, k] = fuel_table.add_segment(  # Add the PWL fuel constraints.
            m, speed_2d[i,k] / TIMESTEP_DT, (f_z[i][k-1] + f_z[i][k]) / 2,  # Pass speed and average altitude.
            d_z[i,k] / TIMESTEP_DT, route_inactive[i,k], TIMESTEP_DT,
            f"fuel_{i}_{k}", altitude_bounds, start_point=start_point,
        )  # Finish this segment's fuel model.
        obj += w_fuel * fuel_step_kg[i, k]  # Add fuel cost to the objective.

        # Heading-rate penalty (k_entry+2 onward)
        if k >= k_entry + 2:
            for dd, ud, df, dfp in zip(
                (dd_x[i,k],    dd_y[i,k]  ),
                (u_dd_x[i,k],  u_dd_y[i,k]),
                (d_x[i,k],     d_y[i,k]   ),
                (d_x[i,k-1],   d_y[i,k-1] ),
            ):
                m.addConstr(dd == df - dfp) # dd = Δ(Δx) = Δx[k] - Δx[k-1]
                m.addConstr(ud == abs_(dd)) # ud = |dd| = |Δ(Δx)| = |Δx[k] - Δx[k-1]|
            hold_transition = m.addVar(vtype=GRB.BINARY, name=f"hold_transition_{i}_{k}")
            m.addGenConstrOr(hold_transition, [hold[i,k], hold[i,k-1]])
            m.addGenConstrIndicator(hold_transition, 0, accel_cost[i,k] == u_dd_x[i,k] + u_dd_y[i,k])
            m.addGenConstrIndicator(hold_transition, 1, accel_cost[i,k] == 0)
            obj += w_accel * accel_cost[i,k] # Heading-rate penalty (m/step)

#print("Objective function created...")

# 2.5. Define constraints
# i) Entry point constraints (hold each flight at its entry position until k_entry)
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']
    for k in range(k_entry + 1):
        m.addConstr(f_x[i][k] == flights.iloc[i]['entry_x'], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(f_y[i][k] == flights.iloc[i]['entry_y'], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(f_z[i][k] == flights.iloc[i]['entry_z'], f"c_pre_entry_z_{i}_t{k}")

# ii) STAR fix (exit point) constraints — pin final position to one chosen fix (metres)
for j in range(N_flights):
    m.addConstr(quicksum(fix_sel[j][i] for i in range(n_fixes)) == 1, f"one_fix{j+1}")
    m.addConstr(f_x[j][N_steps-1] == LinExpr(x_vals,     fix_sel[j].values()), f"x_choice{j+1}")
    m.addConstr(f_y[j][N_steps-1] == LinExpr(y_vals,     fix_sel[j].values()), f"y_choice{j+1}")
    m.addConstr(f_z[j][N_steps-1] <= LinExpr(z_vals_max, fix_sel[j].values()), f"z_choice_max{j+1}")
    m.addConstr(f_z[j][N_steps-1] >= LinExpr(z_vals_min, fix_sel[j].values()), f"z_choice_min{j+1}")
    obj += w_alt_final * f_z[j][N_steps-1]  # penalize high final altitude (m)

# iii) Arrival flag (fix_reached / fix_enters) logic
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']

    # (1) Before entry: fix_reached must be 0
    for k in range(k_entry + 1):
        m.addConstr(fix_reached[i, k] == 0, f"fix_reached_pre_entry_{i}_{k}")
        m.addConstr(fix_enters[i, k] == 0, f"fix_enters_pre_entry_{i}_{k}")
        m.addConstr(sep_bypass[i, k] == 0, f"separation_at_entry_{i}_{k}")  # Keep separation active before entry.

    # (2) After entry: fix_reached is monotone (0→1); position freezes once reached
    for k in range(k_entry + 1, N_steps):  # Start arrival rules at the first route point.
        m.addConstr(sep_bypass[i, k] == fix_reached[i, k-1], f"separation_after_arrival_{i}_{k}")  # Bypass separation only after arrival.
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

# iv) Max speed constraints — bound per-step displacement in metres
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']
    for k in range(k_entry + 1, N_steps):
        m.addConstr(f_x[i][k] - f_x[i][k-1] <=  VMAX_XY)
        m.addConstr(f_y[i][k] - f_y[i][k-1] <=  VMAX_XY)
        m.addConstr(f_z[i][k] - f_z[i][k-1] <=  VMAX_Z)

        m.addConstr(f_x[i][k-1] - f_x[i][k] <=  VMAX_XY)
        m.addConstr(f_y[i][k-1] - f_y[i][k] <=  VMAX_XY)
        m.addConstr(f_z[i][k-1] - f_z[i][k] <=  VMAX_Z)
        # Minimum 2D speed while airborne (m/step)
        m.addConstr(speed_2d[i,k] >= VMIN_2D / NORM_COSINE * (1 - route_inactive[i,k]), f"min_speed_{i}_{k}")

# vi) Simplified flight physics constraints
    # These ensure MILP waypoints are physically achievable by the NLP aerodynamic model. 
for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    for k in range(k_entry + 1, N_steps):
        # 1. Flight path angle coupling Prevents unrealistically steep climb / descent
        m.addConstr(
            f_z[i][k] - f_z[i][k-1] <=  GAMMA_MAX_TAN * speed_2d[i,k] * NORM_COSINE,  # Correct for the PWL speed overestimate.
            f"fpa_up_{i}_{k}") # Climb flight path angle: |Δz| ≤ tan(5°) × speed_2d
        m.addConstr(
            f_z[i][k-1] - f_z[i][k] <=  GAMMA_MAX_TAN * speed_2d[i,k] * NORM_COSINE,  # Correct for the PWL speed overestimate.
            f"fpa_down_{i}_{k}") # Descent flight path angle: |Δz| ≤ tan(5°) × speed_2d 

        # 2. Speed consistency |speed_2d[k] - speed_2d[k-1]| ≤ DELTA_SPEED_MAX
            # Prevents step-to-step speed jumps the NLP aerodynamics cannot match.
        if k >= k_entry + 2:
            bypass = BIG_M * (fix_reached[i,k] + fix_reached[i,k-1] + hold[i,k] + hold[i,k-1])
            m.addConstr(
                speed_2d[i,k] - speed_2d[i,k-1] <=  DELTA_SPEED_MAX + bypass,
                f"speed_cons_up_{i}_{k}")
            m.addConstr(
                speed_2d[i,k-1] - speed_2d[i,k] <=  DELTA_SPEED_MAX + bypass,
                f"speed_cons_dn_{i}_{k}")

# vii) Hard heading-change bound — prevents physically impossible sharp turns
for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    for k in range(k_entry + 2, N_steps):
        heading_bound = MAX_ACCEL_XY + BIG_M * (hold[i,k] + hold[i,k-1])
        m.addConstr(dd_x[i,k] <=  heading_bound, f"accel_x_ub_{i}_{k}")
        m.addConstr(dd_x[i,k] >= -heading_bound, f"accel_x_lb_{i}_{k}")
        m.addConstr(dd_y[i,k] <=  heading_bound, f"accel_y_ub_{i}_{k}")
        m.addConstr(dd_y[i,k] >= -heading_bound, f"accel_y_lb_{i}_{k}")

# v) Separation constraints — all thresholds in metres
# Use exact entry offsets and full holding envelopes rather than waypoint-only checks.
flight_variables = [
    ([f_x[i], f_y[i], f_z[i]], {k: fix_reached[i, k] for k in range(N_steps)},
     {k: hold[i, k] for k in range(N_steps)}, {k: fuel_step_kg[i, k] for k in range(1, N_steps)},
     {k: hold_step_kg[i, k] for k in range(1, N_steps)})
    for i in range(N_flights)
]
add_separation(m, scenario, flight_variables)

# viii) Protect each region only during intervals overlapping its shared-clock window.
for i, flight in flights.iterrows():
    for k in range(1, N_steps):
        for region, bounds in active_weather_cells(
                flight.entry_epoch_s + (k - 1) * TIMESTEP_DT,
                flight.entry_epoch_s + k * TIMESTEP_DT, scenario):
            add_weather_interval(
                m, [(f_x[i][j], f_y[i][j]) for j in (k - 1, k)], hold[i, k],
                [bounds], weather_origin, weather_basis, HOLD_SPEED_MPS,
                TIMESTEP_DT, scenario["clearance_m"], f"weather_{i}_{k}_w{region + 1}")

#print("Constraints created...")

# 2.6. Initiate optimization
m.setObjective(obj, GRB.MINIMIZE)  # Set the completed fuel-aware objective.
case_results, case_schedules = [], {}
try:
    for holding_enabled in (False, True):
        case_name = "holding_on" if holding_enabled else "holding_off"
        case_dir = output_dir / case_name
        case_dir.mkdir()
        for budget in hold_budgets:
            budget.RHS = N_steps - 1 if holding_enabled else 0
        initialization_seconds = 0.0
        if not holding_enabled:
            initialization_seconds = seed_geometric_start(m, min(30.0, TIME_LIMIT), case_dir / "initialization.log")
        m.Params.LogFile = str(case_dir / "solver.log")
        weather_label = ", ".join(f"W{j + 1}: {start:g}-{end:g}s"
                                  for j, (start, end) in enumerate(WEATHER_WINDOWS_S))
        print(f"\n=== {case_name.upper()} | same weather: {weather_label} since first entry ===")
        m.optimize()

        # 3. OUTPUT: all extraction, validation, summaries and plots live in the helper.
        case_scenario = deepcopy(scenario)
        if m.SolCount > 0:
            for i, flight in enumerate(case_scenario["aircraft"]):
                chosen = next(j for j in range(n_fixes) if fix_sel[i][j].X > 0.5)
                flight.update(star_fix=fix_names[chosen], star_xy_m=[x_vals[chosen], y_vals[chosen]],
                              star_altitude_bounds_m=[z_vals_min[chosen], z_vals_max[chosen]])
        result, schedules = write_case_output(
            m, case_scenario, flight_variables, fuel_table, case_dir, holding_enabled,
            initialization_seconds=initialization_seconds,
            warm_start_labels=["holding_off_incumbent"] if holding_enabled else ["geometric_incumbent"])
        case_results.append(result)
        case_schedules[case_name] = schedules
        if not holding_enabled:
            for variable in m.getVars():
                variable.Start = variable.X
    write_comparison_output(scenario, case_results, case_schedules, output_dir)
finally:
    m.dispose()
    env.dispose()