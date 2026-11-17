from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
from Functions.fuel_model import *
from Functions.fuel_pwl import (  # Import the PWL fuel tools.
    FuelFlowTable, add_horizontal_speed, NORM_COSINE,  # Import fuel and speed helpers.
)  # Finish the PWL imports.
from Functions.utilities import (
    load_flights_utc_xyz, load_star_fixes_xyz, compute_time_grid,
    print_chosen_star_fixes_xyz, print_exit_times, print_flight_results,
    save_trajectory_csv_xyz, print_waypoint_table_xyz,
    print_separation_check_xyz, _cluster_flights_by_time, _find_cluster_for_query,
    select_cluster, write_simulation_summary, record_terminal_output
)
import os
import argparse
import json

parser = argparse.ArgumentParser(description="Optional holding at any feasible grid position and airborne interval.")
parser.add_argument("--time-limit", type=float, default=60.0)
parser.add_argument("--max-hold-steps", type=int, default=2)
parser.add_argument("--force-hold-steps", type=int, default=0, help="Force this many entry circuits for the first flight.")
parser.add_argument("--flight-limit", type=int, default=1)
parser.add_argument("--clear-weather", action="store_true")
parser.add_argument("--run-nlp", action="store_true")
parser.add_argument("--nlp-time-limit", type=float, default=30.0, help="IPOPT CPU seconds per flight.")
parser.add_argument("--plots", action="store_true")
parser.add_argument("--output-dir", type=Path, default=None)
args = parser.parse_args()
if args.flight_limit != 1 or args.force_hold_steps != 0 or args.max_hold_steps < 0:
    parser.error("This scenario uses one flight, zero forced holds, and a nonnegative optional-hold limit.")
if args.run_nlp:
    parser.error("The entry-only NLP handoff cannot process interior holds; use this scenario's MILP fuel report.")
if not 0 <= args.force_hold_steps <= args.max_hold_steps or args.time_limit <= 0 or args.nlp_time_limit <= 0:
    parser.error("Require 0 <= force-hold-steps <= max-hold-steps and positive solve limits.")
if args.flight_limit is not None and args.flight_limit < 1:
    parser.error("flight-limit must be positive.")
if not args.plots:
    os.environ["MPLBACKEND"] = "Agg"
from Functions.free_holding import add_hold_interval, add_weather_interval, report_free_holding
# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

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
N_STEPS_HORIZON = 6
TIME_LIMIT = args.time_limit
MIP_GAP = 0.01  # Accept a solution within a 1% gap.
HOLD_SPEED_MPS = 210.0 * NM2M / 3600.0
HOLD_ENVELOPE_M = HOLD_SPEED_MPS * TIMESTEP_DT / 2

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
if args.flight_limit is not None:
    FLIGHTS_SUBSET = FLIGHTS_SUBSET[:args.flight_limit]
if FLIGHTS_SUBSET:
    _cluster           = [(ac, t) for ac, t in _cluster if ac in FLIGHTS_SUBSET]
    flights_to_optimize = [ac for ac in flights_to_optimize if ac in FLIGHTS_SUBSET]
    _t_start, _t_end   = _cluster[0][1], _cluster[-1][1]
    _span              = (_t_end - _t_start).total_seconds() / 60

## 1.4. Load flight data — positions converted to metres, altitude to metres
csv_path = script_dir / "Input" / "entry_exit_points.csv"
grid_start = _t_start.normalize() + pd.Timedelta(  # Start the grid near the first flight.
    seconds=int((_t_start - _t_start.normalize()).total_seconds() // TIMESTEP_DT) * TIMESTEP_DT  # Align it to a full time step.
)  # Finish the aligned start time.
flights, GRID_EPOCH_UTC = load_flights_utc_xyz(
    csv_path, flights_to_optimize, TIMESTEP_DT, DTW_LAT0, DTW_LON0, grid_epoch=grid_start)  # Use the shorter aligned grid.

## 1.5. Determine time steps
N_steps, max_entry_k = compute_time_grid(flights, N_STEPS_HORIZON)
_epoch_end_utc = GRID_EPOCH_UTC + pd.Timedelta(seconds=(N_steps - 1) * TIMESTEP_DT)
N_flights = len(flights)
scenario_origin = flights.iloc[0][['entry_x', 'entry_y']].to_numpy(dtype=float)
scenario_fix = int(np.argmin(np.hypot(np.asarray(x_vals) - scenario_origin[0], np.asarray(y_vals) - scenario_origin[1])))
scenario_finish = np.array([x_vals[scenario_fix], y_vals[scenario_fix]])
route_length = np.linalg.norm(scenario_finish - scenario_origin)
along = (scenario_finish - scenario_origin) / route_length
across = np.array([-along[1], along[0]])
scenario_basis = np.column_stack([along, across])
hint_anchor = scenario_origin + 0.35 * (scenario_finish - scenario_origin)
hint_altitude = float(flights.iloc[0]['entry_z'] - 0.9 * VMAX_Z)
hold_leg = HOLD_SPEED_MPS * TIMESTEP_DT / 4
hold_radius = hold_leg / np.pi
hold_along = 0.35 * route_length
pocket_low = np.array([hold_along - hold_radius - 1500, -1500])
pocket_high = np.array([hold_along + hold_leg + hold_radius + 1500, 2 * hold_radius + 1500])
weather_boxes = [
    [20000, -70000, pocket_low[0], 70000],
    [pocket_high[0], -70000, 160000, 70000],
    [pocket_low[0], -70000, pocket_high[0], pocket_low[1]],
    [pocket_low[0], pocket_high[1], pocket_high[0], 70000],
]
entry_seconds = int(flights.iloc[0]['flight_entry_timestep']) * TIMESTEP_DT
scenario = {
    "weather_enabled": not args.clear_weather, "weather_start_s": entry_seconds + TIMESTEP_DT,
    "weather_end_s": entry_seconds + 2 * TIMESTEP_DT, "weather_boxes_uv_m": weather_boxes,
    "origin_xy_m": scenario_origin.tolist(), "basis": scenario_basis.tolist(),
    "hold_location_free": True, "hold_altitude_free": True, "entry_holding_allowed": True,
    "max_hold_steps": args.max_hold_steps, "time_step_seconds": TIMESTEP_DT,
    "hold_speed_mps": HOLD_SPEED_MPS, "clearance_m": 1000.0, "holding_forced": False,
}

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

print(f"UTC grid epoch  : {GRID_EPOCH_UTC.strftime('%Y-%m-%d %H:%M:%S')} UTC (k=0)")
print(f"Grid step size  : {int(TIMESTEP_DT)}s  ({TIMESTEP_DT / 60:.0f} min per step)")
print(f"Time steps      : {N_steps}  (k=0 → {GRID_EPOCH_UTC.strftime('%H:%M')} UTC, k={N_steps-1} → {_epoch_end_utc.strftime('%H:%M')} UTC)")
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
m.addConstr(fix_sel[0][scenario_fix] == 1, "fixed_scenario_star")

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
hold_step_kg = {}
for i, flight in flights.iterrows():
    entry_step = int(flight['flight_entry_timestep'])
    for k in range(N_steps):
        hold[i, k].Start = float(not args.clear_weather and args.max_hold_steps >= 1 and k == entry_step + 2)
        if k <= entry_step:
            m.addConstr(hold[i, k] == 0)
            m.addConstr(route_inactive[i, k] == 1)
        else:
            m.addConstr(route_inactive[i, k] == fix_reached[i, k-1] + hold[i, k])
            m.addConstr(hold[i, k] + fix_reached[i, k] <= 1)
            hold_step_kg[i, k] = add_hold_interval(
                m, hold[i, k], (f_x[i][k-1], f_y[i][k-1], f_z[i][k-1]),
                (f_x[i][k], f_y[i][k], f_z[i][k]), fuel_table,
                HOLD_SPEED_MPS, TIMESTEP_DT, f"hold_{i}_{k}")
    m.addConstr(quicksum(hold[i, k] for k in range(N_steps)) <= args.max_hold_steps)
#print("Decision variables created...")

# 2.3. Warm start Gurobi with .Start for each parameter
start_routes = {}  # Store simple routes used as solver hints.
for flight_index, flight in flights.iterrows():  # Create one hint for each flight.
    entry_step = int(flight['flight_entry_timestep'])  # Read when the flight enters.
    start = np.array([flight['entry_x'], flight['entry_y'], flight['entry_z']])  # Build the entry point.
    chosen = int(np.argmin(np.hypot(np.asarray(x_vals) - start[0], np.asarray(y_vals) - start[1])))  # Pick the nearest STAR fix.
    start_altitude = z_vals_min[chosen] + (z_vals_max[chosen] - z_vals_min[chosen]) * (flight_index + 1) / (N_flights + 1)  # Give flights different final altitudes.
    finish = np.array([x_vals[chosen], y_vals[chosen], start_altitude])  # Build the hinted final point.
    release_step = entry_step + (args.force_hold_steps if flight_index == 0 else 0)
    route = np.array([start + np.clip((step - release_step) / (N_steps - 1 - release_step), 0, 1) * (finish - start)  # Interpolate a straight route.
                      for step in range(N_steps)])  # Create a point at every time step.
    hinted_arrival = N_steps - 1
    if not args.clear_weather and args.max_hold_steps >= 1:
        holding_point = np.r_[hint_anchor, hint_altitude]
        route[entry_step + 1] = holding_point
        route[entry_step + 2] = holding_point
        route[entry_step + 3] = holding_point + 0.4 * (finish - holding_point)
        route[entry_step + 4] = holding_point + 0.75 * (finish - holding_point)
        route[entry_step + 5:] = finish
        hinted_arrival = entry_step + 5
    elif not args.clear_weather:
        detour_uv = np.array([[0, 0], [15000, -30000], [15000, -55000],
                              [40000, -70000], [85000, -65000], [140000, -35000],
                              [route_length, 0]])
        route[:, :2] = detour_uv @ scenario_basis.T + scenario_origin
    start_routes[flight_index] = route  # Save the route for fuel hints.
    for step, point in enumerate(route):  # Assign every hinted point.
        f_x[flight_index][step].Start, f_y[flight_index][step].Start, f_z[flight_index][step].Start = point  # Hint the position.
        fix_reached[flight_index, step].Start = float(step >= hinted_arrival)
        fix_enters[flight_index, step].Start = float(step == hinted_arrival)
    for fix_index in range(n_fixes):  # Hint the selected STAR fix.
        fix_sel[flight_index][fix_index].Start = float(fix_index == chosen)  # Select only the nearest fix.
    k_arrive[flight_index].Start = hinted_arrival

# 2.4. Define Objective Function
obj = LinExpr()
fuel_step_kg = {}  # Store fuel used by each flight segment.

for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    
    for k in range(k_entry + 1, N_steps):
        active = 1 - fix_reached[i, k-1]  # Count the interval that reaches the fix.
        obj += w_time * TIMESTEP_DT * active            # penalize each active step → minimizes flight time
        obj += w_fuel * hold_step_kg[i, k] + w_dist * HOLD_SPEED_MPS * TIMESTEP_DT * hold[i, k]

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
        obj += w_smooth * (u_x[i][k-1] + u_y[i][k-1] + w_z * u_z[i][k-1])    # Smoothness penalty (L1, m)
        obj += w_dist  * speed_2d[i,k]                                       # Distance penalty (Euclidean m/step)
        obj += w_descent * d_z[i, k]                                         # descent reward (d_z < 0 descending → reduces objective)

        reachable_change = (k - k_entry - 0.5) * VMAX_Z  # Find the reachable altitude change.
        altitude_bounds = (  # Limit this segment's fuel-table altitude.
            max(0.0, flights.iloc[i]['entry_z'] - reachable_change),  # Set its lowest reachable altitude.
            min(fuel_table.axes[1][-1], flights.iloc[i]['entry_z'] + reachable_change),  # Set its highest reachable altitude.
        )  # Finish the altitude bounds.
        fuel_step_kg[i, k] = fuel_table.add_segment(  # Add the PWL fuel constraints.
            m, speed_2d[i,k] / TIMESTEP_DT, (f_z[i][k-1] + f_z[i][k]) / 2,  # Pass speed and average altitude.
            d_z[i,k] / TIMESTEP_DT, route_inactive[i,k], TIMESTEP_DT,
            f"fuel_{i}_{k}", altitude_bounds,  # Pass the bounds.
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
# Axis-aligned box: 4 horizontal + 2 vertical = 6 binary vars per pair/step
for k in range(N_steps):
    for i in range(N_flights - 1):
        for j in range(i + 1, N_flights):
            if k >= flights.iloc[i]['flight_entry_timestep'] and k >= flights.iloc[j]['flight_entry_timestep']:
                bin_vars = m.addVars(range(6), name=f'bin_{i}_{j}_{k}', vtype=GRB.BINARY)
                hold_i = hold[i, min(k + 1, N_steps - 1)]
                hold_j = hold[j, min(k + 1, N_steps - 1)]
                horizontal_sep = SEP_HOR_M + HOLD_ENVELOPE_M * (hold_i + hold_j)

                m.addConstr(f_x[i][k] - f_x[j][k] >= horizontal_sep - BIG_M*(1 - bin_vars[0]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_y[i][k] - f_y[j][k] >= horizontal_sep - BIG_M*(1 - bin_vars[1]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_z[i][k] - f_z[j][k] >= SEP_VERT_M - BIG_M*(1 - bin_vars[2]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_x[j][k] - f_x[i][k] >= horizontal_sep - BIG_M*(1 - bin_vars[3]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_y[j][k] - f_y[i][k] >= horizontal_sep - BIG_M*(1 - bin_vars[4]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_z[j][k] - f_z[i][k] >= SEP_VERT_M - BIG_M*(1 - bin_vars[5]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])

                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5] >= 1)

if scenario['weather_enabled']:
    for step in range(1, N_steps):
        if step * TIMESTEP_DT <= scenario['weather_start_s'] or (step - 1) * TIMESTEP_DT >= scenario['weather_end_s']:
            continue
        add_weather_interval(m, [(f_x[0][endpoint], f_y[0][endpoint]) for endpoint in (step - 1, step)],
                             hold[0, step], weather_boxes, scenario_origin, scenario_basis,
                             HOLD_SPEED_MPS, TIMESTEP_DT, scenario['clearance_m'], f"weather_{step}")

#print("Constraints created...")

# 2.6. Initiate optimization
import time as _time
m.setObjective(obj, GRB.MINIMIZE)  # Set the completed fuel-aware objective.
solver_log_lines = []
def capture_solver_log(model, where):
    if where == GRB.Callback.MESSAGE:
        solver_log_lines.append(model.cbGet(GRB.Callback.MSG_STRING))

print("Starting optimization...")
_t0 = _time.perf_counter()
m.optimize(capture_solver_log)
_opt_elapsed = _time.perf_counter() - _t0
solver_log = "".join(solver_log_lines)
print(f"Optimization completed in {_opt_elapsed:.1f} s.\n")

_STATUS_NAMES = {1:"LOADED",2:"OPTIMAL",3:"INFEASIBLE",4:"INF_OR_UNBD",5:"UNBOUNDED",
                 6:"CUTOFF",7:"ITERATION_LIMIT",8:"NODE_LIMIT",9:"TIME_LIMIT",10:"SOLUTION_LIMIT",
                 11:"INTERRUPTED",12:"NUMERIC",13:"SUBOPTIMAL",15:"USER_OBJ_LIMIT"}
print(f"Gurobi status: {_STATUS_NAMES.get(m.status, m.status)} ({m.status})")
if m.SolCount == 0:  # Check that Gurobi found a usable route.
    raise RuntimeError(f"No feasible incumbent; solver status: {m.status}")  # Stop before reading missing results.
print(f'Incumbent objective: {m.ObjVal:.4g}; relative gap: {m.MIPGap:.2%}')  # Show solution quality.

# ###########
# 3. OUTPUT #
# ###########
print(" === OUTPUT RESULTS ===")
if m.SolCount > 0:

    # 3.2. Extract and save optimized trajectories
    # Primary columns: f{i}_x/y/z (metres). lat/lon/alt_ft also written for human readability.
    output_dir = args.output_dir or script_dir / "Output" / "v17_free_holding_test" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = write_simulation_summary(output_dir, {
        "Script": Path(__file__).name,
        "Temporary weather": scenario,
        "Report time (UTC)": pd.Timestamp.now(tz="UTC").isoformat(),
        "Input files": f"{csv_path}; {script_dir / 'Input' / 'star_fixes.csv'}",
        "Cluster query (UTC) / window (min)": f"{QUERY_TIME} / {CLUSTER_WINDOW_MIN}",
        "Selected entry window (UTC)": f"{_t_start} to {_t_end}",
        "Flight subset": FLIGHTS_SUBSET,
        "Flight count": N_flights,
        "Grid epoch / end (UTC)": f"{GRID_EPOCH_UTC} / {_epoch_end_utc}",
        "Time step (s) / grid points / extra horizon steps": f"{TIMESTEP_DT} / {N_steps} / {N_STEPS_HORIZON}",
        "Projection origin (latitude, longitude)": f"{DTW_LAT0}, {DTW_LON0}",
        "Horizontal / vertical separation (m)": f"{SEP_HOR_M} / {SEP_VERT_M}",
        "Separation policy": "Waypoint checks through arrival; bypass only after arrival. Not a continuous-time safety guarantee.",  # Describe v16 separation timing.
        "Maximum per-axis horizontal / vertical displacement (m/step)": f"{VMAX_XY} / {VMAX_Z}",
        "Minimum horizontal displacement (m/step)": VMIN_2D,
        "Heading-change / speed-change limits (m/step)": f"{MAX_ACCEL_XY} / {DELTA_SPEED_MAX}",
        "Flight-path-angle tangent limit": GAMMA_MAX_TAN,
        "Big-M": BIG_M,
        "Objective weights": f"time={w_time}, distance={w_dist}, smoothness={w_smooth}, vertical={w_z}, acceleration={w_accel}, descent={w_descent}, final_altitude={w_alt_final}, fuel={w_fuel}",  # Record the added fuel weight.
        "Distance model": f"32-direction PWL norm; maximum relative overestimate={1 / NORM_COSINE - 1:.4%}",  # Record the linear distance model.
        "Fuel model": f"OpenAP local PWL; {FUEL_AIRCRAFT}; fixed mass={fuel_table.mass_kg:g} kg ({FUEL_MASS_FRACTION:g} MTOW); no wind.",  # Record the fuel assumptions.
        "Arrival interval": "Included in time and fuel costs.",  # Record arrival-interval accounting.
        "Solver time limit (s) / target gap": f"{m.Params.TimeLimit} / {m.Params.MIPGap:.2%}",
        "Solver threads / MIPFocus": f"{m.Params.Threads} / {m.Params.MIPFocus}",
        "Solver status / runtime (s)": f"{_STATUS_NAMES.get(m.status, m.status)} / {m.Runtime:.3f}",
        "Objective / best bound / achieved gap": f"{m.ObjVal:.6f} / {m.ObjBound:.6f} / {m.MIPGap:.2%}",
        "Holding": f"Free position/altitude at any airborne grid interval, including entry; up to {args.max_hold_steps} circuits; {TIMESTEP_DT:g} s at {HOLD_SPEED_MPS:g} m/s TAS; level, fixed mass, no bank or wind correction. Altitude fuel PWL at 100-ft knots; fixed racetrack orientation.",
        "Holding separation": "Conservative path-length/2 horizontal envelope at hold-start grid nodes only; not continuous-time or operational separation.",
        "Downstream analysis": "Interior interval-aware OpenAP report; entry-only NLP handoff disabled.",
    }, _flights_display)
    with record_terminal_output(report_path, "Flight Results"):
        print_flight_results(flights, fix_sel, fix_names, star_fixes_xyz, f_z, N_steps,
                              k_arrive, GRID_EPOCH_UTC, TIMESTEP_DT)

    df_wide = save_trajectory_csv_xyz(
        f_x, f_y, f_z, N_flights, N_steps, GRID_EPOCH_UTC, TIMESTEP_DT,
        output_dir, DTW_LAT0, DTW_LON0, flights=flights, k_arrive=k_arrive)
    holding_rows = []
    holding_intervals = []
    for i, flight in flights.iterrows():
        hold_flags = [int(round(hold[i, k].X)) for k in range(N_steps)]
        df_wide[f"f{i+1}_hold"] = hold_flags
        hold_seconds = sum(hold_flags) * TIMESTEP_DT
        for k, selected in enumerate(hold_flags):
            if selected:
                holding_intervals.append({
                    "acId": flight['acId'], "end_step": k, "start_t": (k - 1) * TIMESTEP_DT,
                    "end_t": k * TIMESTEP_DT, "x": f_x[i][k].X, "y": f_y[i][k].X, "z": f_z[i][k].X,
                    "hold_fuel_kg": TIMESTEP_DT * float(fuel_table.exact(HOLD_SPEED_MPS, f_z[i][k].X, 0.0)),
                    "hold_pwl_fuel_kg": hold_step_kg[i, k].X,
                })
        holding_rows.append({
            "acId": flight['acId'], "hold_loops": sum(hold_flags),
            "hold_seconds": hold_seconds, "hold_speed_mps": HOLD_SPEED_MPS,
            "hold_fuel_kg": sum(row['hold_fuel_kg'] for row in holding_intervals if row['acId'] == flight['acId']),
            "hold_pwl_fuel_kg": sum(hold_step_kg[i, k].X for k in range(int(flight['flight_entry_timestep']) + 1, N_steps)),
            "route_pwl_fuel_kg": sum(fuel_step_kg[i, k].X for k in range(int(flight['flight_entry_timestep']) + 1, N_steps)),
        })
    df_wide.to_csv(output_dir / "df_wide.csv", index=False)
    pd.DataFrame(holding_intervals, columns=['acId', 'end_step', 'start_t', 'end_t', 'x', 'y', 'z',
                                            'hold_fuel_kg', 'hold_pwl_fuel_kg']).to_csv(output_dir / "holding_intervals.csv", index=False)
    holding_summary = pd.DataFrame(holding_rows)
    holding_summary.to_csv(output_dir / "holding_summary.csv", index=False)
    print(holding_summary.to_string(index=False))
    with (output_dir / "run_summary.json").open("w") as stream:
        json.dump({"status": _STATUS_NAMES.get(m.Status, str(m.Status)), "runtime_seconds": m.Runtime,
                   "mip_gap": m.MIPGap, "objective": m.ObjVal, "objective_bound": m.ObjBound, "binary_variables": m.NumBinVars,
                   "quadratic_constraints": m.NumQConstrs,
                   "hold_fuel_kg": float(holding_summary.hold_fuel_kg.sum()),
                   "hold_pwl_fuel_kg": float(holding_summary.hold_pwl_fuel_kg.sum()),
                   "route_pwl_fuel_kg": float(holding_summary.route_pwl_fuel_kg.sum())}, stream, indent=2)

    # 3.3. Step-by-step animation of MILP waypoint trajectories
    if args.plots:
        from Functions.fuel_emission_analysis_plot import animate_milp_trajectories
        animate_milp_trajectories(
            df_wide, flights, star_fixes_xyz,
            output_path=output_dir / "plots" / "milp_animation.gif",
            fps=2, dpi=100,
        )

    # 3.4. Analyze and visualize optimized trajectory
    print("\n=== HOLDING AND ROUTE FUEL/EMISSIONS ===")
    aircraft_list = [
        {"acId": flights.iloc[i]['acId'], "acType": "B737"}
        for i in range(N_flights)
    ]
    fuel_report = report_free_holding(df_wide, aircraft_list[0], scenario, output_dir, FUEL_MASS_FRACTION)
    print(fuel_report[["acId", "hold_fuel_kg", "route_fuel_kg", "total_fuel_kg", "total_CO2_g"]].to_string(index=False))
    try:
        if args.run_nlp:
            from Functions.fuel_emission_analysis_main_holding import analyze_optimized_trajectory_xyz
            with record_terminal_output(report_path, "Downstream NLP Analysis"):
                results = analyze_optimized_trajectory_xyz(
                    df_wide, aircraft_list, DTW_LAT0, DTW_LON0, output_dir=output_dir / "nlp",
                    hold_speed_mps=HOLD_SPEED_MPS, mass_fraction=FUEL_MASS_FRACTION,
                    nlp_time_limit=args.nlp_time_limit,
                )
    finally:
        with report_path.open("a", encoding="utf-8") as report:
            report.write("\n## Appendix: MILP Solver Log\n\n```text\n")
            report.write(solver_log)
            report.write("\n```\n")
    print("Holding experiment complete; macro hold anchors are not physical stationary aircraft.")
    print(f"Simulation summary saved to {report_path}")