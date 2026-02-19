from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
from Functions.fuel_model import *
from Functions.fuel_emission_analysis_main import analyze_optimized_trajectory
import math
import os

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

############
# 1. SETUP #
############
print(" === PROBLEM SETUP ===")

## 1.1 Set parameters
# Global parameters
FT2NM               = 1 / 6076.12               # Feet to NM
flights_to_optimize = ["DAL1208_KORDtoKDTW"]    # Define flights to optimize

# Model parameters
DT = 60.0                                        # Time step seconds

# Cost parameters
CT = 1.0                                         # time cost weight
CF = 1.5                                         # fuel cost weight
CSMOOTH = 0.05                                   # NEW: smoothness weight (minimal change, tune if needed)
ALPHA_Z = 0.25                                   # NEW: relative penalty on vertical changes

# Constraint parameters
BIG_M = 1e5                                      # Disjunction constant

V_MAX_X = 0.25 / 60                              # Max latitude speed (deg/sec?) / 60 as in your original
V_MAX_Y = 0.072 / 60                             # Max longitude speed
V_MAX_Z = 1000 / 60                              # Max altitude speed (ft/sec)

SEP_HOR_NM  = 500.0 * FT2NM                      # Horizontal separation minimum
SEP_VERT_FT = 100.0                              # Vertical separation minimum

WEATHER_EPS_DEG  = 1e-4                          # Weather buffer tolerance
WEATHER_STEP_SEC = 300                           # Weather frame interval in seconds (5 minutes)

# Example aircraft parameters (used by compute_fuel_emission_flow)
S    = 122.6                                     # Wing area m^2
mtow = 70000                                     # Max takeoff weight kg
tsfc = 0.00003                                   # Thrust specific fuel consumption
cd0  = 0.02                                       # Zero-lift drag coefficient
k    = 0.045                                      # Induced drag factor

print("Parameters loaded...")

## 1.2. Define STAR fixes
# NOTE: your tuples look like (alt_max, alt_min) even though comment says (alt_min, alt_max).
# Your later extraction matches (max,min) ordering, so we keep it unchanged.
star_fixes = {
    "BONZZ": (41.7483, -82.7972, (21000, 15000)), "CRAKN": (41.6730, -82.9405, (26000, 12000)), "CUUGR": (42.3643, -83.0975, (11000, 10000)),
    "FERRL": (42.4165, -82.6093, (10000, 8000)),  "GRAYT": (42.9150, -83.6020, (22000, 17000)), "HANBL": (41.7375, -84.1773, (21000, 17000)),
    "HAYLL": (41.9662, -84.2975, (11000, 11000)), "HTROD": (42.0278, -83.3442, (12000, 12000)), "KKISS": (42.5443, -83.7620, (15000, 12000)),
    "KLYNK": (41.8793, -82.9888, (10000, 9000)),  "LAYKS": (42.8532, -83.5498, (10000, 10000)), "LECTR": (41.9183, -84.0217, (10000, 8000)),
    "RKCTY": (42.6869, -83.9603, (13000, 11000)), "VCTRZ": (41.9878, -84.0670, (15000, 12000))  # name: (lat, lon, (alt_max_ft, alt_min_ft))
}
print("STAR fixes loaded...")

## 1.3. Load flight data
script_dir = Path(__file__).parent
csv_path   = script_dir / "Input" / "entry_exit_points.csv"

all_flights_df = pd.read_csv(csv_path)
flights_df = all_flights_df[all_flights_df['acId'].isin(flights_to_optimize)].reset_index(drop=True)

flights_df['entry_rectime'] = pd.to_datetime(flights_df['entry_rectime'])
flights_df['exit_rectime']  = pd.to_datetime(flights_df['exit_rectime'])

min_time = flights_df['entry_rectime'].min()
max_time = flights_df['exit_rectime'].max()

flights_df['rel_entry_time']   = (flights_df['entry_rectime'] - min_time).dt.total_seconds()
flights_df['rel_landing_time'] = (flights_df['exit_rectime']  - min_time).dt.total_seconds()

required_columns = [
    'acId',
    'entry_lat', 'entry_lon', 'entry_alt',
    'rel_entry_time',
    'exit_lat', 'exit_lon', 'exit_alt',
    'rel_landing_time'
]
flights = flights_df[required_columns].copy()
print("Flight data loaded...")

## 1.4. Load weather data
fast_milp_dir = script_dir / "Input"
weather_files = [
    str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t00min.csv")
]

weather_dfs = []
for fp in weather_files:
    dfw = pd.read_csv(fp)
    dfw = dfw[['min_lat', 'max_lat', 'min_lon', 'max_lon']].dropna().reset_index(drop=True)
    weather_dfs.append(dfw)
print("Weather data loaded...")

## 1.5. Determine time steps
max_time = flights['rel_landing_time'].max()
t0 = 0
tN = max_time if max_time > 2100 else 2100

flights['flight_entry_timestep'] = (flights['rel_entry_time'] / DT).astype(int)
N_steps = int((tN - t0) / DT) + 1

print("Time steps loaded...")
print()
print("Final flight data for optimization:")
print(flights)
print()
print()

########################
# 2. MILP Optimization #
########################
print(" === MILP OPTIMIZATION ===")

# 2.1. Create model
env = Env(empty=True)
env.setParam("OutputFlag", 0)
env.start()

m = Model("mip1", env=env)
print("Model created...")

# 2.2. Create decision variables
N_flights = len(flights)

x, y, z = [], [], []
ux, uy, uz = [], [], []

for i in range(1, N_flights + 1):
    x.append(m.addVars(range(N_steps), name=f"f{i}_lat", lb=-100000))
    y.append(m.addVars(range(N_steps), name=f"f{i}_lon", lb=-100000))
    z.append(m.addVars(range(N_steps), name=f"f{i}_alt_ft"))
    ux.append(m.addVars(range(N_steps), name=f"uf{i}_x"))
    uy.append(m.addVars(range(N_steps), name=f"uf{i}_y"))
    uz.append(m.addVars(range(N_steps), name=f"uf{i}_z"))

# STAR fix selection binaries
fix_names    = list(star_fixes)
lat_vals     = [star_fixes[k][0] for k in fix_names]
lon_vals     = [star_fixes[k][1] for k in fix_names]
alt_vals_min = [star_fixes[k][2][1] for k in fix_names]  # (max,min) -> min is index 1
alt_vals_max = [star_fixes[k][2][0] for k in fix_names]  # (max,min) -> max is index 0
k_fixes      = len(fix_names)

b = []
for i in range(N_flights):
    b.append(m.addVars(k_fixes, vtype=GRB.BINARY, name=f"y{i+1}"))

# Cost/aux variables
diffx  = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dx")
diffy  = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dy")
diffz  = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dz")
is_end = m.addVars(N_flights, N_steps, vtype=GRB.BINARY, name="is_end")
speed  = m.addVars(N_flights, N_steps, name="speed")
tfuel  = m.addVars(N_flights, N_steps, name="t")  # (your original name)

print("Decision variables created...")

# 2.3. Define Objective Function (FIXED: build first, set once at end)
obj = LinExpr()

for i in range(N_flights):
    entry_k = int(flights.iloc[i]['flight_entry_timestep'])

    for k in range(entry_k + 1, N_steps):
        # Define differences
        m.addConstr(diffx[i, k] == x[i][k] - x[i][k-1])
        m.addConstr(diffy[i, k] == y[i][k] - y[i][k-1])
        m.addConstr(diffz[i, k] == z[i][k] - z[i][k-1])

        # Absolute steps (your existing "controls")
        m.addConstr(ux[i][k-1] == abs_(diffx[i, k]))
        m.addConstr(uy[i][k-1] == abs_(diffy[i, k]))
        m.addConstr(uz[i][k-1] == abs_(diffz[i, k]))

        # Speed definition (kept minimal change: quadratic equality)
        m.addConstr(speed[i, k] * speed[i, k] == diffx[i, k] * diffx[i, k] + diffy[i, k] * diffy[i, k])

        # Fuel: active until end, zero after end
        m.addGenConstrIndicator(
            is_end[i, k], 0,
            tfuel[i, k] == compute_fuel_emission_flow(
                speed[i, k], z[i][k], diffz[i, k],
                0.8 * mtow, S, cd0, k, tsfc,
                m, limit=True, cal_emission=False, mode="full"
            )
        )
        m.addGenConstrIndicator(is_end[i, k], 1, tfuel[i, k] == 0)

        # NEW (minimal but important): better-scaled time + smoothness
        active = 1 - is_end[i, k]
        obj += CF * tfuel[i, k]
        obj += CT * DT * active
        obj += CSMOOTH * (ux[i][k-1] + uy[i][k-1] + ALPHA_Z * uz[i][k-1])

# Set objective ONCE (this was a real bug/issue in your original structure)
m.setObjective(obj, GRB.MINIMIZE)
print("Objective function created...")

# 2.4. Define constraints

# i) is_end logic constraints
for i in range(N_flights):
    entry_k = int(flights.iloc[i]['flight_entry_timestep'])

    # Before entry: is_end must be 0
    for k in range(entry_k + 1):
        m.addConstr(is_end[i, k] == 0, f"is_end_before_entry_{i}_{k}")

    # After entry: monotone and "freeze" position when ended
    for k in range(entry_k + 2, N_steps):
        m.addConstr((is_end[i, k] == 1) >> (x[i][k] == x[i][N_steps-1]))
        m.addConstr((is_end[i, k] == 1) >> (y[i][k] == y[i][N_steps-1]))
        m.addConstr((is_end[i, k] == 1) >> (z[i][k] == z[i][N_steps-1]))
        m.addConstr(is_end[i, k] >= is_end[i, k-1], f"is_end_monotonic_{i}_{k}")

print("is_end logic constraints created...")

# ii) Entry point constraints
for i in range(N_flights):
    entry_k = int(flights.iloc[i]['flight_entry_timestep'])
    for k in range(entry_k + 1):
        m.addConstr(x[i][k] == flights.iloc[i]['entry_lat'], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(y[i][k] == flights.iloc[i]['entry_lon'], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(z[i][k] == flights.iloc[i]['entry_alt'], f"c_pre_entry_z_{i}_t{k}")

print("Entry point constraints created...")

# iii) STAR fix (exit point) constraints
for j in range(N_flights):
    m.addConstr(quicksum(b[j][i] for i in range(k_fixes)) == 1, f"one_fix{j+1}")
    m.addConstr(x[j][N_steps-1] == LinExpr(lat_vals, b[j].values()), f"lat_choice{j+1}")
    m.addConstr(y[j][N_steps-1] == LinExpr(lon_vals, b[j].values()), f"lon_choice{j+1}")
    m.addConstr(z[j][N_steps-1] <= LinExpr(alt_vals_max, b[j].values()), f"alt_choice_max{j+1}")
    m.addConstr(z[j][N_steps-1] >= LinExpr(alt_vals_min, b[j].values()), f"alt_choice_min{j+1}")

print("STAR fix constraints created...")

# iv) Max step constraints after entry
for i in range(N_flights):
    entry_k = int(flights.iloc[i]['flight_entry_timestep'])
    for k in range(entry_k + 1, N_steps):
        m.addConstr(x[i][k] - x[i][k-1] <= V_MAX_X * DT)
        m.addConstr(y[i][k] - y[i][k-1] <= V_MAX_Y * DT)
        m.addConstr(z[i][k] - z[i][k-1] <= V_MAX_Z * DT)

        m.addConstr(x[i][k-1] - x[i][k] <= V_MAX_X * DT)
        m.addConstr(y[i][k-1] - y[i][k] <= V_MAX_Y * DT)
        m.addConstr(z[i][k-1] - z[i][k] <= V_MAX_Z * DT)

print("Max speed constraints created...")

# v) Weather constraints
if len(weather_dfs) > 0:
    last_idx = len(weather_dfs) - 1

    for i in range(N_flights):
        entry_k = int(flights.iloc[i]['flight_entry_timestep'])

        for k in range(entry_k, N_steps):
            t_sec = k * DT
            frame_idx = int(t_sec // WEATHER_STEP_SEC)
            if frame_idx > last_idx:
                frame_idx = last_idx

            dfw = weather_dfs[frame_idx]
            if dfw.empty:
                continue

            for r, row in dfw.iterrows():
                out = m.addVars(4, vtype=GRB.BINARY, name=f"w_out_{i}_{k}_{frame_idx}_{r}")
                m.addConstr(out.sum() >= 1, name=f"w_outside_{i}_{k}_{frame_idx}_{r}")

                m.addConstr(
                    x[i][k] <= row['min_lat'] - WEATHER_EPS_DEG + BIG_M * (1 - out[0]),
                    name=f"w_left_{i}_{k}_{frame_idx}_{r}"
                )
                m.addConstr(
                    x[i][k] >= row['max_lat'] + WEATHER_EPS_DEG - BIG_M * (1 - out[1]),
                    name=f"w_right_{i}_{k}_{frame_idx}_{r}"
                )
                m.addConstr(
                    y[i][k] <= row['min_lon'] - WEATHER_EPS_DEG + BIG_M * (1 - out[2]),
                    name=f"w_below_{i}_{k}_{frame_idx}_{r}"
                )
                m.addConstr(
                    y[i][k] >= row['max_lon'] + WEATHER_EPS_DEG - BIG_M * (1 - out[3]),
                    name=f"w_above_{i}_{k}_{frame_idx}_{r}"
                )

print("Weather constraints created...")

# vi) Separation constraints
for k in range(N_steps):
    for i in range(N_flights - 1):
        for j in range(i + 1, N_flights):
            if k >= int(flights.iloc[i]['flight_entry_timestep']) and k >= int(flights.iloc[j]['flight_entry_timestep']):
                bin_vars = m.addVars(range(6), name=f'bin_{i}_{j}_{k}', vtype=GRB.BINARY)

                m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM  - BIG_M * (1 - bin_vars[0]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])
                m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM  - BIG_M * (1 - bin_vars[1]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])
                m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M * (1 - bin_vars[2]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])

                m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM  - BIG_M * (1 - bin_vars[3]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])
                m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM  - BIG_M * (1 - bin_vars[4]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])
                m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M * (1 - bin_vars[5]) - BIG_M * is_end[i][k] - BIG_M * is_end[j][k])

                m.addConstr(bin_vars.sum() >= 1)

print("Separation constraints created...")

# 2.6. Optimize
print("Starting optimization...")
m.optimize()
print("Optimization completed.")
print()
print()

#############
# 3. OUTPUT #
#############
print(" === OUTPUT RESULTS ===")

if m.status == GRB.OPTIMAL:
    print(f"Optimization success! Obj= {m.ObjVal:g}")

    # Extract trajectories
    rows = []
    for k in range(N_steps):
        row = {"t": k * DT}
        for i in range(N_flights):
            row[f"f{i+1}_lat"]    = x[i][k].X
            row[f"f{i+1}_lon"]    = y[i][k].X
            row[f"f{i+1}_alt_ft"] = z[i][k].X
        rows.append(row)

    df_wide = pd.DataFrame(rows)

    # Save CSV
    output_dir = script_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weathertrialstatic.csv"
    df_wide.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print()

    # Analyze
    print(" === ANALYZING OPTIMIZED TRAJECTORY ===")
    aircraft_list = []
    for idx in range(N_flights):
        acId = flights.iloc[idx]['acId']
        acType = "B737"  # TODO: connect to your data if available
        aircraft_list.append({"acId": acId, "acType": acType})

    results = analyze_optimized_trajectory(df_wide, aircraft_list)
    print("Analysis and visualization complete!")

else:
    print("Optimization was not successful. Status code:", m.status)
    df_wide = None
