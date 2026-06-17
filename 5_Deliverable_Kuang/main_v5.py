from gurobipy import *
import numpy as np
import pandas as pd
from pathlib import Path
from Functions.fuel_model import *
from Functions.fuel_emission_analysis_main import analyze_optimized_trajectory
from Functions.utilities import load_flights, load_star_fixes, compute_time_grid
import os
# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

############
# 1. SETUP #
############
print(" === PROBLEM SETUP ===")
## 1.1 Set parameters
# Global parameters
FT2NM             = 1 / 6076.12               # Feet to NM
flights_to_optimize = [
    # "DAL1066_KTPAtoKDTW",   # enters 07:56 (from SE, lon -82.14)
    # "EDV5018_CYULtoKDTW",   # enters 08:12 (from NE, lon -80.48)
    # "DAL2140_KSTLtoKDTW",   # enters 08:09 (from W,  lon -86.35)
    # "DAL1120_KMSNtoKDTW",   # enters 08:10 (from W,  lon -86.16)
    "AAL419_KCLTtoKDTW",   # enters 08:11 (from SE, lon -82.79)
]

# Model parameters
TIMESTEP_DT = 480.0                                  # Time step seconds
N_STEPS_BUDGET = 6                          # Active steps every flight gets after its entry (= number of middle waypoints + 1); tune this to control trajectory smoothness vs. separation flexibility

# Cost parameters
w_time      = 1.0    # Time cost weight (penalizes each active step; drives earlier arrival)
w_smooth = 0.05   # Path smoothness weight
w_accel  = 2.0    # Heading-rate penalty weight
w_z = 0.25   # Relative penalty on vertical changes

# Constraint parameters
BIG_M             = 1e5                    # Disjunction constant

VMAX_LAT  = 0.25/60                         # Max latitude speed
VMAX_LON  = 0.072/60                        # Max longitude speed
VMAX_ALT  = 1000/60                         # Max altitude speed

SEP_HOR_NM = 500.0 * FT2NM                 # Horizontal separation minimum
SEP_VERT_FT = 100.0                        # Vertical separation minimum

# Aircraft parameters
S    = 122.6    # Wing area (m²)
mtow = 70000    # Max takeoff weight (kg)
tsfc = 0.00003  # Thrust specific fuel consumption
cd0  = 0.02     # Zero-lift drag coefficient
print("Parameters loaded...")

script_dir = Path(__file__).parent

## 1.2. Define STAR fixes
star_fixes, lat_vals, lon_vals, alt_vals_max, alt_vals_min = load_star_fixes(script_dir / "Input" / "star_fixes.csv")
print("STAR fixes loaded...")

## 1.3. Load flight data
csv_path = script_dir / "Input" / "entry_exit_points.csv"
flights    = load_flights(csv_path, flights_to_optimize, TIMESTEP_DT)
print("Flight data loaded...")

## 1.4. Determine time steps
N_steps, max_entry_k = compute_time_grid(flights, N_STEPS_BUDGET)
print(f"Time steps = {N_steps}  (max_entry_k={max_entry_k}, budget={N_STEPS_BUDGET} steps/flight)...")
print(f"\nFinal flight data for optimization:\n{flights}\n")

########################
# 2. MILP Optimization #
########################
print(" === MILP OPTIMIZATION ===")
# 2.1. Create model
env = Env(empty=True)
env.setParam("OutputFlag", 0)   # disable all output from Gurobi
env.start()

m = Model("mip1", env=env)
print("Model created...")

# 2.2. Create decision variables
# i) Position and control variables (per flight, per step)
N_flights = len(flights)
f_lat = [m.addVars(range(N_steps), lb=-100000, name=f"f{i}_lat")    for i in range(1, N_flights+1)]
f_lon = [m.addVars(range(N_steps), lb=-100000, name=f"f{i}_lon")    for i in range(1, N_flights+1)]
f_alt = [m.addVars(range(N_steps),             name=f"f{i}_alt_ft") for i in range(1, N_flights+1)]
u_lat = [m.addVars(range(N_steps),             name=f"uf{i}_x")     for i in range(1, N_flights+1)]
u_lon = [m.addVars(range(N_steps),             name=f"uf{i}_y")     for i in range(1, N_flights+1)]
u_alt = [m.addVars(range(N_steps),             name=f"uf{i}_z")     for i in range(1, N_flights+1)]

# ii) STAR fix selection (binary)
fix_names = list(star_fixes)
n_fixes   = len(fix_names)
fix_sel = [m.addVars(n_fixes, vtype=GRB.BINARY, name=f"y{i+1}") for i in range(N_flights)]

# iii) Auxiliary variables
d_lat, d_lon, d_alt            = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("dx",       "dy",      "dz"      )]  # Δlat, Δlon, Δalt per step
dd_lat,   dd_lon               = [m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=n) for n in ("ddx",      "ddy"                 )]  # Heading-rate proxies Δ(Δlat), Δ(Δlon)
u_dd_lat,   u_dd_lon           = [m.addVars(N_flights, N_steps, lb=0.0,           vtype=GRB.CONTINUOUS, name=n) for n in ("udx",      "udy"                 )]  # |ddx|, |ddy|
accel_cost, speed_2d           = [m.addVars(N_flights, N_steps, lb=0.0,           vtype=GRB.CONTINUOUS, name=n) for n in ("accel_xy", "speed"               )]  # Gated heading-rate cost, 2-D speed
fix_reached, sep_bypass, fix_enters = [m.addVars(N_flights, N_steps, lb=0.0,           vtype=GRB.BINARY,     name=n) for n in ("is_end",   "landed",  "delta_end")]  # Arrival flag, separation bypass, 0→1 transition
k_arrive                       =  m.addVars(N_flights,          lb=0,             vtype=GRB.INTEGER,    name="exit_k", ub=N_steps-1)                             # Step of first arrival (per flight)
print("Decision variables created...")

# 2.3. Define Objective Function
obj = LinExpr()

for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    
    for k in range(k_entry + 1, N_steps):
        active = 1 - fix_reached[i, k]          # 1 while airborne, 0 after arrival
        obj += w_time * TIMESTEP_DT * active            # penalize each active step → minimizes flight time

        # Step differences and their L1 absolutes
        for diff, u, cur, prv in zip(
            (d_lat[i,k],      d_lon[i,k],      d_alt[i,k]   ),
            (u_lat[i][k-1],   u_lon[i][k-1],   u_alt[i][k-1]),
            (f_lat[i][k],     f_lon[i][k],     f_alt[i][k]  ),
            (f_lat[i][k-1],   f_lon[i][k-1],   f_alt[i][k-1]),
        ):
            m.addConstr(diff == cur - prv)
            m.addConstr(u    == abs_(diff))

        m.addConstr(speed_2d[i,k]**2 == d_lat[i,k]**2 + d_lon[i,k]**2)
        obj += w_smooth * (u_lat[i][k-1] + u_lon[i][k-1] + w_z * u_alt[i][k-1]) # Path smoothness (L1 on displacements)

        # Heading-rate penalty (k_entry+2 onward; k_arrive enforced via arrived gate)
        if k >= k_entry + 2:
            for dd, ud, df, dfp in zip(
                (dd_lat[i,k],     dd_lon[i,k]    ),
                (u_dd_lat[i,k],     u_dd_lon[i,k]    ),
                (d_lat[i,k],   d_lon[i,k]  ),
                (d_lat[i,k-1], d_lon[i,k-1]),
            ):
                m.addConstr(dd == df - dfp)
                m.addConstr(ud == abs_(dd))
            m.addGenConstrIndicator(fix_reached[i,k], 0, accel_cost[i,k] == u_dd_lat[i,k] + u_dd_lon[i,k])
            m.addGenConstrIndicator(fix_reached[i,k], 1, accel_cost[i,k] == u_dd_lat[i,k] + u_dd_lon[i,k])
            obj += w_accel * accel_cost[i,k] # Heading-rate penalty, gated by fix_reached (active only while airborne)

m.setObjective(obj, GRB.MINIMIZE)
print("Objective function created...")


# 2.4. Define constraints
# i) Entry point constraints (hold each flight at its entry position until k_entry)
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']
    for k in range(k_entry + 1):
        m.addConstr(f_lat[i][k] == flights.iloc[i]['entry_lat'], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(f_lon[i][k] == flights.iloc[i]['entry_lon'], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(f_alt[i][k] == flights.iloc[i]['entry_alt'], f"c_pre_entry_z_{i}_t{k}")
    print(f"Entry point constraints created, k_entry={k_entry}...")

# ii) STAR fix (exit point) constraints (pin final position to one chosen fix)
for j in range(N_flights):
    m.addConstr(quicksum(fix_sel[j][i] for i in range(n_fixes)) == 1, f"one_fix{j+1}")
    m.addConstr(f_lat[j][N_steps-1] == LinExpr(lat_vals,     fix_sel[j].values()), f"lat_choice{j+1}")
    m.addConstr(f_lon[j][N_steps-1] == LinExpr(lon_vals,     fix_sel[j].values()), f"lon_choice{j+1}")
    m.addConstr(f_alt[j][N_steps-1] <= LinExpr(alt_vals_max, fix_sel[j].values()), f"alt_choice_max{j+1}")
    m.addConstr(f_alt[j][N_steps-1] >= LinExpr(alt_vals_min, fix_sel[j].values()), f"alt_choice_min{j+1}")
print("STAR fix constraints created...")

# iii) Arrival flag (fix_reached / fix_enters) logic
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']

    # Before entry: fix_reached and fix_enters must be 0
    for k in range(k_entry + 1):
        m.addConstr(fix_reached[i, k] == 0, f"fix_reached_pre_entry_{i}_{k}")
        m.addConstr(fix_enters[i, k] == 0, f"fix_enters_pre_entry_{i}_{k}")

    # After entry: fix_reached is monotone (0→1 only); once 1, position freezes at the chosen STAR fix
    for k in range(k_entry + 2, N_steps):
        m.addConstr(fix_reached[i, k] >= fix_reached[i, k-1], f"fix_reached_monotone_{i}_{k}")
        m.addConstr((fix_reached[i, k] == 1) >> (f_lat[i][k] == f_lat[i][N_steps-1]))
        m.addConstr((fix_reached[i, k] == 1) >> (f_lon[i][k] == f_lon[i][N_steps-1]))
        m.addConstr((fix_reached[i, k] == 1) >> (f_alt[i][k] == f_alt[i][N_steps-1]))

    # fix_enters encodes the 0→1 transition; link k_arrive to the transition step
    for k in range(k_entry + 1, N_steps):
        m.addConstr(fix_enters[i, k] >= fix_reached[i, k] - fix_reached[i, k-1], f"fix_enters_lb_{i}_{k}")
        m.addConstr(fix_enters[i, k] <= fix_reached[i, k],                        f"fix_enters_ub_{i}_{k}")
        m.addConstr(fix_enters[i, k] <= 1 - fix_reached[i, k-1],                  f"fix_enters_ub_prev_{i}_{k}")

    delta_sum = quicksum(fix_enters[i, k] for k in range(k_entry + 1, N_steps))
    m.addConstr(
        k_arrive[i] == quicksum(k * fix_enters[i, k] for k in range(k_entry + 1, N_steps)) + (N_steps - 1) * (1 - delta_sum),
        f"k_arrive_def_{i}"
    )

# Every flight must have arrived by the final step (guarantees k_arrive is always defined)
for i in range(N_flights):
    m.addConstr(fix_reached[i, N_steps-1] == 1, f"fix_reached_at_final_{i}")

# Step budget: every flight must arrive within N_STEPS_BUDGET steps of its own entry
for i in range(N_flights):
    k_entry = int(flights.iloc[i]['flight_entry_timestep'])
    m.addConstr(k_arrive[i] <= k_entry + N_STEPS_BUDGET, f"step_budget_{i}")
print("Arrival flag constraints created...")
print(f"  Step budget: k_arrive[i] <= k_entry + {N_STEPS_BUDGET} for all flights")

# iv) Max speed constraints (bound per-step displacement in each axis)
for i in range(N_flights):
    k_entry = flights.iloc[i]['flight_entry_timestep']
    for k in range(k_entry + 1, N_steps):
        m.addConstr(f_lat[i][k] - f_lat[i][k-1] <=  VMAX_LAT*TIMESTEP_DT)
        m.addConstr(f_lon[i][k] - f_lon[i][k-1] <=  VMAX_LON*TIMESTEP_DT)
        m.addConstr(f_alt[i][k] - f_alt[i][k-1] <=  VMAX_ALT*TIMESTEP_DT)

        m.addConstr(f_lat[i][k-1] - f_lat[i][k] <=  VMAX_LAT*TIMESTEP_DT)
        m.addConstr(f_lon[i][k-1] - f_lon[i][k] <=  VMAX_LON*TIMESTEP_DT)
        m.addConstr(f_alt[i][k-1] - f_alt[i][k] <=  VMAX_ALT*TIMESTEP_DT)
print("Max speed constraints created...")

# v) Separation constraints
# sep_bypass is a free binary — the solver can set it to 1 to waive separation when flights
# are co-located (e.g. two flights choosing the same STAR fix), keeping the MILP feasible.
for k in range(N_steps):
    for i in range(N_flights - 1):
        for j in range(i + 1, N_flights):
            if k >= flights.iloc[i]['flight_entry_timestep'] and k >= flights.iloc[j]['flight_entry_timestep']:
                bin_vars = m.addVars(range(6), name=f'bin_{i}_{j}_{k}', vtype=GRB.BINARY)

                m.addConstr(f_lat[i][k] - f_lat[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[0]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_lon[i][k] - f_lon[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[1]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_alt[i][k] - f_alt[j][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_lat[j][k] - f_lat[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[3]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_lon[j][k] - f_lon[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[4]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])
                m.addConstr(f_alt[j][k] - f_alt[i][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[5]) - BIG_M*sep_bypass[i, k] - BIG_M*sep_bypass[j, k])

                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5] >= 1)
print("Separation constraints created...")

# 2.5. Initiate optimization
print("Starting optimization...")
m.optimize()
print("Optimization completed.")
print()
print()


# ###########
# 3. OUTPUT #
# ###########
print(" === OUTPUT RESULTS ===")
if m.status == GRB.OPTIMAL: # Only extract results if Gurobi found a valid optimal solution.
    # 3.1. Print objective value and chosen STAR fixes
    print('Optimization success! Obj= %g' % m.ObjVal)
    print('\nChosen STAR fixes for each flight:')
    for i in range(N_flights):
        flight_id = flights.iloc[i]['acId']
        for k in range(n_fixes):
            if fix_sel[i][k].X > 0.5:  # Binary variable is 1 (with tolerance for numerical issues)
                chosen_fix = fix_names[k]
                fix_lat, fix_lon, (fix_alt_max, fix_alt_min) = star_fixes[chosen_fix]
                final_alt = f_alt[i][N_steps-1].X
                print(f'  {flight_id}: {chosen_fix} (lat={fix_lat:.4f}, lon={fix_lon:.4f}, alt={final_alt:.0f} ft)')
                break
    print()

    print('Optimized exit_k by flight:')
    for i in range(N_flights):
        flight_id = flights.iloc[i]['acId']
        k_exit = int(round(k_arrive[i].X))
        print(f'  {flight_id}: exit_k={k_exit}, exit_t={k_exit * TIMESTEP_DT:.1f} s')
    print()

    # 3.2. Extract and save optimized trajectories
    rows = []
    for k in range(N_steps):
        row = {"t": k * TIMESTEP_DT}
        for i in range(N_flights):
            row[f"f{i+1}_lat"]    = f_lat[i][k].X
            row[f"f{i+1}_lon"]    = f_lon[i][k].X
            row[f"f{i+1}_alt_ft"] = f_alt[i][k].X
        rows.append(row)
    df_wide = pd.DataFrame(rows)

    output_dir = script_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weathertrialstatic.csv"
    df_wide.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print()

    # 3.3. Print waypoint table per flight
    for i in range(N_flights):
        flight_id = flights.iloc[i]['acId']
        k_entry   = int(flights.iloc[i]['flight_entry_timestep'])
        print(f"  Waypoints for {flight_id}:")
        print(f"  {'Step':>4}  {'Time (s)':>9}  {'Lat':>10}  {'Lon':>11}  {'Alt (ft)':>10}  {'fix_reached':>11}  {'sep_bypass':>10}")
        print(f"  {'-'*4}  {'-'*9}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}")
        for k in range(N_steps):
            t_sec      = k * TIMESTEP_DT
            lat_val    = f_lat[i][k].X
            lon_val    = f_lon[i][k].X
            alt_val    = f_alt[i][k].X
            end_val    = int(round(fix_reached[i, k].X)) if k >= k_entry else 0
            land_val   = int(round(sep_bypass[i, k].X))
            marker     = " <-- entry" if k == k_entry else (" <-- ARRIVED" if end_val == 1 and (k == 0 or int(round(fix_reached[i, k-1].X)) == 0) else "")
            print(f"  {k:>4}  {t_sec:>9.1f}  {lat_val:>10.4f}  {lon_val:>11.4f}  {alt_val:>10.1f}  {end_val:>11}  {land_val:>10}{marker}")
        print()

    # Diagnostic: show pairwise separation at each step (skip steps where either flight has landed)
    print("  --- Pairwise separation check (active flights only) ---")
    flight_ids = [flights.iloc[i]['acId'] for i in range(N_flights)]
    print(f"  {'Step':>4}  {'Pair':<40}  {'dLat':>8}  {'dLon':>8}  {'dAlt':>8}  {'Sep OK?'}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*16}")
    for k in range(N_steps):
        for i in range(N_flights - 1):
            for j in range(i + 1, N_flights):
                # Skip if either flight has landed (bypass mechanism active)
                if int(round(sep_bypass[i, k].X)) == 1 or int(round(sep_bypass[j, k].X)) == 1:
                    continue
                dlat = abs(f_lat[i][k].X - f_lat[j][k].X)
                dlon = abs(f_lon[i][k].X - f_lon[j][k].X)
                dalt = abs(f_alt[i][k].X - f_alt[j][k].X)
                sep_ok = (dlat >= SEP_HOR_NM or dlon >= SEP_HOR_NM or dalt >= SEP_VERT_FT)
                pair = f"{flight_ids[i]} vs {flight_ids[j]}"
                flag = "OK" if sep_ok else "*** VIOLATION ***"
                print(f"  {k:>4}  {pair:<40}  {dlat:>8.4f}  {dlon:>8.4f}  {dalt:>8.1f}  {flag}")
    print()

    # 3.4. Save decision variables and cost breakdown to CSV
    # Pre-compute chosen fix name per flight
    chosen_fix_per_flight = []
    for i in range(N_flights):
        cf = "?"
        for kf in range(n_fixes):
            if fix_sel[i][kf].X > 0.5:
                cf = fix_names[kf]
                break
        chosen_fix_per_flight.append(cf)

    var_rows = []
    for i in range(N_flights):
        flight_id  = flights.iloc[i]['acId']
        entry_k_i  = int(flights.iloc[i]['flight_entry_timestep'])
        exit_k_val = int(round(k_arrive[i].X))
        chosen_fix = chosen_fix_per_flight[i]

        for k in range(N_steps):
            t_sec = k * TIMESTEP_DT
            active_step  = k >= entry_k_i + 1   # displacement variables defined
            accel_step   = k >= entry_k_i + 2   # heading-rate variables defined

            # Cost terms
            smooth = w_smooth * (u_lat[i][k-1].X + u_lon[i][k-1].X + w_z * u_alt[i][k-1].X) if active_step else 0.0
            accel  = w_accel  * accel_cost[i, k].X                                          if accel_step  else 0.0

            var_rows.append({
                # --- Identity ---
                "flight":       flight_id,
                "step":         k,
                "time_s":       t_sec,
                "entry_k":      entry_k_i,
                "exit_k":       exit_k_val,
                "chosen_fix":   chosen_fix,
                # --- State flags ---
                "fix_reached":  int(round(fix_reached[i, k].X)) if k >= entry_k_i else 0,
                "fix_enters":   int(round(fix_enters[i, k].X))  if k >= entry_k_i else 0,
                "sep_bypass":   int(round(sep_bypass[i, k].X)),
                # --- Position ---
                "lat":          f_lat[i][k].X,
                "lon":          f_lon[i][k].X,
                "alt_ft":       f_alt[i][k].X,
                # --- Displacements (step k vs k-1) ---
                "dx":           d_lat[i, k].X  if active_step else 0.0,
                "dy":           d_lon[i, k].X  if active_step else 0.0,
                "dz":           d_alt[i, k].X  if active_step else 0.0,
                # --- Absolute displacements (L1 terms) ---
                "ux":           u_lat[i][k-1].X   if active_step else 0.0,
                "uy":           u_lon[i][k-1].X   if active_step else 0.0,
                "uz":           u_alt[i][k-1].X   if active_step else 0.0,
                # --- Speed (2-D, degrees/step) ---
                "speed_2d":     speed_2d[i, k].X  if active_step else 0.0,
                # --- Heading-rate (direction change, step k vs k-1) ---
                "ddx":          dd_lat[i, k].X    if accel_step else 0.0,
                "ddy":          dd_lon[i, k].X    if accel_step else 0.0,
                "udx":          u_dd_lat[i, k].X    if accel_step else 0.0,
                "udy":          u_dd_lon[i, k].X    if accel_step else 0.0,
                "accel_xy":     accel_cost[i, k].X if accel_step else 0.0,
                # --- Cost terms ---
                "cost_smooth":  smooth,
                "cost_accel":   accel,
                "cost_total":   smooth + accel,
            })

    df_vars = pd.DataFrame(var_rows)
    vars_path = output_dir / "decision_variables.csv"
    df_vars.to_csv(vars_path, index=False)
    print(f"Decision variables saved to {vars_path}")
    print()

    # 3.5. Analyze and visualize optimized trajectory
    print(" === ANALYZING OPTIMIZED TRAJECTORY ===")
    aircraft_list = [
        {"acId": flights.iloc[i]['acId'], "acType": "B737"}
        for i in range(N_flights)
    ]
    results = analyze_optimized_trajectory(df_wide, aircraft_list)
    print("Analysis and visualization complete!")
else:
    print("Optimization was not successful. Status code:", m.status)
    if m.status == GRB.INFEASIBLE:
        print("Computing IIS (Irreducible Infeasible Subsystem) to identify conflicting constraints...")
        m.computeIIS()
        print("Infeasible constraints (IIS):")
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"  CONSTR: {c.ConstrName}")
        print("Infeasible variable bounds (IIS):")
        for v in m.getVars():
            if v.IISLB or v.IISUB:
                print(f"  VAR: {v.VarName}  (lb_infeas={v.IISLB}, ub_infeas={v.IISUB})")
    df_wide = None