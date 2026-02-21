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
FT2NM             = 1 / 6076.12               # Feet to NM
flights_to_optimize = ["DAL1208_KORDtoKDTW"]  # Define flights to optimize
# flights_to_optimize = [
#     "DAL1208_KORDtoKDTW",
#     "DAL1066_KTPAtoKDTW",
#     "EDV5018_CYULtoKDTW",
#     "AAL1456_KORDtoKDTW"
# ]

# Model parameters
DT = 480.0                                  # Time step seconds

# Cost parameters
CT = 1                                     # Time cost weight
CF = 1.5                                   # Fuel cost weight
CSMOOTH = 0.05                                   # NEW: smoothness weight (minimal change, tune if needed)
ALPHA_Z = 0.25                                   # NEW: relative penalty on vertical changes

# Constraint parameters
BIG_M             = 1e5                    # Disjunction constant

V_MAX_X  = 0.25/60                         # Max latitude speed
V_MAX_Y  = 0.072/60                        # Max longitude speed
V_MAX_Z  = 1000/60                         # Max altitude speed

SEP_HOR_NM = 500.0 * FT2NM                 # Horizontal separation minimum
SEP_VERT_FT = 100.0                        # Vertical separation minimum

#GLIDE_SLOPE = 100                          # Max glide slope (descent/horizontal) ≈ 5.7 degrees

WEATHER_EPS_DEG = 1e-4                     # Weather buffer tolerance
WEATHER_STEP_SEC = 300                     # Weather frame interval in seconds (5 minutes) 

# Example aircraft parameters
S = 122.6                                  # Wing area m^2
mtow = 70000                               # Max takeoff weight kg
tsfc = 0.00003                             # Thrust specific fuel consumption
cd0 = 0.02                                 # Zero-lift drag coefficient
k = 0.045                                  # Induced drag factor
print("Parameters loaded...")

## 1.2. Define STAR fixes
star_fixes ={
        "BONZZ": (41.7483, -82.7972, (21000, 15000)), "CRAKN": (41.6730, -82.9405, (26000, 12000)), "CUUGR": (42.3643, -83.0975, (11000, 10000)),
        "FERRL": (42.4165, -82.6093, (10000, 8000)), "GRAYT": (42.9150, -83.6020, (22000, 17000)), "HANBL": (41.7375, -84.1773, (21000, 17000)),
        "HAYLL": (41.9662, -84.2975, (11000, 11000)), "HTROD": (42.0278, -83.3442, (12000, 12000)), "KKISS": (42.5443, -83.7620, (15000, 12000)),
        "KLYNK": (41.8793, -82.9888, (10000, 9000)), "LAYKS": (42.8532, -83.5498, (10000, 10000)), "LECTR": (41.9183, -84.0217, (10000, 8000)),
        "RKCTY": (42.6869, -83.9603, (13000, 11000)), "VCTRZ": (41.9878, -84.0670, (15000, 12000)) # name: (lat, lon, (alt_min_ft, alt_max_ft))
}
print("STAR fixes loaded...")

## 1.3. Load flight data
# Load and filter flights
script_dir = Path(__file__).parent # gets the directory where the current script is located
csv_path = script_dir / "Input" / "entry_exit_points.csv"
all_flights_df = pd.read_csv(csv_path)
flights_df = all_flights_df[all_flights_df['acId'].isin(flights_to_optimize)].reset_index(drop=True)

# Find relative time for all flight histories in seconds from the earliest entry time
flights_df['entry_rectime'] = pd.to_datetime(flights_df['entry_rectime'])
flights_df['exit_rectime'] = pd.to_datetime(flights_df['exit_rectime'])
min_time = flights_df['entry_rectime'].min()
max_time = flights_df['exit_rectime'].max() # Set the beginning of the time grid to the earliest entry time across all flights, and the end to the latest exit time.

flights_df['rel_entry_time'] = (flights_df['entry_rectime'] - min_time).dt.total_seconds()
flights_df['rel_landing_time'] = (flights_df['exit_rectime'] - min_time).dt.total_seconds() # compute the [relative landing time] in seconds relative to the [earliest entry time] (in seconds)

# Keep flights in DataFrame format for easier access
required_columns = ['acId',
    'entry_lat', 'entry_lon', 'entry_alt',
    'rel_entry_time',
    'exit_lat', 'exit_lon', 'exit_alt',
    'rel_landing_time'
]
flights = flights_df[required_columns].copy() # make a copy to avoid SettingWithCopyWarning
print(f"Flight data loaded...")

## 1.4. Load weather data
# define weather csv file pathes
fast_milp_dir = script_dir / "Input"
weather_files = [
    str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t00min.csv")
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t00min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t05min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t10min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t15min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t20min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t25min.csv"),
    # str(fast_milp_dir / "infeasible_regions_fake_frames" / "infeasible_regions_t30min.csv"),
]

# Saves the weather CSV files into memory as a list of DataFrames
weather_dfs = []
for fp in weather_files: # fp means "file path"
    dfw = pd.read_csv(fp)
    dfw = dfw[['min_lat', 'max_lat', 'min_lon', 'max_lon']].dropna().reset_index(drop=True)
    weather_dfs.append(dfw)
print(f"Weather data loaded...")

## 1.5. Determine time steps
max_time = flights['rel_landing_time'].max() 
# Set the end of the time grid to the latest exit time across all flights, or to a fixed value (e.g., 2100 seconds = 35 min) if the max exit time is too short for testing purposes.
if max_time > 2100:
    t0 = 0
    tN = max_time

else:
    t0 = 0
    tN = 2100

flights['flight_entry_timestep'] = (flights['rel_entry_time'] / DT).astype(int) # find out which time step each flight comes in, 
                                                                                # and then append this information as a new column in the flights DataFrame for easier access later. 
                                                                                # This will be used to enforce that separation constraints and fuel costs only apply after the flight has entered the simulation.

N_steps  = int((tN - t0) / DT) + 1 # N_steps is both the number of time steps and the number of optimization steps.
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
env.setParam("OutputFlag", 0)   # disable all output from Gurobi
env.start()

m = Model("mip1", env=env)
print("Model created...")

# 2.2. Create decision variables
# i) Create decision variables of position and control inputs for every flight and every time step.
N_flights = len(flights)
x = []
y = []
z = []
ux = []
uy = []
uz = []
for i in range(1,N_flights+1):
    x.append(m.addVars(range(N_steps), name=f"f{i}_lat", lb = -100000))       # latitude of flight i at time step k, for k in [0, N_steps-1], with a very loose lower bound to avoid unboundedness issues in the early stages of optimization.
    y.append(m.addVars(range(N_steps), name=f"f{i}_lon", lb=-100000))         # longitude of flight i at time step k, for k in [0, N_steps-1], with a very loose lower bound to avoid unboundedness issues in the early stages of optimization.
    z.append(m.addVars(range(N_steps), name=f"f{i}_alt_ft"))                  # altitude of flight i at time step k, for k in [0, N_steps-1]
    ux.append(m.addVars(range(N_steps), name=f"uf{i}_x"))                     # control input in x direction for flight i at time step k, for k in [0, N_steps-1]
    uy.append(m.addVars(range(N_steps), name=f"uf{i}_y"))                     # control input in y direction for flight i at time step k, for k in [0, N_steps-1]
    uz.append(m.addVars(range(N_steps), name=f"uf{i}_z"))                     # control input in z direction for flight i at time step k, for k in [0, N_steps-1]

# ii) Add binary STAR fix decision variables
fix_names = list(star_fixes)                                            # Get list of STAR fix names
lat_vals  = [star_fixes[k][0] for k in fix_names]                       # Get list of latitude of each fix k
lon_vals  = [star_fixes[k][1] for k in fix_names]                       # Get list of longitude of each fix
alt_vals_min = [star_fixes[k][2][1] for k in fix_names]                 # Get list of minimum altitude of each fix
alt_vals_max = [star_fixes[k][2][0] for k in fix_names]                 # Get list of maximum altitude of each fix
k_fixes = len(fix_names)                                                # Count number of STAR fixes
b = []
for i in range(N_flights):
    b.append(m.addVars(k_fixes, vtype=GRB.BINARY, name=f"y{i+1}"))      # Binary variable y[i][k] is 1 if flight i chooses fix k, and 0 otherwise.

# iii) Add cost function variables
diffx = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dx") # add variable difference_x, diffx[i,k] for ith flight at time step k will be defined with x[i][k] - x[i][k-1]
diffy = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dy") # add variable difference_y, diffy[i,k] for ith flight at time step k will be defined with y[i][k] - y[i][k-1]
diffz = m.addVars(N_flights, N_steps, lb=-GRB.INFINITY, name="dz") # add variable difference_z, diffz[i,k] for ith flight at time step k will be defined with z[i][k] - z[i][k-1]
is_end = m.addVars(N_flights, N_steps, vtype=GRB.BINARY, name="is_end")  # add binary variable to indicate whether the flight has reached its chosen STAR fix at time step k, is_end[i,k] ∈ {0,1} for ith flight at time step k
                                                           # is_end[i,k] = 1 means the flight has reached its chosen STAR fix at time step k, and is_end[i,k] = 0 means it has not reached its chosen STAR fix at time step k.
speed = m.addVars(N_flights, N_steps, name="speed") # add variable for speed at time step k for flight i, which will be defined with sqrt(diffx[i,k]^2 + diffy[i,k]^2)
tfuel = m.addVars(N_flights, N_steps, name="t") # add variable for fuel usage at time step k for flight i, which will be defined with compute_fuel_emission_flow()) when is_end[i,k] = 0 (active branch), and 0 when is_end[i,k] = 1 (inactive branch)
print("Decision variables created...")

# 2.3. Define Objective Function
obj = LinExpr()

for i in range(N_flights):
    entry_k = int(flights.iloc[i]['flight_entry_timestep'])

    for k in range(entry_k + 1, N_steps):
        # Define differences
        m.addConstr(diffx[i, k] == x[i][k] - x[i][k-1])
        m.addConstr(diffy[i, k] == y[i][k] - y[i][k-1])
        m.addConstr(diffz[i, k] == z[i][k] - z[i][k-1])

        # Absolute steps
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

        # cost for altitude smoothness
        active = 1 - is_end[i, k]
        obj += CF * tfuel[i, k]
        obj += CT * DT * active 
        obj += CSMOOTH * (ux[i][k-1] + uy[i][k-1] + ALPHA_Z * uz[i][k-1])

# Set objective
m.setObjective(obj, GRB.MINIMIZE)
print("Objective function created...")


# 2.4. Define constraints
# i) Add is_end logic constraints
for i in range(N_flights):
    entry_k = flights.iloc[i]['flight_entry_timestep']
    
    # Before entry: is_end must be 0 (aircraft hasn't entered yet)
    for k in range(entry_k + 1):
        m.addConstr(is_end[i, k] == 0, f"is_end_before_entry_{i}_{k}")
    
    # After entry: enforce monotonicity (once is_end becomes 1, it stays 1)
    for k in range(entry_k + 2, N_steps):
        m.addConstr((is_end[i, k] == 1) >> (x[i][k] == x[i][N_steps-1])) # if is_end = 1, then the flight has reached its chosen STAR fix in the x direction
        m.addConstr((is_end[i, k] == 1) >> (y[i][k] == y[i][N_steps-1])) # if is_end = 1, then the flight has reached its chosen STAR fix in the y direction
        m.addConstr((is_end[i, k] == 1) >> (z[i][k] == z[i][N_steps-1])) # if is_end = 1, then the flight has reached its chosen STAR fix in the z direction
        m.addConstr(is_end[i, k] >= is_end[i, k-1], f"is_end_monotonic_{i}_{k}")
print("is_end logic constraints created...")

# ii) Add entry point constraints
for i in range(N_flights):
    entry_k = flights.iloc[i]['flight_entry_timestep'] # flight enters airspace at timestep k
    for k in range(entry_k + 1): # Constraint aircraft to be at entry point from time step 0 up to and including the entry time step k, then allow it to start moving after that.
        m.addConstr(x[i][k] == flights.iloc[i]['entry_lat'], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(y[i][k] == flights.iloc[i]['entry_lon'], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(z[i][k] == flights.iloc[i]['entry_alt'], f"c_pre_entry_z_{i}_t{k}")
print("Entry point constraints created...")

# iii) Add STAR fix (exit point) constraints
for j in range(N_flights):
    m.addConstr(quicksum(b[j][i] for i in range(k_fixes)) == 1, f"one_fix{j+1}") # Ensure only one binary variable is 1 for flight j -> exactly one fix is chosen for each flight
    m.addConstr(x[j][N_steps-1] == LinExpr(lat_vals, b[j].values()),  f"lat_choice{j+1}")
    m.addConstr(y[j][N_steps-1] == LinExpr(lon_vals, b[j].values()),  f"lon_choice{j+1}")
    m.addConstr(z[j][N_steps-1] <= LinExpr(alt_vals_max, b[j].values()), f"alt_choice_max{j+1}")
    m.addConstr(z[j][N_steps-1] >= LinExpr(alt_vals_min, b[j].values()), f"alt_choice_min{j+1}")
print(f"STAR fix constraints created...")

# iv) Add max x, y, z constraints for each flight after entry time step
for i in range(N_flights): # for each flight i
    entry_k = flights.iloc[i]['flight_entry_timestep'] # This loop now starts from the time step *after* the aircraft enters,
    for k in range(entry_k + 1, N_steps): # for time steps after the entry time step, add constraints and costs. Before the entry time step, the aircraft is stationary at the entry point, so we don't need to add constraints or costs.
        # # Physical constraints
        m.addConstr(x[i][k] - x[i][k-1] <=  V_MAX_X*DT)
        m.addConstr(y[i][k] - y[i][k-1] <=  V_MAX_Y*DT)
        m.addConstr(z[i][k] - z[i][k-1] <=  V_MAX_Z*DT)

        m.addConstr(x[i][k-1] - x[i][k] <=  V_MAX_X*DT)
        m.addConstr(y[i][k-1] - y[i][k] <=  V_MAX_Y*DT)
        m.addConstr(z[i][k-1] - z[i][k] <=  V_MAX_Z*DT)
print("Max speed constraints created...")

# v) Add weather constraints to each flight and each time step based on the weather frames loaded into memory.
if len(weather_dfs) > 0:
    last_idx = len(weather_dfs) - 1

    for i in range(N_flights): # for each flight i
        entry_k = flights.iloc[i]['flight_entry_timestep'] # flight enters airspace at timestep k

        for k in range(entry_k, N_steps): # start applying weather constraints from the entry time step k to final timestep N_steps
            t_sec = k * DT # convert time step k to actual time in seconds
            frame_idx = int(t_sec // WEATHER_STEP_SEC) # determine which weather frame index to use based on the actual time in seconds of current time step k
            if frame_idx > last_idx: # if the computed frame index exceeds the available weather frames, hold it at the last available frame index to avoid index out of range errors. 
                frame_idx = last_idx # This means that after we run out of new weather frames, we will keep applying the constraints of the last weather frame for all subsequent time steps.

            dfw = weather_dfs[frame_idx] # get the weather DataFrame for the current time step k based on the computed frame index
            if dfw.empty:
                continue # if there are no weather constraints for this frame, skip to the next iteration without adding any constraints to avoid errors from trying to add constraints with an empty DataFrame

            # Force the aircraft point (x[i][k], y[i][k]) to be outside the weather rectangle
            for r, row in dfw.iterrows():
                out = m.addVars(4, vtype=GRB.BINARY, name=f"w_out_{i}_{k}_{frame_idx}_{r}") # Binary variables to determine which side of the weather rectangle the aircraft is on: left, right, below, or above. 
                m.addConstr(out.sum() >= 1, name=f"w_outside_{i}_{k}_{frame_idx}_{r}") # force at least one side to be chosen

                m.addConstr(x[i][k] <= row['min_lat'] - WEATHER_EPS_DEG + BIG_M * (1 - out[0]), # out[0] is 0 -> forces x[i][k] to be less than or equal to row['min_lat'] - WEATHER_EPS_DEG -> meaning the aircraft must be on the left side of the left boundary of the weather rectangle.
                            name=f"w_left_{i}_{k}_{frame_idx}_{r}")
                m.addConstr(x[i][k] >= row['max_lat'] + WEATHER_EPS_DEG - BIG_M * (1 - out[1]), # out[1] is 0 -> forces x[i][k] to be greater than or equal to row['max_lat'] + WEATHER_EPS_DEG -> meaning the aircraft must be on the right side of the right boundary of the weather rectangle.
                            name=f"w_right_{i}_{k}_{frame_idx}_{r}")
                m.addConstr(y[i][k] <= row['min_lon'] - WEATHER_EPS_DEG + BIG_M * (1 - out[2]), # out[2] is 0 -> forces y[i][k] to be less than or equal to row['min_lon'] - WEATHER_EPS_DEG -> meaning the aircraft must be below the bottom boundary of the weather rectangle.
                            name=f"w_below_{i}_{k}_{frame_idx}_{r}")
                m.addConstr(y[i][k] >= row['max_lon'] + WEATHER_EPS_DEG - BIG_M * (1 - out[3]), # out[3] is 0 -> forces y[i][k] to be greater than or equal to row['max_lon'] + WEATHER_EPS_DEG -> meaning the aircraft must be above the top boundary of the weather rectangle.
                            name=f"w_above_{i}_{k}_{frame_idx}_{r}")
print("Weather constraints created...")

# vi) defin consne safety seperatiotraints
for k in range(N_steps):
    for i in range(N_flights-1):
        for j in range(i+1,N_flights): # Loop over every pair of aircraft (i, j), without repeating pairs
            if k >= flights.iloc[i]['flight_entry_timestep'] and k >= flights.iloc[j]['flight_entry_timestep']: # Only enforce separation if BOTH aircraft have entered the airspace.
                bin_vars = m.addVars(range(6), name='bin', vtype=GRB.BINARY) # binary variables for the 6 possible separation scenarios: i left of j, i right of j, i above j, i below j, i in front of j, i behind j

                m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[0]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[0] = 1, this constraint enforces that x[i][k] - x[j][k] >= SEP_HOR_NM, meaning i is to the right of j by at least the horizontal separation minimum.
                m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[1]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[1] = 1, this constraint enforces that y[i][k] - y[j][k] >= SEP_HOR_NM, meaning i is above j by at least the horizontal separation minimum.
                m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[2] = 1, this constraint enforces that z[i][k] - z[j][k] >= SEP_VERT_FT, meaning i is above j by at least the vertical separation minimum.
                m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[3]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[3] = 1, this constraint enforces that x[j][k] - x[i][k] >= SEP_HOR_NM, meaning j is to the right of i by at least the horizontal separation minimum.
                m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[4]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[4] = 1, this constraint enforces that y[j][k] - y[i][k] >= SEP_HOR_NM, meaning j is above i by at least the horizontal separation minimum.
                m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[5]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k]) # when bin_vars[5] = 1, this constraint enforces that z[j][k] - z[i][k] >= SEP_VERT_FT, meaning j is above i by at least the vertical separation minimum.

                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5]>= 1) # at least one of the separation scenarios must be true to confirm seperation
print("Separation constraints created...")

# 2.6. Initiate optimization
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
        for k in range(k_fixes):
            if b[i][k].X > 0.5:  # Binary variable is 1 (with tolerance for numerical issues)
                chosen_fix = fix_names[k]
                fix_lat, fix_lon, (fix_alt_max, fix_alt_min) = star_fixes[chosen_fix]
                final_alt = z[i][N_steps-1].X
                print(f'  {flight_id}: {chosen_fix} (lat={fix_lat:.4f}, lon={fix_lon:.4f}, alt={final_alt:.0f} ft)')
                break
    print()

    # 3.2. Extract the optimized trajectories for each flight and save them to a CSV file for visualization and analysis.
    # i) initialize pattern to extract the optimized lat, lon and alt_ft for each flight j at each time step i
    rows = []
    for k in range(N_steps): # for each time step k
        row = {"t": k * DT}  # convert time step k to actual time in seconds
        for i in range(N_flights): # for each flight j
            row[f"f{i+1}_lat"]    = x[i][k].X # .X means we want to extract the optimized value
            row[f"f{i+1}_lon"]    = y[i][k].X 
            row[f"f{i+1}_alt_ft"] = z[i][k].X
        rows.append(row)

    df_wide = pd.DataFrame(rows) # convert pd into a DataFrame

    # Save the results to a CSV file
    output_dir = script_dir / "Output"
    output_dir.mkdir(parents=True, exist_ok=True) # create the output directory if it doesn't exist
    output_path = output_dir / "weathertrialstatic.csv" # define the output file path for the optimized trajectories CSV file
    df_wide.to_csv(output_path, index=False) # save the optimized trajectories to a CSV file without the index column
    print(f"Results saved to {output_path}")
    print()
    
    # 3.3. Analyze and visualize optimized trajectory
    print(" === ANALYZING OPTIMIZED TRAJECTORY ===")
    # Prepare aircraft list with acId and acType
    aircraft_list = []
    for idx in range(N_flights):
        acId = flights.iloc[idx]['acId']
        # Default to B737 if acType not available
        acType = "B737"  # TODO: Add acType to flights dataframe if needed
        aircraft_list.append({"acId": acId, "acType": acType})
    
    # Call the analysis function
    results = analyze_optimized_trajectory(df_wide, aircraft_list)
    print("Analysis and visualization complete!")
else:
    print("Optimization was not successful. Status code:", m.status)
    df_wide = None