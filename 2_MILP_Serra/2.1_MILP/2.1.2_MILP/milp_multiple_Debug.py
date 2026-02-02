# Step 1: Imports and global constants
from gurobipy import *
import numpy as np
import pandas as pd
from main_ver4_gurobi_debug import *
import math
import os
import sys

DT = 60.0
FT2NM             = 1 / 6076.12

BIG_M             = 1e5  
V_MAX_X = 250
V_MAX_Y = 250
V_MAX_Z  = 1000/60
GLIDE_RATIO = 2

SEP_HOR_NM = 500.0 * FT2NM
SEP_VERT_FT = 100.0

CT = 1
CF = 1.5

WEATHER_EPS_DEG = 1e-4

# Example aircraft parameters
S = 122.6      # m^2 (wing area)
mtow = 70000   # kg (max takeoff weight)
tsfc = 0.00003 # kg/Ns (thrust specific fuel consumption)
cd0 = 0.02     # zero-lift drag coefficient
k = 0.045      # induced drag factor


# Step 2: Load and preprocess flight entry/exit data
try:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir, "2.1.1_Input_Data", "entry_exit_points.csv")
    flights_df = pd.read_csv(csv_path)[12:13]
    flights_df = flights_df.sort_values(by='entry_rectime').reset_index(drop=True)
    flights_df = flights_df
    print(flights_df)
    flights_df['entry_rectime'] = pd.to_datetime(flights_df['entry_rectime'])
    flights_df['exit_rectime'] = pd.to_datetime(flights_df['exit_rectime'])
    min_time = flights_df['entry_rectime'].min()
    max_time = flights_df['exit_rectime'].max()
    print(f"Reference start time (t=0): {min_time}", f"End time: {max_time}")

    # Calculate entry and landing times in total seconds relative to the min_time
    flights_df['entry_time_sec'] = (flights_df['entry_rectime'] - min_time).dt.total_seconds()
    flights_df['landing_time_sec'] = (flights_df['exit_rectime'] - min_time).dt.total_seconds()

    # Create the final list of lists in the format required by the model:
    # [entry_lat, entry_lon, entry_alt, entry_time_sec, exit_lat, exit_lon, exit_alt, landing_time_sec]
    required_columns = ['acId',
        'entry_lat', 'entry_lon', 'entry_alt',
        'entry_time_sec',
        'exit_lat', 'exit_lon', 'exit_alt',
        'landing_time_sec'
    ]
    flights = flights_df[required_columns].values.tolist()
    print(flights)
    print(f"Successfully loaded {len(flights)} flights from entry_exit_points.csv")
except FileNotFoundError:
    print("Error: 'entry_exit_points.csv' not found. Please ensure the file is in the same directory as the script.")
    flights = []
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    flights = []


# Step 3: Define STAR fixes (candidate terminal waypoints)
star_fixes = {'BONZZ': (np.float64(46142.63262673297), np.float64(-51455.44055057798), (21000, 15000)), 'CRAKN': (np.float64(34294.83535890254), np.float64(-59895.72883085624), (26000, 12000)), 
              'CUUGR': (np.float64(21024.51683518695), np.float64(16922.101415528916), (11000, 10000)), 'FERRL': (np.float64(61083.1599606936), np.float64(22961.796796482908), (10000, 8000)), 
              'GRAYT': (np.float64(-20245.67574577981), np.float64(78155.09504878709), (22000, 17000)), 'HANBL': (np.float64(-68361.78373155238), np.float64(-52477.4658479923), (21000, 17000)), 
              'HAYLL': (np.float64(-78054.67623105628), np.float64(-26945.079163737624), (11000, 11000)), 'HTROD': (np.float64(759.9551189871775), np.float64(-20526.540523690936), (12000, 12000)), 
              'KKISS': (np.float64(-33474.12951926423), np.float64(36985.80493080173), (15000, 12000)), 'KLYNK': (np.float64(30185.5684596365), np.float64(-36974.57610805848), (10000, 9000)), 
              'LAYKS': (np.float64(-16010.594434343164), np.float64(71272.13930802463), (10000, 10000)), 'LECTR': (np.float64(-55294.67078726591), np.float64(-32486.361165977654), (10000, 8000)), 
              'RKCTY': (np.float64(-49605.97832190019), np.float64(52938.81765769922), (13000, 11000)), 'VCTRZ': (np.float64(-58978.255222062704), np.float64(-24728.180763285345), (15000, 12000))}

# Step 4: Establish time horizon and discretization
max_time = max(f[8] for f in flights)
if max_time > 2100:
    t0 = 0
    tN = max_time

else:
    t0 = 0
    tN = 2100
    
print(f"Actual number of time steps:{((max_time-t0)/DT)+1}")
N  = int((tN - t0) / DT) + 1
print(f"Number of time steps: {N}")
times = np.linspace(t0, tN, N, dtype=int)

# Convert entry times in seconds to discrete time step indices
entry_indices = [int(f[4] / DT) for f in flights]
print(f"Entry time indices for flights: {entry_indices}")

# Step 5: Create optimization model and decision variables
m = Model("mip1") 

n = len(flights)
x = []
y = []
z = []
ux = []
uy = []
uz = []
for i in range(1,n+1):
    x.append(m.addVars(range(N), name=f"f{i}_lat", lb = -100000))
    y.append(m.addVars(range(N), name=f"f{i}_lon", lb=-100000))
    z.append(m.addVars(range(N), name=f"f{i}_alt_ft"))
    ux.append(m.addVars(range(N), name=f"uf{i}_x"))
    uy.append(m.addVars(range(N), name=f"uf{i}_y"))
    uz.append(m.addVars(range(N), name=f"uf{i}_z"))

# Step 6: Fix aircraft states before entry time
for i in range(n):
    # The original code only constrained the position at k=0.
    # Now, we fix the aircraft's position at its entry point for all time steps
    # up to and including its entry time index.
    entry_k = entry_indices[i]
    for k in range(entry_k + 1):
        m.addConstr(x[i][k] == flights[i][1], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(y[i][k] == flights[i][2], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(z[i][k] == flights[i][3], f"c_pre_entry_z_{i}_t{k}")


# Step 7: STAR fix selection and terminal constraints
fix_names = list(star_fixes)           
lat_vals  = [star_fixes[k][0] for k in fix_names]
lon_vals  = [star_fixes[k][1] for k in fix_names]
alt_vals_min = [star_fixes[k][2][1] for k in fix_names]
alt_vals_max = [star_fixes[k][2][0] for k in fix_names]
k_fixes = len(fix_names) # Renamed from k to avoid conflict
b = []
for i in range(n):
    b.append(m.addVars(k_fixes, vtype=GRB.BINARY, name=f"y{i+1}"))

for j in range(n):
    m.addConstr(quicksum(b[j][i] for i in range(k_fixes)) == 1, f"one_fix{j+1}")
    m.addConstr(x[j][N-1] == LinExpr(lat_vals, b[j].values()),  f"lat_choice{j+1}")
    m.addConstr(y[j][N-1] == LinExpr(lon_vals, b[j].values()),  f"lon_choice{j+1}")
    m.addConstr(z[j][N-1] <= LinExpr(alt_vals_max, b[j].values()), f"alt_choice_max{j+1}")
    m.addConstr(z[j][N-1] >= LinExpr(alt_vals_min, b[j].values()), f"alt_choice_min{j+1}")

# Step 8: Create auxiliary binary state variables
is_end = [[m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}') for k in range(N)] for i in range(n)]

landed = [[m.addVar(vtype=GRB.BINARY, name=f'landed_{i}_{k}') for k in range(N)] for i in range(n)]


# Step 9: Initialize objective function
obj = LinExpr()

# Step 10: Add dynamics, kinematics, and fuel consumption constraints
for i in range(n):
    entry_k = entry_indices[i]
    # This loop now starts from the time step *after* the aircraft enters,
    # ensuring that dynamics constraints and costs only apply when it's flying.
    for k in range(entry_k + 1, N):
        # # Physical constraints
        m.addConstr(x[i][k] - x[i][k-1] <=  V_MAX_X*DT)
        m.addConstr(y[i][k] - y[i][k-1] <=  V_MAX_Y*DT)
        m.addConstr(z[i][k] - z[i][k-1] <=  V_MAX_Z*DT)

        m.addConstr(x[i][k-1] - x[i][k] <=  V_MAX_X*DT)
        m.addConstr(y[i][k-1] - y[i][k] <=  V_MAX_Y*DT)
        m.addConstr(z[i][k-1] - z[i][k] <=  V_MAX_Z*DT)


        # Dummy variables for the objective
        diffx1 = m.addVar(lb=-GRB.INFINITY, name=f'dx{i}_{k}')     
        m.addConstr(diffx1 == x[i][k] - x[i][k-1])
        m.addConstr(ux[i][k-1] == abs_(diffx1))  

        diffy1 = m.addVar(lb=-GRB.INFINITY, name=f'dy{i}_{k}')     
        m.addConstr(diffy1 == y[i][k] - y[i][k-1])
        m.addConstr(uy[i][k-1] == abs_(diffy1))  

        diffz1 = m.addVar(lb=-GRB.INFINITY, name=f'dz{i}_{k}') 
        pos = m.addVar(vtype=GRB.BINARY)    
        m.addConstr(diffz1 == z[i][k] - z[i][k-1])
        m.addGenConstrIndicator(pos, 1, diffz1, GRB.GREATER_EQUAL, 1e-6, name=f"pos_is_one_{i}_{k}")
        m.addGenConstrIndicator(pos, 0, diffz1, GRB.LESS_EQUAL,     0.0, name=f"pos_is_zero_{i}_{k}")
        m.addConstr(uz[i][k-1] == abs_(diffz1)) 

        is_end = m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}')
        m.addConstr((is_end == 1) >> (x[i][k] == x[i][N-1]))
        m.addConstr((is_end == 1) >> (y[i][k] == y[i][N-1]))
        m.addConstr((is_end == 1) >> (z[i][k] == z[i][N-1]))


        # Fuel usage with gliding effect
        speed = m.addVar()
        m.addConstr(speed*speed == diffx1*diffx1 + diffy1*diffy1)
        def f(u):  return math.atan(u)
        lbx = -2
        ubx =  2    
        npts = 101
        x_pts = []
        y_pts = []
        for p in range(npts):    
            x_pts.append(lbx + (ubx - lbx) * p / (npts - 1))    
            y_pts.append(f(x_pts[p]))

        # if vs/tas >= 2 or vs/tas<=-2: print("WARNING: vs/ts out of range, flight angle too steep") 
        gamma = m.addVar()
        lx = m.addVar(lb=lbx, ub=ubx, vtype=GRB.CONTINUOUS, name="z")
        m.addGenConstrPWL(lx,gamma,x_pts,y_pts,"PWLarctan")

        t = m.addVar(name="t")

        m.addGenConstrIndicator(is_end, 0, t == compute_fuel_emission_flow(speed, z[i][k], diffz1, 0.8*mtow, S, cd0, k, tsfc, m, limit=True, cal_emission=False, mode="full"))   # active branch
        m.addGenConstrIndicator(is_end, 1, t == 0)
        obj += t
        obj += (CT/CF)*(1-is_end)


# Step 11: Add pairwise separation (safety) constraints
for k in range(N):
    # Safety constraints
    for i in range(n-1):
        for j in range(i+1,n):
            # This 'if' condition ensures that separation constraints are only enforced
            # at a time k if both aircraft i and j have entered the simulation.
            if k >= entry_indices[i] and k >= entry_indices[j]:
                bin_vars = m.addVars(range(6), name='bin', vtype=GRB.BINARY)
                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5]>= 1)


                m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[0]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[1]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[3]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[4]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[5]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])



# Step 12: Set objective and optimize the model
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

# Step 13: Extract results and save trajectory data
if m.status == GRB.OPTIMAL:
    print('Obj: %g' % m.ObjVal)
    pat = []
    for i in range(N):
        for j in range(n):
            pat.append(f"f{j+1}_lat[{i}]")
            pat.append(f"f{j+1}_lon[{i}]")
            pat.append(f"f{j+1}_alt_ft[{i}]")

    data = {
        "var": [v.VarName for v in m.getVars() if v.VarName in pat],
        "value": [v.X       for v in m.getVars() if v.VarName in pat],
    }
    df = pd.DataFrame(data)

    df["root"] = df["var"].str.extract(r"^([^\[]+)", expand=False)        
    df["t"]  = (df["var"].str.extract(r"\[(\d+)\]",  expand=False).astype(int))*60

    wide = (df.pivot(index="t", columns="root", values="value")
              .sort_index()
              .reset_index())

    ordered = ['t']
    for i in range(n):
        ordered.extend([f'f{i+1}_lat', f'f{i+1}_lon', f'f{i+1}_alt_ft'])

    wide = wide[ordered + [c for c in wide.columns if c not in ordered]]

    # Save results to central outputs folder (script-relative, robust)
    out_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(out_root, "2.3_Outputs_and_Results", "res")
    os.makedirs(out_dir, exist_ok=True)
    wide.to_csv(os.path.join(out_dir, "weathertrial.csv"), index=False)
    # print("Results saved to staggered_entry_10.csv")
else:
    print("Optimization was not successful. Status code:", m.status)