from gurobipy import *
import numpy as np
import pandas as pd
from main_ver4 import *
import math

DT = 60.0
FT2NM             = 1 / 6076.12

BIG_M             = 1e5
V_MAX_X  = 0.25/60    # grid units per s
V_MAX_Y  = 0.072/60    
V_MAX_Z  = 1000/60
GLIDE_RATIO = 2
# SEP_HOR_NM        = 500.0 * FT2NM  + DT*V_MAX_X
# SEP_VERT_FT       = 100.0 + DT*V_MAX_Z
SEP_HOR_NM = 500.0 * FT2NM
SEP_VERT_FT = 100.0

CT = 999
CF = 1

# DAL1066, DAL498, EDV5018
# Format: [start_lat, start_lon, start_alt, entry_time_sec, end_lat, end_lon, end_alt, landing_time_sec]
# flights = [
#     [39.471957, -82.139821, 34843.470164, 0,    41.673148, -82.943072, 19820.938696, 2100],
#     [39.471965, -82.139803, 34406.851392, 900,  41.673149, -82.943096, 20227.116630, 2400] 
# ]
# df = pd.DataFrame({
#             'd_ts': [60, 60, 60, 60, 60],              # seconds
#             'groundSpeed': [250, 255, 260, 265, 270],   # knots
#             'alt': [10000, 12000, 14000, 16000, 18000], # feet
#             'rateOfClimb': [500, 600, 700, 800, 900]    # ft/min
#         })

# # print(df)
# print(df)

# Example aircraft parameters
S = 122.6      # m^2 (wing area)
mtow = 70000   # kg (max takeoff weight)
tsfc = 0.00003 # kg/Ns (thrust specific fuel consumption)
cd0 = 0.02     # zero-lift drag coefficient
k = 0.045      # induced drag factor

# # Calculate total fuel usage only
# total_fuel = compute_fuel_emission_for_flight(df, S, mtow, tsfc, cd0, k, limit=True, cal_emission=False)
# print(f"Total fuel used: {total_fuel:.2f} kg")


try:
    flights_df = pd.read_csv("entry_exit_points.csv")
    flights_df = flights_df.sort_values(by='entry_rectime').reset_index(drop=True)
    flights_df = flights_df[:10]
    print(flights_df)
    flights_df['entry_rectime'] = pd.to_datetime(flights_df['entry_rectime'])
    flights_df['exit_rectime'] = pd.to_datetime(flights_df['exit_rectime'])
    min_time = flights_df['entry_rectime'].min()
    max_time = flights_df['exit_rectime'].max()
    print(f"Reference start time (t=0): {min_time}", f"End time: {max_time}")

    # Calculate entry and landing times in total seconds relative to the min_time
    flights_df['entry_time_sec'] = (flights_df['entry_rectime'] - min_time).dt.total_seconds()
    flights_df['landing_time_sec'] = (flights_df['exit_rectime'] - min_time).dt.total_seconds()

    # flights_df = flights_df.sort_values(by='entry_time_sec').reset_index(drop=True)

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

v_avg = []
for i in range(len(flights)):
    dt = (flights[i][8]-flights[i][4])
    v_x = abs(flights[i][1]-flights[i][5])/(dt)
    v_y = abs(flights[i][2]-flights[i][6])/(dt)
    v_z = abs(flights[i][3]-flights[i][7])/(dt)
    v_avg.append([v_x,v_y,v_z])

# print(v_avg)

star_fixes ={
        "BONZZ": (41.7483, -82.7972, (21000, 15000)), "CRAKN": (41.6730, -82.9405, (26000, 12000)), "CUUGR": (42.3643, -83.0975, (11000, 10000)),
        "FERRL": (42.4165, -82.6093, (10000, 8000)), "GRAYT": (42.9150, -83.6020, (22000, 17000)), "HANBL": (41.7375, -84.1773, (21000, 17000)),
        "HAYLL": (41.9662, -84.2975, (11000, 11000)), "HTROD": (42.0278, -83.3442, (12000, 12000)), "KKISS": (42.5443, -83.7620, (15000, 12000)),
        "KLYNK": (41.8793, -82.9888, (10000, 9000)), "LAYKS": (42.8532, -83.5498, (10000, 10000)), "LECTR": (41.9183, -84.0217, (10000, 8000)),
        "RKCTY": (42.6869, -83.9603, (13000, 11000)), "VCTRZ": (41.9878, -84.0670, (15000, 12000)) # (lat, lon)
}

# The simulation runs until the latest scheduled landing time.
# if flights:
#     # Set t0 to the earliest entry time and tN to the latest landing time.
#     t0 = 0
#     tN = max(f[8] for f in flights)
#     print(tN)
# else:
#     t0 = 0
#     tN = 2100 
max_time = max(f[8] for f in flights)
if max_time > 210000:
    t0 = 0
    tN = max_time

else:
    t0 = 0
    tN = 210000
    
print(f"Actual number of time steps:{((max_time-t0)/DT)+1}")
N  = int((tN - t0) / DT) + 1
print(f"Number of time steps: {N}")
times = np.linspace(t0, tN, N, dtype=int)

# Convert entry times in seconds to discrete time step indices
entry_indices = [int(f[4] / DT) for f in flights]
print(f"Entry time indices for flights: {entry_indices}")

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

for i in range(n):
    # The original code only constrained the position at k=0.
    # Now, we fix the aircraft's position at its entry point for all time steps
    # up to and including its entry time index.
    entry_k = entry_indices[i]
    for k in range(entry_k + 1):
        m.addConstr(x[i][k] == flights[i][1], f"c_pre_entry_x_{i}_t{k}")
        m.addConstr(y[i][k] == flights[i][2], f"c_pre_entry_y_{i}_t{k}")
        m.addConstr(z[i][k] == flights[i][3], f"c_pre_entry_z_{i}_t{k}")


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

is_end = [[m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}') for k in range(N)] for i in range(n)]

landed = [[m.addVar(vtype=GRB.BINARY, name=f'landed_{i}_{k}') for k in range(N)] for i in range(n)]

obj = LinExpr()

for i in range(n):
    entry_k = entry_indices[i]
    # This loop now starts from the time step *after* the aircraft enters,
    # ensuring that dynamics constraints and costs only apply when it's flying.
    for k in range(entry_k + 1, N):
        # # Physical constraints
        # m.addConstr(x[i][k] - x[i][k-1] <=  V_MAX_X*DT)
        # m.addConstr(y[i][k] - y[i][k-1] <=  V_MAX_Y*DT)
        # m.addConstr(z[i][k] - z[i][k-1] <=  V_MAX_Z*DT)

        # m.addConstr(x[i][k-1] - x[i][k] <=  V_MAX_X*DT)
        # m.addConstr(y[i][k-1] - y[i][k] <=  V_MAX_Y*DT)
        # m.addConstr(z[i][k-1] - z[i][k] <=  V_MAX_Z*DT)
        m.addConstr(x[i][k] - x[i][k-1] <=  v_x*DT)
        m.addConstr(y[i][k] - y[i][k-1] <=  v_y*DT)
        m.addConstr(z[i][k] - z[i][k-1] <=  v_z*DT)

        m.addConstr(x[i][k-1] - x[i][k] <=  v_x*DT)
        m.addConstr(y[i][k-1] - y[i][k] <=  v_y*DT)
        m.addConstr(z[i][k-1] - z[i][k] <=  v_z*DT)


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
        
        # for i in range(len(ptu)-1):
        #     slope = (ptf[i+1]-ptf[i])/(ptu[i+1]-ptu[i])
        #     c = ptf[i] - slope*ptu[i]
        # if tas == 0 : print("WARNING: true air speed is 0")

        # if vs/tas >= 2 or vs/tas<=-2: print("WARNING: vs/ts out of range, flight angle too steep") 
        gamma = m.addVar()
        x = m.addVar(lb=lbx, ub=ubx, vtype=GRB.CONTINUOUS, name="z")
        m.addGenConstrPWL(x,gamma,x_pts,y_pts,"PWLarctan")
        fuel_flow = compute_fuel_emission_flow(speed, z[i][k], diffz1, gamma, mtow,  122.6, cd0, k, tsfc, m, limit=True, cal_emission=True)
        # obj += (ux[i][k-1]-uz[i][k-1]*FT2NM*(1/18)*(1-pos))
        # obj += (uy[i][k-1]-uz[i][k-1]*FT2NM*(1/18)*(1-pos))
        # obj += uz[i][k-1]*FT2NM*pos
        obj += fuel_flow*DT
        obj += (CT/CF)*(1-is_end)


for k in range(N):
    # Safety constraints
    for i in range(n-1):
        for j in range(i+1,n):
            # This 'if' condition ensures that separation constraints are only enforced
            # at a time k if both aircraft i and j have entered the simulation.
            if k >= entry_indices[i] and k >= entry_indices[j]:
                bin_vars = m.addVars(range(6), name='bin', vtype=GRB.BINARY)
                m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5]>= 1)
                # m.addConstr(x[i][k] - x[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[0]))
                # m.addConstr(y[i][k] - y[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[1]))
                # m.addConstr(z[i][k] - z[j][k] >=  SEP_VERT_FT - BIG_M*(1 - bin_vars[2]))
                # m.addConstr(x[j][k] - x[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[3]))
                # m.addConstr(y[j][k] - y[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[4]))
                # m.addConstr(z[j][k] - z[i][k] >=  SEP_VERT_FT - BIG_M*(1 - bin_vars[5]))

                m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[0]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[1]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[3]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[4]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])
                m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[5]) - BIG_M*landed[i][k] - BIG_M*landed[j][k])



m.setObjective(obj, GRB.MINIMIZE)
m.optimize()

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
    df["t"]  = (df["var"].str.extract(r"\[(\d+)\]",  expand=False).astype(int))

    wide = (df.pivot(index="t", columns="root", values="value")
              .sort_index()
              .reset_index())

    ordered = ['t']
    for i in range(n):
        ordered.extend([f'f{i+1}_lat', f'f{i+1}_lon', f'f{i+1}_alt_ft'])

    wide = wide[ordered + [c for c in wide.columns if c not in ordered]]

    wide.to_csv("trial.csv", index=False)
    # print("Results saved to staggered_entry_10.csv")
else:
    print("Optimization was not successful. Status code:", m.status)