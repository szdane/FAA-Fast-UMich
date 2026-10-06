from gurobipy import *
import numpy as np
import pandas as pd
from main_ver4_gurobi_debug import *
import math

DT = 60.0*5
FT2NM             = 1 / 6076.12

BIG_M             = 1e5
V_MAX_X  = 0.14/60    # grid units per s
V_MAX_Y  = 0.072/60 
# V_MAX_X = 250
# V_MAX_Y = 250
V_MAX_Z  = 1000/60
GLIDE_RATIO = 2
# SEP_HOR_NM        = 500.0 * FT2NM  + DT*V_MAX_X
# SEP_VERT_FT       = 100.0 + DT*V_MAX_Z
SEP_HOR_NM = 500.0 * FT2NM
SEP_VERT_FT = 100.0

CT = 1
CF = 1.5

WEATHER_EPS_DEG = 1e-4
EDGE_ALPHAS = [0.25, 0.5, 0.75]
M_LAT = 10.0
M_LON = 10.0

# DAL1066, DAL498, EDV5018
# Format: [start_lat, start_lon, start_alt, entry_time_sec, end_lat, end_lon, end_alt, landing_time_sec]
# flights = [
#     [39.471957, -82.139821, 34843.470164, 0,    41.673148, -82.943072, 19820.938696, 2100],
#     [39.471965, -82.139803, 34406.851392, 900,  41.673149, -82.943096, 20227.116630, 2400] 
# ]
# flights = [[0, 43.050, -83.850, 30000.0, 0, 41.673148, -82.943072, 19820.938696, 2100]]
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
    # flights_df = pd.concat ([pd.read_csv("../2.1.1_Input_Data/entry_exit_points.csv")[:1], (pd.read_csv("../2.1.1_Input_Data/entry_exit_points.csv")[2:3])], ignore_index=True)
    flights_df = pd.read_csv("../2.1.1_Input_Data/entry_exit_points.csv")[:2]
    print(flights_df)
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

WEATHER_STEP_SEC = 300  # 5 minutes

weather_files = [
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t00min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t05min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t10min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t15min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t20min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t25min.csv", 
    "../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_fake_east/infeasible_regions_t30min.csv", 
]

# weather_files = ["../2.1.1_Input_Data/infeasible_regions_testing/infeasible_regions_static_single/infeasible_regions_t00min.csv"]
weather_dfs = []
for fp in weather_files:
    try:
        dfw = pd.read_csv(fp)
        dfw = dfw[['min_lat', 'max_lat', 'min_lon', 'max_lon']].dropna().reset_index(drop=True)
        weather_dfs.append(dfw)
    except FileNotFoundError:
        print(f"WARNING: missing weather file: {fp}")
    except Exception as e:
        print(f"WARNING: could not read {fp}: {e}")

# If nothing loaded, use empty list
if len(weather_dfs) == 0:
    print("No weather south loaded; skipping weather constraints.")



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

# star_fixes = {'BONZZ': (np.float64(46142.63262673297), np.float64(-51455.44055057798), (21000, 15000)), 'CRAKN': (np.float64(34294.83535890254), np.float64(-59895.72883085624), (26000, 12000)), 
#               'CUUGR': (np.float64(21024.51683518695), np.float64(16922.101415528916), (11000, 10000)), 'FERRL': (np.float64(61083.1599606936), np.float64(22961.796796482908), (10000, 8000)), 
#               'GRAYT': (np.float64(-20245.67574577981), np.float64(78155.09504878709), (22000, 17000)), 'HANBL': (np.float64(-68361.78373155238), np.float64(-52477.4658479923), (21000, 17000)), 
#               'HAYLL': (np.float64(-78054.67623105628), np.float64(-26945.079163737624), (11000, 11000)), 'HTROD': (np.float64(759.9551189871775), np.float64(-20526.540523690936), (12000, 12000)), 
#               'KKISS': (np.float64(-33474.12951926423), np.float64(36985.80493080173), (15000, 12000)), 'KLYNK': (np.float64(30185.5684596365), np.float64(-36974.57610805848), (10000, 9000)), 
#               'LAYKS': (np.float64(-16010.594434343164), np.float64(71272.13930802463), (10000, 10000)), 'LECTR': (np.float64(-55294.67078726591), np.float64(-32486.361165977654), (10000, 8000)), 
#               'RKCTY': (np.float64(-49605.97832190019), np.float64(52938.81765769922), (13000, 11000)), 'VCTRZ': (np.float64(-58978.255222062704), np.float64(-24728.180763285345), (15000, 12000))}

max_time = max(f[8] for f in flights)
if max_time > 48000:
    t0 = 0
    tN = max_time

else:
    t0 = 0
    tN = 48000

print("tN", tN)
    
print(f"Actual number of time steps:{((max_time-t0)/DT)+1}")
N  = int((tN - t0) / DT) + 1
print(f"Number of time steps: {N}")
times = np.linspace(t0, tN, N, dtype=int)

# Convert entry times in seconds to discrete time step indices
entry_indices = [int(f[4] / DT) for f in flights]
print(f"Entry time indices for flights: {entry_indices}")

m = Model("mip1")
m.Params.NonConvex = 2

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

# Per-(aircraft, timestep) capsule geometry, populated in the main dynamics
# loop below and consumed by the pairwise capsule separation constraints.
seg_geom = [dict() for _ in range(n)]

for i in range(n):
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

# is_end = [[m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}') for k in range(N)] for i in range(n)]

# Active-corner capsule separation constraints - see capsule_geometry.py for
# the geometry/derivation and test_capsule_geometry.py for the verification
# of each of the 8 direction cases. capsule_geometry_slope.py is a second,
# independent implementation of the same constraint (scalar-slope form
# instead of vector dot/cross-products, closer to the original handwritten
# derivation) - same function signatures, so it's a drop-in swap for the
# two imports below if ever wanted; both are checked against each other and
# against a reference formula in test_capsule_geometry.py.
from capsule_geometry_slope import build_seg_geom_slope, add_capsule_direction_slope


# if len(weather_dfs) > 0:
#     last_idx = len(weather_dfs) - 1

#     for i in range(n):
#         entry_k = entry_indices[i]

#         for k in range(entry_k, N):
#             t_sec = k * DT
#             frame_idx = int(t_sec // WEATHER_STEP_SEC)
#             if frame_idx > last_idx:
#                 frame_idx = last_idx  # hold the last frame after 30 min for now

#             dfw = weather_dfs[frame_idx]
#             if dfw.empty:
#                 continue

#             # One set of "outside rectangle" constraints per rectangle row
#             for r, row in dfw.iterrows():
#                 out = m.addVars(4, vtype=GRB.BINARY, name=f"w_out_{i}_{k}_{frame_idx}_{r}")
#                 m.addConstr(out.sum() >= 1, name=f"w_outside_{i}_{k}_{frame_idx}_{r}")

#                 m.addConstr(x[i][k] <= row['min_lat'] - WEATHER_EPS_DEG + BIG_M * (1 - out[0]),
#                             name=f"w_left_{i}_{k}_{frame_idx}_{r}")
#                 m.addConstr(x[i][k] >= row['max_lat'] + WEATHER_EPS_DEG - BIG_M * (1 - out[1]),
#                             name=f"w_right_{i}_{k}_{frame_idx}_{r}")
#                 m.addConstr(y[i][k] <= row['min_lon'] - WEATHER_EPS_DEG + BIG_M * (1 - out[2]),
#                             name=f"w_below_{i}_{k}_{frame_idx}_{r}")
#                 m.addConstr(y[i][k] >= row['max_lon'] + WEATHER_EPS_DEG - BIG_M * (1 - out[3]),
#                             name=f"w_above_{i}_{k}_{frame_idx}_{r}")
#         # sample the trajectory in between two waypoints to have a better discretization to 
#         # stay out of weather regions in between two nodes
#         for k in range(entry_k, N - 1):
#             for s, alpha in enumerate(EDGE_ALPHAS):
#                 t_sec_alpha = (k + alpha) * DT
#                 frame_idx = int(t_sec_alpha // WEATHER_STEP_SEC)
#                 if frame_idx > last_idx:
#                     frame_idx = last_idx

#                 dfw = weather_dfs[frame_idx]
#                 if dfw.empty:
#                     continue

#                 x_alpha = (1 - alpha) * x[i][k] + alpha * x[i][k + 1]
#                 y_alpha = (1 - alpha) * y[i][k] + alpha * y[i][k + 1]

#                 for r, row in dfw.iterrows():
#                     out = m.addVars(4, vtype=GRB.BINARY, name=f"w_edge_out_{i}_{k}_{s}_{frame_idx}_{r}")
#                     m.addConstr(out.sum() >= 1, name=f"w_edge_outside_{i}_{k}_{s}_{frame_idx}_{r}")

#                     m.addConstr(
#                         x_alpha <= row['min_lat'] - WEATHER_EPS_DEG + M_LAT * (1 - out[0]),
#                         name=f"w_edge_left_{i}_{k}_{s}_{frame_idx}_{r}"
#                     )
#                     m.addConstr(
#                         x_alpha >= row['max_lat'] + WEATHER_EPS_DEG - M_LAT * (1 - out[1]),
#                         name=f"w_edge_right_{i}_{k}_{s}_{frame_idx}_{r}"
#                     )
#                     m.addConstr(
#                         y_alpha <= row['min_lon'] - WEATHER_EPS_DEG + M_LON * (1 - out[2]),
#                         name=f"w_edge_below_{i}_{k}_{s}_{frame_idx}_{r}"
#                     )
#                     m.addConstr(
#                         y_alpha >= row['max_lon'] + WEATHER_EPS_DEG - M_LON * (1 - out[3]),
#                         name=f"w_edge_above_{i}_{k}_{s}_{frame_idx}_{r}"
#                     )
                

obj = LinExpr()

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
        # m.addConstr(x[i][k] - x[i][k-1] <=  v_x*DT)
        # m.addConstr(y[i][k] - y[i][k-1] <=  v_y*DT)
        # m.addConstr(z[i][k] - z[i][k-1] <=  v_z*DT)

        # m.addConstr(x[i][k-1] - x[i][k] <=  v_x*DT)
        # m.addConstr(y[i][k-1] - y[i][k] <=  v_y*DT)
        # m.addConstr(z[i][k-1] - z[i][k] <=  v_z*DT)


        # Dummy variables for the objective
        diffx1 = m.addVar(lb=-GRB.INFINITY, name=f'dx{i}_{k}')     
        m.addConstr(diffx1 == x[i][k] - x[i][k-1])
        m.addConstr(ux[i][k-1] == abs_(diffx1))  

        diffy1 = m.addVar(lb=-GRB.INFINITY, name=f'dy{i}_{k}')
        m.addConstr(diffy1 == y[i][k] - y[i][k-1])
        m.addConstr(uy[i][k-1] == abs_(diffy1))

        # Capsule geometry for the active-corner separation constraints:
        # segment length and direction-sign indicators for (diffx1, diffy1).
        seg_geom[i][k] = build_seg_geom_slope(m, diffx1, diffy1, tag=f"{i}_{k}")

        diffz1 = m.addVar(lb=-GRB.INFINITY, name=f'dz{i}_{k}')
        pos = m.addVar(vtype=GRB.BINARY)    
        m.addConstr(diffz1 == z[i][k] - z[i][k-1])
        m.addGenConstrIndicator(pos, 1, diffz1, GRB.GREATER_EQUAL, 1e-6, name=f"pos_is_one_{i}_{k}")
        m.addGenConstrIndicator(pos, 0, diffz1, GRB.LESS_EQUAL,     0.0, name=f"pos_is_zero_{i}_{k}")
        m.addConstr(uz[i][k-1] == abs_(diffz1)) 

        # is_end = m.addVar(vtype=GRB.BINARY, name=f'is_end_{i}_{k}')
        m.addConstr((is_end[i][k] == 1) >> (x[i][k] == x[i][N-1]))
        m.addConstr((is_end[i][k] == 1) >> (y[i][k] == y[i][N-1]))
        m.addConstr((is_end[i][k] == 1) >> (z[i][k] == z[i][N-1]))


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
        # gamma = m.addVar()
        # lx = m.addVar(lb=lbx, ub=ubx, vtype=GRB.CONTINUOUS, name="z")
        # m.addGenConstrPWL(lx,gamma,x_pts,y_pts,"PWLarctan")

        # t = m.addVar(name="t")

        # m.addGenConstrIndicator(is_end[i][k], 0, t == compute_fuel_emission_flow(speed, z[i][k], diffz1, 0.8*mtow, S, cd0, k, tsfc, m, limit=True, cal_emission=False, mode="full"))   # active branch
        # m.addGenConstrIndicator(is_end[i][k], 1, t == 0)
        # obj += (ux[i][k-1]-uz[i][k-1])*FT2NM*(1/18)*(1-pos)
        # obj += (uy[i][k-1]-uz[i][k-1])*FT2NM*(1/18)*(1-pos)
        # obj += uz[i][k-1]*FT2NM*pos
        obj += ux[i][k-1] + uy[i][k-1] + (uz[i][k-1])*FT2NM
        # obj += t
        obj += (CT/CF)*(1-is_end[i][k])


# for k in range(N):
#     # Safety constraints
#     for i in range(n-1):
#         for j in range(i+1,n):
#             # This 'if' condition ensures that separation constraints are only enforced
#             # at a time k if both aircraft i and j have entered the simulation.
#             if k >= entry_indices[i] and k >= entry_indices[j]:
#                 bin_vars = m.addVars(range(6), name='bin', vtype=GRB.BINARY)
#                 m.addConstr(bin_vars[0]+bin_vars[1]+bin_vars[2]+bin_vars[3]+bin_vars[4]+bin_vars[5]>= 1)
#                 # m.addConstr(x[i][k] - x[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[0]))
#                 # m.addConstr(y[i][k] - y[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[1]))
#                 # m.addConstr(z[i][k] - z[j][k] >=  SEP_VERT_FT - BIG_M*(1 - bin_vars[2]))
#                 # m.addConstr(x[j][k] - x[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[3]))
#                 # m.addConstr(y[j][k] - y[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin_vars[4]))
#                 # m.addConstr(z[j][k] - z[i][k] >=  SEP_VERT_FT - BIG_M*(1 - bin_vars[5]))

#                 m.addConstr(x[i][k] - x[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[0]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])
#                 m.addConstr(y[i][k] - y[j][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[1]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])
#                 m.addConstr(z[i][k] - z[j][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])
#                 m.addConstr(x[j][k] - x[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[3]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])
#                 m.addConstr(y[j][k] - y[i][k] >= SEP_HOR_NM - BIG_M*(1 - bin_vars[4]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])
#                 m.addConstr(z[j][k] - z[i][k] >= SEP_VERT_FT - BIG_M*(1 - bin_vars[5]) - BIG_M*is_end[i][k] - BIG_M*is_end[j][k])

# ---------------------------------------------------------------------------
# Pairwise separation, per timestep k: aircraft i and j must be separated
# either horizontally (by >= SEP_HOR_NM) or vertically (by >= SEP_VERT_FT).
#
#   SEP(i,j,k)  =  CAPSULE(i's segment, j's point)
#              OR  CAPSULE(j's segment, i's point)
#              OR  VERTICAL(i,j,k)
#
# Horizontally, instead of only checking the two aircraft's positions at the
# same timestep k (a point-vs-point check, blind to what happens between
# k-1 and k), each direction treats one aircraft's motion from waypoint k-1
# to k as sweeping a disk of radius SEP_HOR_NM along that segment (a
# "capsule"/stadium shape) and requires the other aircraft's position at k
# to stay outside it - i.e. distance(other aircraft's point, this
# aircraft's segment) >= SEP_HOR_NM. Both directions are checked (i's
# segment vs j's point, and j's segment vs i's point) since neither alone
# guarantees the other aircraft's own mid-segment motion is safe.
#
# CAPSULE(segment, point):
#   Let (x_n,y_n) = near waypoint (k-1), (x_f,y_f) = far waypoint (k),
#   (x_o,y_o) = the other aircraft's point, delta = SEP_HOR_NM,
#   dx = x_f - x_n, dy = y_f - y_n, L = sqrt(dx^2 + dy^2). Define:
#
#     along_far  = dx*(x_o - x_f) + dy*(y_o - y_f)   (dot product: >=0 iff
#                  the point's along-track projection is past the far
#                  waypoint, in the direction of travel)
#     along_near = dx*(x_o - x_n) + dy*(y_o - y_n)   (dot product: <=0 iff
#                  the point's projection is before the near waypoint)
#     perp       = dx*(y_o - y_n) - dy*(x_o - x_n)   (cross product: the
#                  point's signed perpendicular offset from the segment's
#                  line, scaled by L)
#
#   Because dx,dy are the segment's own signed direction components, all
#   three formulas are already direction-consistent for any orientation -
#   no case-by-case sign handling is needed (unlike a scalar-slope
#   formulation y = m*x + b, which is sign-ambiguous and would need one).
#
#   The point is outside the capsule (>= delta from the segment) iff it
#   clears the segment via at least one of four disjuncts - the standard
#   decomposition of point-to-segment distance by nearest feature:
#
#     (A) perp >= delta*L                    [clear of the strip, one side]
#     (B) perp <= -delta*L                   [clear of the strip, other side]
#     (C) along_far  >= 0  AND  (x_f-x_o)^2 + (y_f-y_o)^2 >= delta^2
#                                             [past the far endpoint AND
#                                              outside the far endpoint's
#                                              circle - the far end-cap]
#     (D) along_near <= 0  AND  (x_n-x_o)^2 + (y_n-y_o)^2 >= delta^2
#                                             [before the near endpoint AND
#                                              outside the near endpoint's
#                                              circle - the near end-cap]
#
#   (A) and (B) are each sufficient alone because distance-to-segment is
#   always >= distance-to-the-infinite-line; the two end-cap disjuncts (C)
#   and (D) need both their along-track test and their circle test
#   together, since being past an endpoint isn't itself enough - the point
#   also has to actually clear that endpoint's circle.
#
# MILP encoding (see capsule_geometry.add_capsule_direction): each disjunct
# is gated by a binary (case_A..case_D, sum >= 1) and relaxed by big-M when
# not selected. A stationary segment (L=0) makes (A)/(B) vacuously true
# (perp and delta*L both 0), which would let the solver "cheat" past real
# checking, so case_A+case_B is additionally capped at a "moves" flag that's
# 1 only when the segment has non-negligible length - forcing the (still
# meaningful, circle-based) end-cap disjuncts to carry stationary segments.
#
# VERTICAL(i,j,k): the original point-vs-point formulation, unchanged -
# |z[i][k] - z[j][k]| >= SEP_VERT_FT.
#
# All three top-level disjuncts are further relaxed by BIG_M*is_end[i][k]
# and BIG_M*is_end[j][k], so the constraint switches off once either
# aircraft has reached (and is being held at) its terminal waypoint.
# ---------------------------------------------------------------------------
for k in range(N):
    for i in range(n - 1):
        for j in range(i + 1, n):
            if k >= entry_indices[i] and k >= entry_indices[j]:
                bin_vars = m.addVars(range(3), vtype=GRB.BINARY, name=f'bin_{i}_{j}_{k}')
                m.addConstr(bin_vars[0] + bin_vars[1] + bin_vars[2] >= 1, name=f"sep_or_{i}_{j}_{k}")

                relax_common = BIG_M * is_end[i][k] + BIG_M * is_end[j][k]

                # Branch 0: aircraft i's segment [k-1,k] capsule vs aircraft j's point at k
                if k >= entry_indices[i] + 1:
                    add_capsule_direction_slope(
                        m, seg_geom[i][k],
                        x[i][k], y[i][k], x[i][k-1], y[i][k-1],
                        x[j][k], y[j][k],
                        bin_vars[0], relax_common, SEP_HOR_NM,
                        tag=f"ij_{i}_{j}_{k}", big_m=BIG_M)
                else:
                    # aircraft i has no real segment yet (just entered) - fall back
                    # to a plain point separation check for this branch
                    m.addConstr(
                        (x[i][k]-x[j][k])*(x[i][k]-x[j][k]) + (y[i][k]-y[j][k])*(y[i][k]-y[j][k]) >=
                        SEP_HOR_NM*SEP_HOR_NM - BIG_M*(1-bin_vars[0]) - relax_common,
                        name=f"cap_fallback_ij_{i}_{j}_{k}")

                # Branch 1: aircraft j's segment [k-1,k] capsule vs aircraft i's point at k
                if k >= entry_indices[j] + 1:
                    add_capsule_direction_slope(
                        m, seg_geom[j][k],
                        x[j][k], y[j][k], x[j][k-1], y[j][k-1],
                        x[i][k], y[i][k],
                        bin_vars[1], relax_common, SEP_HOR_NM,
                        tag=f"ji_{i}_{j}_{k}", big_m=BIG_M)
                else:
                    m.addConstr(
                        (x[i][k]-x[j][k])*(x[i][k]-x[j][k]) + (y[i][k]-y[j][k])*(y[i][k]-y[j][k]) >=
                        SEP_HOR_NM*SEP_HOR_NM - BIG_M*(1-bin_vars[1]) - relax_common,
                        name=f"cap_fallback_ji_{i}_{j}_{k}")

                # Branch 2: vertical separation (unchanged from the original formulation)
                m.addConstr(
                    (z[i][k] - z[j][k]) * (z[i][k] - z[j][k]) >=
                    SEP_VERT_FT * SEP_VERT_FT - BIG_M*(1 - bin_vars[2]) - relax_common,
                    name=f"sep_vert_{i}_{j}_{k}")

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
    df["t"]  = (df["var"].str.extract(r"\[(\d+)\]",  expand=False).astype(int))*DT

    wide = (df.pivot(index="t", columns="root", values="value")
            .sort_index()
            .reset_index())

    ordered = ['t']
    for i in range(n):
        ordered.extend([f'f{i+1}_lat', f'f{i+1}_lon', f'f{i+1}_alt_ft'])

    wide = wide[ordered + [c for c in wide.columns if c not in ordered]]

    wide.to_csv("../../2.3_Outputs_and_Results/test.csv", index=False)
    print("Results saved to test.csv")
else:
    print("Optimization was not successful. Status code:", m.status)