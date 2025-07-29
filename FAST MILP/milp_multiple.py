from gurobipy import *
import numpy as np
import pandas as pd

DT = 60.0 
FT2NM             = 1 / 6076.12
SEP_HOR_NM        = 500.0 * FT2NM      # 0.082 NM
SEP_VERT_FT       = 100.0
BIG_M             = 1e5
V_MAX_X  = 0.25    # grid units per 30 s
V_MAX_Y  = 0.072    
V_MAX_Z  = 1000
GLIDE_RATIO = 2
#first 4 indices -> entry, last 4 -> landing

#DENtoDTW BWItoDTW
# flight1 = [40.67333,-80.76722, 34000.0, 0, 41.51444,  -82.56222,    10000.0, 900]
# flight2 = [43.08,  -86.2251,  37000.0,   0, 42.6068, -83.9982, 10000.0, 900 ]
# flights = [flight1, flight2]


#DAL1066, DAL498, EDV5018
flights = [[39.471957,-82.139821,34843.470164, 0, 41.673148, -82.943072, 19820.938696, 1200],[39.471965,-82.139803,34406.851392,0,41.673149, -82.943096, 20227.116630, 900]]#,[44.62722,-77.79222, 36000.0,900]]
v_avg = []
for i in range(len(flights)):
    v_x = abs(flights[i][0]-flights[i][4])/20
    v_y = abs(flights[i][1]-flights[i][5])/20
    v_z = abs(flights[i][2]-flights[i][6])/20
    v_avg.append([v_x,v_y,v_z])

print(v_avg)

# HAYLL and LAYKS only had lowerbound constraint on the altitude, VCTRZ didn't have any and I put the one for the closest starfix
star_fixes ={
        "BONZZ": (41.7483, -82.7972, (21000, 15000)), "CRAKN": (41.6730, -82.9405, (26000, 12000)), "CUUGR": (42.3643, -83.0975, (11000, 10000)),
        "FERRL": (42.4165, -82.6093, (10000, 8000)), "GRAYT": (42.9150, -83.6020, (22000, 17000)), "HANBL": (41.7375, -84.1773, (21000, 17000)),
        "HAYLL": (41.9662, -84.2975, (11000, 11000)), "HTROD": (42.0278, -83.3442, (12000, 12000)), "KKISS": (42.5443, -83.7620, (15000, 12000)),
        "KLYNK": (41.8793, -82.9888, (10000, 9000)), "LAYKS": (42.8532, -83.5498, (10000, 10000)), "LECTR": (41.9183, -84.0217, (10000, 8000)),
        "RKCTY": (42.6869, -83.9603, (13000, 11000)), "VCTRZ": (41.9878, -84.0670, (15000, 12000)) # (lat, lon)
}

tN = flights[0][7]
t0 = 0
N  = int((tN - t0) / DT) + 1
print(N)
times = np.linspace(t0, tN, N, dtype=int)

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
    # Entry point constraints
    m.addConstr(x[i][0] == flights[i][0], f"c{i}1")
    m.addConstr(y[i][0] == flights[i][1], f"c{i}2")
    m.addConstr(z[i][0] == flights[i][2], f"c{i}3")

    # End point altitude constraints
    # m.addConstr(z[i][N-1] == 10000, f"c{i}4")


fix_names = list(star_fixes)           
lat_vals  = [star_fixes[k][0] for k in fix_names]
lon_vals  = [star_fixes[k][1] for k in fix_names]
alt_vals_min = [star_fixes[k][2][1] for k in fix_names]
alt_vals_max = [star_fixes[k][2][0] for k in fix_names]
# print(alt_vals_min, alt_vals_max)
k = len(fix_names)
b = []
for i in range(n):
    # star-fix constraints for the end point
    b.append(m.addVars(k, vtype=GRB.BINARY, name=f"y{i+1}"))

for j in range(n):
    m.addConstr(quicksum(b[j][i] for i in range(k)) == 1, f"one_fix{j+1}")

    # chosen coordinates = Σ y_i * constant
    m.addConstr(x[j][N-1] == LinExpr(lat_vals, b[j].values()),  f"lat_choice{j+1}")
    m.addConstr(y[j][N-1] == LinExpr(lon_vals, b[j].values()),  f"lon_choice{j+1}")
    m.addConstr(z[j][N-1] <= LinExpr(alt_vals_max, b[j].values()), f"alt_choice_max{j+1}")
    m.addConstr(z[j][N-1] >= LinExpr(alt_vals_min, b[j].values()), f"alt_choice_min{j+1}")


obj = LinExpr()
diffx = []
diffy = []
diffz = []
for k in range(1,N):
    #physical constraints
    for i in range(n):
        m.addConstr(x[i][k] - x[i][k-1] <=  v_avg[i][0]*1.1)
        m.addConstr(y[i][k] - y[i][k-1] <=  v_avg[i][1]*1.1)
        m.addConstr(z[i][k] - z[i][k-1] <=  v_avg[i][2]*1.1)

        m.addConstr(x[i][k-1] - x[i][k] <=  v_avg[i][0]*1.1)
        m.addConstr(y[i][k-1] - y[i][k] <=  v_avg[i][1]*1.1)
        m.addConstr(z[i][k-1] - z[i][k] <=  v_avg[i][2]*1.1)


        #dummy variables for the objective
        diffx1 = m.addVar(lb=-GRB.INFINITY, name=f'dx{i}_{k}')     
        m.addConstr(diffx1 == x[i][k] - x[i][k-1])
        m.addConstr(ux[i][k-1] == abs_(diffx1))  

        diffy1 = m.addVar(lb=-GRB.INFINITY, name=f'dy{i}_{k}')     
        m.addConstr(diffy1 == y[i][k] - y[i][k-1])
        m.addConstr(uy[i][k-1] == abs_(diffy1))  

        diffz1 = m.addVar(lb=-GRB.INFINITY, name=f'dz{i}_{k}') 
        pos = m.addVar(vtype=GRB.BINARY)    
        m.addConstr(diffz1 == z[i][k] - z[i][k-1])
        m.addGenConstrIndicator(pos, 1, diffz1, GRB.GREATER_EQUAL, 1e-6, name="pos_is_one")

        # pos = 0  ⇒  diffz1 ≤ 0
        m.addGenConstrIndicator(pos, 0, diffz1, GRB.LESS_EQUAL,     0.0, name="pos_is_zero")
        m.addConstr(uz[i][k-1] == abs_(diffz1)) 

        is_end = m.addVar(vtype=GRB.BINARY, name=f'is_end{i}')
        # m.addConstr((and_([x[i][k] == x[i][N-1], y[i][k] == y[i][N-1], z[i][k] == z[i][N-1]]))>>is_end_x)
        m.addConstr((is_end == 1) >> (x[i][k] == x[i][N-1]))
        m.addConstr((is_end == 1) >> (y[i][k] == y[i][N-1]))
        m.addConstr((is_end == 1) >> (z[i][k] == z[i][N-1]))


        #Fuel usage based on the distance
        # obj += ux[i][k-1]
        # obj += uy[i][k-1]
        # obj += uz[i][k-1]*FT2NM

        #Fuel usage with gliding effect
        obj += (ux[i][k-1]-uz[i][k-1]*FT2NM*(1/18)*(1-pos))
        obj += (uy[i][k-1]-uz[i][k-1]*FT2NM*(1/18)*(1-pos))
        obj += uz[i][k-1]*FT2NM*pos
        # obj += 30*(1-is_end)




for i in range(N):
    #safety constraints
    for i in range(n-1):
        for j in range(i+1,n):
            bin = m.addVars(range(6), name='bin', vtype=GRB.BINARY)
            m.addConstr(bin[0]+bin[1]+bin[2]+bin[3]+bin[4]+bin[5]>= 1)
            m.addConstr(x[i][k] - x[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin[0]))
            m.addConstr(y[i][k] - y[j][k] >=  SEP_HOR_NM - BIG_M*(1 - bin[1]))
            m.addConstr(z[i][k] - z[j][k] >=  SEP_VERT_FT - BIG_M*(1 - bin[2]))
            m.addConstr(x[j][k] - x[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin[3]))
            m.addConstr(y[j][k] - y[i][k] >=  SEP_HOR_NM - BIG_M*(1 - bin[4]))
            m.addConstr(z[j][k] - z[i][k] >=  SEP_VERT_FT - BIG_M*(1 - bin[5]))


m.setObjective(obj, GRB.MINIMIZE)

m.optimize()

for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))

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
df["t"]  = (df["var"].str.extract(r"\[(\d+)\]",  expand=False).astype(int))*60.0000

# 3. turn long → wide: one row per index, one column per root
wide = (df.pivot(index="t", columns="root", values="value")
          .sort_index()
          .reset_index())

# print(wide.head())

ordered = ['t']
for i in range(n):
    ordered.extend([f'f{i+1}_lat', f'f{i+1}_lon', f'f{i+1}_alt_ft'])

wide = wide[ordered + [c for c in wide.columns if c not in ordered]]

wide.to_csv("solution24.csv", index=False)
