"""
Dependencies: `pip install pulp pandas numpy` 

Usage:

python milp_traj_optimization.py  \
    --entry  0  0 10000 0   \
    --landing 50 40     0 600 \
    --intruder_csv intruder.csv
    --output_csv solution.csv

KDEN_KDTW:

python milp_traj_optimization.py  \
    --entry   43.08 -86.225 370 0   \    
    --landing 42.595  -83.97861 130 930 \
    --intruder_csv filtered_flight.csv \
    --output_csv solution.csv


KBWI_KDTW:

python milp_traj_optimization.py  \
    --entry  40.67333 -80.76722  340 0   \
    --landing 41.51444 -82.56222 193 900 \
    --intruder_csv filtered_flight_den.csv \
    --output_csv solution1.csv

Note: `intruder.csv` must contain columns `t,x,y,z` on the same 30 s grid.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

SEP_HOR = 3.0   # horizontal nautical miles 
SEP_VERT = 1000 # feet
DT = 30      # time step [s]
V_MAX_X = 0.05  # ~ 420 kt ground speed in x‑dir (unit/DT) DT: 30
V_MAX_Y = 0.02   # idem for y DT:30
V_MAX_Z = 1   # 100 ft/s climb / descent DT:30
# MIN_HOR_SPEED = 1.0
BIG_M   = 1e4

GLIDE_RATIO = 18 

def build_model(entry, landing, intruder):
    t0 = entry['t']
    tN = landing['t']
    N  = int((tN - t0) / DT) + 1
    times = np.linspace(t0, tN, N, dtype=int)

    prob = pulp.LpProblem("4D_Trajectory", sense=pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts('x', range(N))
    y = pulp.LpVariable.dicts('y', range(N))
    z = pulp.LpVariable.dicts('z', range(N))

    # Aux vars for 1‑norm on control increments
    ux = pulp.LpVariable.dicts('ux', range(1, N), lowBound=0)
    uy = pulp.LpVariable.dicts('uy', range(1, N), lowBound=0)
    uz = pulp.LpVariable.dicts('uz', range(1, N), lowBound=0)

    # Constraints: boundary conditions
    prob += (x[0] == entry['x'])
    prob += (y[0] == entry['y'])
    prob += (z[0] == entry['z'])

    prob += (x[N-1] == landing['x'])
    prob += (y[N-1] == landing['y'])
    prob += (z[N-1] == landing['z'])

    # Kinematic + absolute‑value linearisation
    for k in range(1, N):
        # speed bounds |Δ| ≤ V_MAX * DT
        prob += (x[k] - x[k-1] <=  V_MAX_X * DT)
        prob += (x[k-1] - x[k] <=  V_MAX_X * DT)
        prob += (y[k] - y[k-1] <=  V_MAX_Y * DT)
        prob += (y[k-1] - y[k] <=  V_MAX_Y * DT)
        prob += (z[k] - z[k-1] <=  V_MAX_Z * DT)
        prob += (z[k-1] - z[k] <=  V_MAX_Z * DT)

        # 1‑norm slack |Δ| = u_x etc.
        prob += (x[k] - x[k-1] <= ux[k])
        prob += (x[k-1] - x[k] <= ux[k])
        prob += (y[k] - y[k-1] <= uy[k])
        prob += (y[k-1] - y[k] <= uy[k])
        prob += (z[k] - z[k-1] <= uz[k])
        prob += (z[k-1] - z[k] <= uz[k])

        # If descending, enforce x/y must change accordingly
        prob += (ux[k] + uy[k] >= uz[k] / GLIDE_RATIO)

    # Separation constraints vs intruder using Big‑M disjunction
    for k, t in enumerate(times):
        xI, yI, zI = intruder.get(t, (None, None, None))
        if xI is None:
            continue  

        y1 = pulp.LpVariable(f"y1_{k}", cat="Binary")
        y2 = pulp.LpVariable(f"y2_{k}", cat="Binary")
        y3 = pulp.LpVariable(f"y3_{k}", cat="Binary")
        y4 = pulp.LpVariable(f"y4_{k}", cat="Binary")
        y5 = pulp.LpVariable(f"y5_{k}", cat="Binary")
        y6 = pulp.LpVariable(f"y6_{k}", cat="Binary")

        # At least one condition active
        prob += (y1 + y2 + y3 + y4 + y5 + y6 >= 1)

        # Encode |x - xI| ≥ SEP_HOR OR |y - yI| ≥ SEP_HOR OR |z - zI| ≥ SEP_VERT
        prob += ( x[k] - xI >=  SEP_HOR - BIG_M*(1 - y1))
        prob += (-x[k] + xI >=  SEP_HOR - BIG_M*(1 - y2))
        prob += ( y[k] - yI >=  SEP_HOR - BIG_M*(1 - y3))
        prob += (-y[k] + yI >=  SEP_HOR - BIG_M*(1 - y4))
        prob += ( z[k] - zI >=  SEP_VERT - BIG_M*(1 - y5))
        prob += (-z[k] + zI >=  SEP_VERT - BIG_M*(1 - y6))

    # Objective: minimise sum of 1‑norm increments (proxy for fuel)
    prob += pulp.lpSum([ux[k] + uy[k] + 0.1*uz[k] for k in range(1, N)])

    return prob, times, (x, y, z)

def parse_args():
    p = argparse.ArgumentParser(description="4‑D MILP Trajectory Optimiser")
    p.add_argument('--entry',   nargs=4, type=float, required=True, metavar=('x','y','z','t'))
    p.add_argument('--landing', nargs=4, type=float, required=True, metavar=('x','y','z','t'))
    p.add_argument('--intruder_csv', type=str, required=True)
    p.add_argument('--output_csv', type=str, default='solution.csv')
    return p.parse_args()


def main():
    args = parse_args()

    entry   = dict(zip(['x','y','z','t'], args.entry))
    landing = dict(zip(['x','y','z','t'], args.landing))

    intr_df = pd.read_csv(args.intruder_csv)
    intruder = {int(r.t): (r.x, r.y, r.z) for r in intr_df.itertuples()}

    model, times, (x,y,z) = build_model(entry, landing, intruder)
    model.solve(pulp.PULP_CBC_CMD(msg=True))

    sol = pd.DataFrame({
        't': times,                                
        'x': [float(pulp.value(x[k])) for k in range(len(times))],
        'y': [float(pulp.value(y[k])) for k in range(len(times))],
        'z': [float(pulp.value(z[k])) for k in range(len(times))],
    })

    with pd.option_context('display.float_format', '{:.3f}'.format):
        print(sol)

    sol.to_csv(args.output_csv, index=False, float_format='%.6f')
    print(sol)

if __name__ == '__main__':
    main()
