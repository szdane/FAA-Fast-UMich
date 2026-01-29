#!/usr/bin/env python3
"""
Two‑aircraft 4‑D MILP trajectory optimiser
==========================================

Example
-------
python milp_two_flights.py                          \
    --entry1   0   0   10000   0                   \
    --landing1 50  40      0 600                  \
    --entry2   10  60  12000   0                   \
    --landing2 55 -10      0 600                  \
    --output_csv sol.csv

Dependencies
------------
pip install pulp pandas numpy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

# ------------- global parameters ------------------------------------------------
SEP_HOR  = 3.0     # [NM]
SEP_VERT = 1000    # [ft]
DT       = 30      # [s]

V_MAX_X  = 0.05    # grid units per 30 s  ≈ 420 kt
V_MAX_Y  = 0.02    # idem
V_MAX_Z  = 1.0     # 100 ft per 30 s

GLIDE_RATIO = 18   # |Δx|+|Δy| ≥ Δz / 18 when descending
BIG_M        = 1e4

# ------------- model builder ----------------------------------------------------
def build_model(ent1, land1, ent2, land2):
    # ----- common time grid (assume identical t0/tN/DT for simplicity) ----------
    t0 = int(ent1["t"])
    tN = int(land1["t"])
    assert t0 == int(ent2["t"]) and tN == int(land2["t"]), \
        "For this simple version, both flights must share t0 and tN"
    N  = int((tN - t0) / DT) + 1
    times = np.linspace(t0, tN, N, dtype=int)

    prob = pulp.LpProblem("Two_Flight_4D_Trajectory", sense=pulp.LpMinimize)

    # -- decision vars for flights 1 and 2 --------------------------------------
    def make_state(prefix):
        x = pulp.LpVariable.dicts(f'{prefix}x', range(N))
        y = pulp.LpVariable.dicts(f'{prefix}y', range(N))
        z = pulp.LpVariable.dicts(f'{prefix}z', range(N))
        ux = pulp.LpVariable.dicts(f'{prefix}ux', range(1, N), lowBound=0)
        uy = pulp.LpVariable.dicts(f'{prefix}uy', range(1, N), lowBound=0)
        uz = pulp.LpVariable.dicts(f'{prefix}uz', range(1, N), lowBound=0)
        return (x, y, z, ux, uy, uz)

    x1, y1, z1, ux1, uy1, uz1 = make_state("f1_")
    x2, y2, z2, ux2, uy2, uz2 = make_state("f2_")

    # ----- helper to add kinematics & glide constraints -------------------------
    def add_kinematic_constraints(x, y, z, ux, uy, uz, entry, landing):
        # boundary
        prob += (x[0] == entry["x"])
        prob += (y[0] == entry["y"])
        prob += (z[0] == entry["z"])
        prob += (x[N-1] == landing["x"])
        prob += (y[N-1] == landing["y"])
        prob += (z[N-1] == landing["z"])

        # per‑step bounds + 1‑norm linearisation
        for k in range(1, N):
            prob += (x[k] - x[k-1] <=  V_MAX_X * DT)
            prob += (x[k-1] - x[k] <=  V_MAX_X * DT)
            prob += (y[k] - y[k-1] <=  V_MAX_Y * DT)
            prob += (y[k-1] - y[k] <=  V_MAX_Y * DT)
            prob += (z[k] - z[k-1] <=  V_MAX_Z * DT)
            prob += (z[k-1] - z[k] <=  V_MAX_Z * DT)

            prob += (x[k] - x[k-1] <= ux[k])
            prob += (x[k-1] - x[k] <= ux[k])
            prob += (y[k] - y[k-1] <= uy[k])
            prob += (y[k-1] - y[k] <= uy[k])
            prob += (z[k] - z[k-1] <= uz[k])
            prob += (z[k-1] - z[k] <= uz[k])

            prob += (ux[k] + uy[k] >= uz[k] / GLIDE_RATIO)  # glide rule

    add_kinematic_constraints(x1, y1, z1, ux1, uy1, uz1, ent1, land1)
    add_kinematic_constraints(x2, y2, z2, ux2, uy2, uz2, ent2, land2)

    # ----- mutual separation via Big‑M disjunction ------------------------------
    for k in range(N):
        y1b = pulp.LpVariable(f"sep_y1_{k}", cat="Binary")
        y2b = pulp.LpVariable(f"sep_y2_{k}", cat="Binary")
        y3b = pulp.LpVariable(f"sep_y3_{k}", cat="Binary")
        y4b = pulp.LpVariable(f"sep_y4_{k}", cat="Binary")
        y5b = pulp.LpVariable(f"sep_y5_{k}", cat="Binary")
        y6b = pulp.LpVariable(f"sep_y6_{k}", cat="Binary")
        prob += (y1b + y2b + y3b + y4b + y5b + y6b >= 1)

        prob += ( x1[k] - x2[k] >=  SEP_HOR  - BIG_M*(1 - y1b))
        prob += (-x1[k] + x2[k] >=  SEP_HOR  - BIG_M*(1 - y2b))
        prob += ( y1[k] - y2[k] >=  SEP_HOR  - BIG_M*(1 - y3b))
        prob += (-y1[k] + y2[k] >=  SEP_HOR  - BIG_M*(1 - y4b))
        prob += ( z1[k] - z2[k] >=  SEP_VERT - BIG_M*(1 - y5b))
        prob += (-z1[k] + z2[k] >=  SEP_VERT - BIG_M*(1 - y6b))

    # ----- objective: combined 1‑norm increments -------------------------------
    obj = (
        pulp.lpSum([ux1[k] + uy1[k] + 0.1*uz1[k] for k in range(1, N)]) +
        pulp.lpSum([ux2[k] + uy2[k] + 0.1*uz2[k] for k in range(1, N)])
    )
    prob += obj

    return prob, times, (x1, y1, z1, x2, y2, z2)

# ------------- CLI -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Two‑flight MILP optimiser")
    for i in [1, 2]:
        p.add_argument(f'--entry{i}',   nargs=4, type=float, required=True,
                       metavar=('x','y','z','t'),
                       help=f'Entry point for flight {i}: x y z t')
        p.add_argument(f'--landing{i}', nargs=4, type=float, required=True,
                       metavar=('x','y','z','t'),
                       help=f'Landing point for flight {i}: x y z t')
    p.add_argument('--output_csv', type=str, default='solution.csv')
    return p.parse_args()

# ------------- main ------------------------------------------------------------
def main():
    args     = parse_args()
    ent1     = dict(zip(['x','y','z','t'], getattr(args, 'entry1')))
    land1    = dict(zip(['x','y','z','t'], getattr(args, 'landing1')))
    ent2     = dict(zip(['x','y','z','t'], getattr(args, 'entry2')))
    land2    = dict(zip(['x','y','z','t'], getattr(args, 'landing2')))

    model, times, (x1,y1,z1,x2,y2,z2) = build_model(ent1, land1, ent2, land2)
    model.solve(pulp.PULP_CBC_CMD(msg=True))

    sol = pd.DataFrame({
        't' : times,
        'x1': [pulp.value(x1[k]) for k in range(len(times))],
        'y1': [pulp.value(y1[k]) for k in range(len(times))],
        'z1': [pulp.value(z1[k]) for k in range(len(times))],
        'x2': [pulp.value(x2[k]) for k in range(len(times))],
        'y2': [pulp.value(y2[k]) for k in range(len(times))],
        'z2': [pulp.value(z2[k]) for k in range(len(times))],
    })
    sol.to_csv(args.output_csv, index=False, float_format='%.6f')
    print(sol)

if __name__ == '__main__':
    main()
