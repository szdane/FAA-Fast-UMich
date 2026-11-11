"""SI migration of milp_multiple_Debug.py; run with --help for options.

Coordinates: east/north metres from DTW, altitude metres MSL, time seconds.
Original debug script and its in-progress edits are kept intact.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum

from capsule_geometry_si import build_segment, add_capsule, add_waypoint

FT2M = 0.3048
NM2M = 1852.0
DTW_LAT0, DTW_LON0 = 42.2125, -83.3534
DT = 300.0
SEP_HOR_M = 500.0 * FT2M
SEP_VERT_M = 100.0 * FT2M
VMAX_Z_MPS = (1000.0 / 60.0) * FT2M
CT, CF = 1.0, 1.5
WAYPOINT_SHAPE = "octagon"

# Geographic STAR source data: degrees and feet, converted before modelling.
STAR_FIXES = {'BONZZ': (41.7483, -82.7972, (21000, 15000)),
 'CRAKN': (41.673, -82.9405, (26000, 12000)),
 'CUUGR': (42.3643, -83.0975, (11000, 10000)),
 'FERRL': (42.4165, -82.6093, (10000, 8000)),
 'GRAYT': (42.915, -83.602, (22000, 17000)),
 'HANBL': (41.7375, -84.1773, (21000, 17000)),
 'HAYLL': (41.9662, -84.2975, (11000, 11000)),
 'HTROD': (42.0278, -83.3442, (12000, 12000)),
 'KKISS': (42.5443, -83.762, (15000, 12000)),
 'KLYNK': (41.8793, -82.9888, (10000, 9000)),
 'LAYKS': (42.8532, -83.5498, (10000, 10000)),
 'LECTR': (41.9183, -84.0217, (10000, 8000)),
 'RKCTY': (42.6869, -83.9603, (13000, 11000)),
 'VCTRZ': (41.9878, -84.067, (15000, 12000))}


def latlon_to_xy(lat, lon):
    # Same projection as main_v15's Functions.utilities._latlon_to_xy.
    import openap
    bearing = np.radians(openap.aero.bearing(DTW_LAT0, DTW_LON0, lat, lon))
    distance = openap.aero.distance(DTW_LAT0, DTW_LON0, lat, lon)
    return distance * np.sin(bearing), distance * np.cos(bearing)


def xy_to_latlon(x, y):
    import openap
    return openap.aero.latlon(DTW_LAT0, DTW_LON0, np.hypot(x, y),
                             np.degrees(np.arctan2(x, y)))


def build_model(flights, dt=DT, horizon_seconds=48000.0):
    if flights.empty:
        raise ValueError("No flights selected")
    if dt <= 0 or horizon_seconds <= 0:
        raise ValueError("Time step and horizon must be positive")
    flights = flights.sort_values("entry_rectime").reset_index(drop=True).copy()
    entry = pd.to_datetime(flights.entry_rectime)
    exit_time = pd.to_datetime(flights.exit_rectime)
    epoch = entry.min()
    entry_k = ((entry - epoch).dt.total_seconds() / dt).astype(int).tolist()
    horizon = max(horizon_seconds, (exit_time.max() - epoch).total_seconds())
    steps = math.ceil(horizon / dt) + 1
    count = len(flights)
    entry_x, entry_y = latlon_to_xy(flights.entry_lat.to_numpy(), flights.entry_lon.to_numpy())
    entry_z = flights.entry_alt.to_numpy() * FT2M
    fix_names = list(STAR_FIXES)
    fix_x, fix_y = latlon_to_xy(np.array([STAR_FIXES[f][0] for f in fix_names]),
                               np.array([STAR_FIXES[f][1] for f in fix_names]))
    fix_min = [STAR_FIXES[f][2][1] * FT2M for f in fix_names]
    fix_max = [STAR_FIXES[f][2][0] * FT2M for f in fix_names]
    # Approximate the debug model's degree/s limits at the DTW reference
    # latitude. Axes swap: old x=latitude, new x=east (old longitude).
    origin_x, origin_y = latlon_to_xy(DTW_LAT0, DTW_LON0)
    east_x, _ = latlon_to_xy(DTW_LAT0, DTW_LON0 + 0.001)
    _, north_y = latlon_to_xy(DTW_LAT0 + 0.001, DTW_LON0)
    vx = (0.072 / 60.0) * abs(east_x - origin_x) / 0.001
    vy = (0.14 / 60.0) * abs(north_y - origin_y) / 0.001
    max_dx, max_dy, max_dz = vx * dt, vy * dt, VMAX_Z_MPS * dt
    print(f"Units: metres, seconds; separation: {SEP_HOR_M:g} m horizontal, {SEP_VERT_M:g} m vertical")
    print(f"Component speed limits: east={vx:.3f}, north={vy:.3f}, vertical={VMAX_Z_MPS:g} m/s")
    model = Model("debug_si")
    model.Params.NonConvex = 2
    x, y, z, end, selected = [], [], [], [], []
    geometry = [dict() for _ in range(count)]
    objective = 0
    for i in range(count):
        # Bounds follow from entry position, elapsed steps, and speed limits;
        # no arbitrary airspace box is imposed.
        elapsed = [max(0, k - entry_k[i]) for k in range(steps)]
        x.append(model.addVars(steps, lb={k: entry_x[i] - elapsed[k]*max_dx for k in range(steps)},
                               ub={k: entry_x[i] + elapsed[k]*max_dx for k in range(steps)}, name=f"f{i+1}_x_m"))
        y.append(model.addVars(steps, lb={k: entry_y[i] - elapsed[k]*max_dy for k in range(steps)},
                               ub={k: entry_y[i] + elapsed[k]*max_dy for k in range(steps)}, name=f"f{i+1}_y_m"))
        z.append(model.addVars(steps, lb={k: max(0, entry_z[i] - elapsed[k]*max_dz) for k in range(steps)},
                               ub={k: entry_z[i] + elapsed[k]*max_dz for k in range(steps)}, name=f"f{i+1}_z_m"))
        end.append(model.addVars(steps, vtype=GRB.BINARY, name=f"is_end_{i}"))
        selected.append(model.addVars(len(fix_names), vtype=GRB.BINARY, name=f"fix_{i}"))
        model.addConstr(selected[i].sum() == 1)
        model.addConstr(x[i][steps-1] == quicksum(fix_x[f]*selected[i][f] for f in range(len(fix_names))))
        model.addConstr(y[i][steps-1] == quicksum(fix_y[f]*selected[i][f] for f in range(len(fix_names))))
        model.addConstr(z[i][steps-1] >= quicksum(fix_min[f]*selected[i][f] for f in range(len(fix_names))))
        model.addConstr(z[i][steps-1] <= quicksum(fix_max[f]*selected[i][f] for f in range(len(fix_names))))
        for k in range(entry_k[i] + 1):
            model.addConstr(x[i][k] == entry_x[i])
            model.addConstr(y[i][k] == entry_y[i])
            model.addConstr(z[i][k] == entry_z[i])
            # Prevent the unconstrained entry flag from disabling separation.
            model.addConstr(end[i][k] == 0)
        for k in range(entry_k[i]+1, steps):
            changes = []
            for axis, limit, axis_name in ((x, max_dx, "x"), (y, max_dy, "y"), (z, max_dz, "z")):
                change = model.addVar(lb=-limit, ub=limit, name=f"d{axis_name}_{i}_{k}")
                absolute = model.addVar(ub=limit, name=f"abs_{axis_name}_{i}_{k}")
                model.addConstr(change == axis[i][k] - axis[i][k-1])
                model.addGenConstrAbs(absolute, change)
                changes.append(change)
                # All distances are now comparable. Divide by NM2M so the
                # distance objective and CT/CF retain an NM-based cost scale.
                objective += absolute / NM2M
                model.addConstr((end[i][k] == 1) >> (axis[i][k] == axis[i][steps-1]))
            geometry[i][k] = build_segment(model, changes[0], changes[1],
                                            math.hypot(max_dx, max_dy), f"{i}_{k}")
            objective += (CT / CF) * (1 - end[i][k])
    for k in range(steps):
        for i in range(count-1):
            for j in range(i+1, count):
                if k < max(entry_k[i], entry_k[j]):
                    continue
                tag = f"{i}_{j}_{k}"
                branches = model.addVars(4, vtype=GRB.BINARY, name=f"separation_{tag}")
                model.addConstr(branches.sum() + end[i][k] + end[j][k] >= 1)
                for a, b, branch in ((i, j, 0), (j, i, 1)):
                    point = (x[b][k], y[b][k])
                    far = (x[a][k], y[a][k])
                    if k > entry_k[a]:
                        add_capsule(model, geometry[a][k], (x[a][k-1], y[a][k-1]), far,
                                    point, SEP_HOR_M, branches[branch], f"{tag}_{branch}", WAYPOINT_SHAPE)
                    else:
                        add_waypoint(model, far, point, SEP_HOR_M, branches[branch], f"{tag}_{branch}", WAYPOINT_SHAPE)
                model.addConstr((branches[2] == 1) >> (z[i][k] - z[j][k] >= SEP_VERT_M))
                model.addConstr((branches[3] == 1) >> (z[j][k] - z[i][k] >= SEP_VERT_M))
    model.setObjective(objective, GRB.MINIMIZE)
    return model, (flights, epoch, steps, x, y, z)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=base.parent / "2.1.1_Input_Data/entry_exit_points.csv")
    parser.add_argument("--output", type=Path, default=base.parents[1] / "2.3_Outputs_and_Results/test_si.csv")
    parser.add_argument("--flights", type=int, default=2)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--horizon-seconds", type=float, default=48000.0)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--build-only", action="store_true", help="Build and report model statistics without solving")
    args = parser.parse_args()
    if args.flights < 1:
        parser.error("--flights must be positive")
    model, result = build_model(pd.read_csv(args.input).iloc[:args.flights], args.dt, args.horizon_seconds)
    if args.build_only:
        model.update()
        model.printStats()
        return
    if args.time_limit is not None:
        model.Params.TimeLimit = args.time_limit
    model.optimize()
    if not model.SolCount:
        print(f"No solution available; solver status {model.Status}")
        return
    model.printQuality()
    flights, epoch, steps, x, y, z = result
    output = {"t": np.arange(steps)*args.dt}
    for i in range(len(flights)):
        east = np.array([x[i][k].X for k in range(steps)])
        north = np.array([y[i][k].X for k in range(steps)])
        altitude = np.array([z[i][k].X for k in range(steps)])
        lat, lon = xy_to_latlon(east, north)
        output.update({f"f{i+1}_lat": lat, f"f{i+1}_lon": lon, f"f{i+1}_alt_ft": altitude/FT2M,
                       f"f{i+1}_x_m": east, f"f{i+1}_y_m": north, f"f{i+1}_z_m": altitude})
    pd.DataFrame(output).to_csv(args.output, index=False)
    print(f"Results saved to {args.output}; t=0 is {epoch}")


if __name__ == "__main__":
    main()
