"""Paired airborne-holding experiment: identical weather, holding OFF versus ON."""

import argparse
import json
import math
import os
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
from Functions.free_holding import (
    add_hold_interval, add_weather_interval, add_separation, weather_active, active_weather_cells,
)
from Functions.holding_output import (
    report_aircraft, validate_separation, plot_comparison, write_case_output, write_comparison_output,
)
from Functions.fuel_pwl_v1 import FuelFlowTable, NORM_COSINE, add_horizontal_speed
from Functions.utilities import load_flights_utc_xyz, load_star_fixes_xyz


SCRIPT_DIR = Path(__file__).resolve().parent
FLIGHT_ID = "DAL1120_KMSNtoKDTW"
FLIGHT_IDS = [FLIGHT_ID, "DAL1655_KPHLtoKDTW", "EDV5018_CYULtoKDTW"]
DT = 400.0
N_INTERVALS = 6
FT2M = 0.3048
HOLD_SPEED = 210.0 * 1852 / 3600
VERTICAL_LIMIT = 1000 / 60 * FT2M
MIN_SPEED = 120 * 1852 / 3600
HEADING_LIMIT = 0.25 * 300 * DT
SPEED_CHANGE_LIMIT = 0.5 * (300 - MIN_SPEED)
WEIGHTS = {"time": 1.0, "distance": 0.1, "fuel": 25.0,
           "smoothness": 5e-7, "vertical": 0.1, "heading": 2e-3}
SEP_HORIZONTAL = 500 * FT2M
SEP_VERTICAL = 100 * FT2M


def load_scenario(flight_count=1, weather_regions=1, holding_cost_factor=1.0):
    """Load the flight once; both models receive this same scenario."""
    if not math.isfinite(holding_cost_factor) or holding_cost_factor < 0:
        raise ValueError("holding_cost_factor must be finite and nonnegative.")
    source = SCRIPT_DIR / "Input" / "entry_exit_points.csv"
    entries = pd.read_csv(source)
    if flight_count not in (1, 3) or weather_regions not in (1, 3):
        raise ValueError("flight_count and weather_regions must each be 1 or 3.")
    ids = FLIGHT_IDS[:flight_count]
    selected = entries.loc[entries.acId.isin(ids)]
    if len(selected) != flight_count or selected.acId.nunique() != flight_count:
        raise ValueError(f"Expected exactly one entry record for each of {ids}.")
    epoch = pd.to_datetime(selected.entry_rectime).min()
    flights, _ = load_flights_utc_xyz(source, ids, DT, 42.2125, -83.3534, grid_epoch=epoch)
    fixes, xs, ys, upper, lower = load_star_fixes_xyz(
        SCRIPT_DIR / "Input" / "star_fixes.csv", 42.2125, -83.3534)
    aircraft = []
    for ac_id in ids:
        flight = flights.loc[flights.acId == ac_id].iloc[0]
        start = flight[["entry_x", "entry_y", "entry_z"]].to_numpy(dtype=float)
        chosen = int(np.argmin(np.hypot(np.asarray(xs) - start[0], np.asarray(ys) - start[1])))
        aircraft.append({
            "acId": ac_id, "acType": "B737", "entry_seconds": float(flight.entry_epoch_s),
            "entry_xyz_m": start.tolist(), "star_fix": list(fixes)[chosen],
            "star_xy_m": [float(xs[chosen]), float(ys[chosen])],
            "star_altitude_bounds_m": [float(lower[chosen]), float(upper[chosen])],
        })
    first = aircraft[0]
    start, finish = np.asarray(first["entry_xyz_m"]), np.asarray(first["star_xy_m"])
    length = float(np.linalg.norm(finish - start[:2]))
    if length <= 0:
        raise ValueError("The entry and STAR fix must be distinct.")
    along = (finish - start[:2]) / length
    basis = np.column_stack([along, [-along[1], along[0]]])
    weather_boxes = [[0.25 * length, -70000.0, 0.85 * length, 70000.0]]
    for flight in aircraft[1:weather_regions]:
        entry = np.asarray(flight["entry_xyz_m"][:2])
        target = np.asarray(flight["star_xy_m"])
        center = (entry + 0.5 * (target - entry) - start[:2]) @ basis
        weather_boxes.append([float(center[0] - 25000), float(center[1] - 20000),
                              float(center[0] + 25000), float(center[1] + 20000)])
    if len(weather_boxes) != weather_regions:
        raise ValueError("Three weather regions require the three-aircraft scenario.")
    return {
        "aircraft": aircraft, "acType": "B737", "mass_fraction": 0.8,
        "entry_utc": epoch.isoformat(),
        "time_step_seconds": DT, "horizon_intervals": N_INTERVALS,
        "weather_enabled": True, "weather_start_s": 400.0, "weather_end_s": 800.0,
        "weather_boxes_uv_m": weather_boxes,
        "origin_xy_m": start[:2].tolist(), "basis": basis.tolist(),
        "clearance_m": 1000.0, "hold_speed_mps": HOLD_SPEED,
        "hold_location_free": True, "hold_altitude_free": True,
        "entry_holding_allowed": True, "holding_forced": False,
        "objective_weights": dict(WEIGHTS),
        "holding_cost_factor": holding_cost_factor,
        "horizontal_separation_m": SEP_HORIZONTAL, "vertical_separation_m": SEP_VERTICAL,
        "timing": "Shared seconds after first entry; each flight retains its exact entry offset.",
    }


def add_aircraft(model, scenario, aircraft, fuel_table, holding_enabled, index,
                 kinematic_only=False, hints=None):
    counts = (model.NumVars, model.NumConstrs, model.NumGenConstrs)
    n = scenario["horizon_intervals"]
    dt = scenario["time_step_seconds"]
    start = aircraft["entry_xyz_m"]
    finish = aircraft["star_xy_m"]
    low, high = aircraft["star_altitude_bounds_m"]
    weights = scenario["objective_weights"]
    holding_factor = scenario["holding_cost_factor"]
    nodes = range(n + 1)
    xyz = [model.addVars(nodes, lb=-gp.GRB.INFINITY, name=axis) for axis in ("x", "y")]
    xyz.append(model.addVars(nodes, lb=0, ub=float(fuel_table.axes[1][-1]), name="z"))
    arrived = model.addVars(nodes, vtype=gp.GRB.BINARY, name="arrived")
    hold = model.addVars(nodes, vtype=gp.GRB.BINARY, name="hold")
    model.addConstr(arrived[0] == 0)
    model.addConstr(arrived[n] == 1)
    model.addConstr(hold[0] == 0)
    model.addConstr(hold.sum() <= (n if holding_enabled else 0), name="holding_budget")
    for axis in range(3):
        model.addConstr(xyz[axis][0] == start[axis])
    for axis in range(2):
        model.addConstr(xyz[axis][n] == finish[axis])
    model.addConstr(xyz[2][n] >= low)
    model.addConstr(xyz[2][n] <= high)

    route_fuel, hold_fuel, speeds, displacements = {}, {}, {}, {}
    objective = gp.LinExpr()
    for k in range(1, n + 1):
        model.addConstr(arrived[k] >= arrived[k - 1])
        model.addConstr(hold[k] + arrived[k] <= 1, name=f"hold_airborne_{k}")
        for axis in range(3):
            model.addConstr((arrived[k] == 1) >> (xyz[axis][k] == xyz[axis][n]))
        inactive = model.addVar(vtype=gp.GRB.BINARY, name=f"route_inactive_{k}")
        model.addConstr(inactive == arrived[k - 1] + hold[k])
        delta = model.addVars(3, lb=-gp.GRB.INFINITY, name=f"delta_{k}")
        absolute = model.addVars(3, name=f"absolute_{k}")
        for axis in range(3):
            model.addConstr(delta[axis] == xyz[axis][k] - xyz[axis][k - 1])
            model.addConstr(absolute[axis] == gp.abs_(delta[axis]))
        displacements[k] = delta
        model.addConstr(absolute[2] <= VERTICAL_LIMIT * dt)
        speed = add_horizontal_speed(model, delta[0] / dt, delta[1] / dt, f"speed_{k}")
        speeds[k] = speed
        model.addConstr(speed >= MIN_SPEED / NORM_COSINE * (1 - inactive))
        model.addConstr(absolute[2] <= math.tan(math.radians(5)) * speed * dt)
        hold_fuel[k] = add_hold_interval(
            model, hold[k], [axis[k - 1] for axis in xyz], [axis[k] for axis in xyz],
            fuel_table, scenario["hold_speed_mps"], dt, f"hold_{k}")
        reachable = (k - 0.5) * VERTICAL_LIMIT * dt
        altitude_bounds = (max(0.0, start[2] - reachable),
                           min(float(fuel_table.axes[1][-1]), start[2] + reachable))
        if kinematic_only:
            route_fuel[k] = model.addVar(lb=0, ub=0, name=f"route_fuel_{k}_kg")
        else:
            start_point = None
            if hints is not None and hints[f"f{index}_route_inactive_{k}"] < 0.5:
                start_point = (
                    hints[f"f{index}_speed_{k}_mps"],
                    (hints[f"f{index}_z[{k - 1}]"] + hints[f"f{index}_z[{k}]"]) / 2,
                    hints[f"f{index}_delta_{k}[2]"] / dt)
            route_fuel[k] = fuel_table.add_segment(
                model, speed, (xyz[2][k - 1] + xyz[2][k]) / 2, delta[2] / dt,
                inactive, dt, f"route_fuel_{k}", altitude_bounds, start_point=start_point)
        objective += weights["time"] * dt * (1 - arrived[k - 1] - (1 - holding_factor) * hold[k])
        if not kinematic_only:
            objective += weights["fuel"] * (route_fuel[k] + holding_factor * hold_fuel[k])
        objective += weights["distance"] * dt * (speed + holding_factor * scenario["hold_speed_mps"] * hold[k])
        objective += weights["smoothness"] * (
            absolute[0] + absolute[1] + weights["vertical"] * absolute[2])
        if k >= 2:
            transition = model.addVar(vtype=gp.GRB.BINARY, name=f"hold_transition_{k}")
            model.addGenConstrOr(transition, [hold[k - 1], hold[k]])
            changes = model.addVars(2, lb=-gp.GRB.INFINITY, name=f"heading_change_{k}")
            abs_changes = model.addVars(2, name=f"abs_heading_change_{k}")
            heading_cost = model.addVar(name=f"heading_cost_{k}")
            for axis in range(2):
                model.addConstr(changes[axis] == delta[axis] - displacements[k - 1][axis])
                model.addConstr(abs_changes[axis] == gp.abs_(changes[axis]))
                model.addConstr((transition == 0) >> (abs_changes[axis] <= HEADING_LIMIT))
            model.addConstr((transition == 0) >> (heading_cost == abs_changes.sum()))
            model.addConstr((transition == 1) >> (heading_cost == 0))
            objective += weights["heading"] * heading_cost
            bypass = model.addVar(vtype=gp.GRB.BINARY, name=f"speed_bypass_{k}")
            model.addGenConstrOr(bypass, [transition, arrived[k]])
            model.addConstr((bypass == 0) >> (speed - speeds[k - 1] <= SPEED_CHANGE_LIMIT))
            model.addConstr((bypass == 0) >> (speeds[k - 1] - speed <= SPEED_CHANGE_LIMIT))
        offset = aircraft["entry_seconds"]
        for region, bounds in active_weather_cells(offset + (k - 1) * dt, offset + k * dt, scenario):
            add_weather_interval(
                model, [(xyz[0][j], xyz[1][j]) for j in (k - 1, k)], hold[k],
                [bounds], np.asarray(scenario["origin_xy_m"]),
                np.asarray(scenario["basis"]), scenario["hold_speed_mps"], dt,
                scenario["clearance_m"], f"weather_{k}_w{region + 1}")
    model.update()
    for variable in model.getVars()[counts[0]:]:
        variable.VarName = f"f{index}_{variable.VarName}"
    for constraint in model.getConstrs()[counts[1]:]:
        constraint.ConstrName = f"f{index}_{constraint.ConstrName}"
    for constraint in model.getGenConstrs()[counts[2]:]:
        constraint.GenConstrName = f"f{index}_{constraint.GenConstrName}"
    model.update()
    return (xyz, arrived, hold, route_fuel, hold_fuel), objective


def build_model(scenario, fuel_table, holding_enabled, time_limit, env,
                kinematic_only=False, hints=None):
    """Only the per-aircraft holding budgets differ between the two joint models."""
    model = gp.Model("v18_airborne_holding", env=env)
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.01
    model.Params.Threads = min(4, os.cpu_count() or 1)
    model.Params.Seed = 0
    model.Params.NonConvex = 0
    variables, objective = [], gp.LinExpr()
    for index, aircraft in enumerate(scenario["aircraft"]):
        flight_vars, cost = add_aircraft(
            model, scenario, aircraft, fuel_table, holding_enabled, index, kinematic_only, hints)
        variables.append(flight_vars)
        objective += cost
    add_separation(model, scenario, variables)
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.update()
    if hints is not None:
        for variable in model.getVars():
            if variable.VarName in hints and "_route_fuel_" not in variable.VarName:
                variable.Start = hints[variable.VarName]
    return model, variables


def solve_case(scenario, fuel_table, holding_enabled, time_limit, output, env, warm_start=None):
    output.mkdir()
    initializer, _ = build_model(
        scenario, fuel_table, holding_enabled, min(time_limit, 30.0), env, kinematic_only=True)
    with initializer:
        initializer.Params.LogFile = str(output / "initialization.log")
        initializer.optimize()
        if initializer.SolCount == 0:
            raise RuntimeError(f"No feasible geometric start for {output.name}; status {initializer.Status}.")
        hints = {variable.VarName: variable.X for variable in initializer.getVars()}
        initialization_seconds = initializer.Runtime
    model, variables = build_model(
        scenario, fuel_table, holding_enabled, time_limit, env, hints=hints)
    with model:
        if warm_start is not None:
            model.NumStart = 2
            model.Params.StartNumber = 1
            for variable in model.getVars():
                variable.Start = warm_start[variable.VarName]
            model.Params.StartNumber = 0
        model.Params.LogFile = str(output / "solver.log")
        model.optimize()
        result, schedules = write_case_output(
            model, scenario, variables, fuel_table, output, holding_enabled, initialization_seconds,
            ["geometric_incumbent"] + (["holding_off_incumbent"] if warm_start is not None else []))
        incumbent = {variable.VarName: variable.X for variable in model.getVars()}
    return result, schedules, incumbent


def run_comparison(time_limit=1200.0, output_dir=None, flight_count=1, weather_regions=1,
                   holding_cost_factor=1.0):
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("The per-case time limit must be finite and positive.")
    scenario = load_scenario(flight_count, weather_regions, holding_cost_factor)
    output = Path(output_dir) if output_dir is not None else (
        SCRIPT_DIR / "Output" / "v18_holding_comparison" / pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    table = FuelFlowTable(scenario["acType"], scenario["mass_fraction"])
    results, schedules = [], {}
    warm_start = None
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        for holding_enabled in (False, True):
            name = "holding_on" if holding_enabled else "holding_off"
            print(f"\n=== {name.upper()} | same storm: 400-800 seconds after entry ===", flush=True)
            result, data, warm_start = solve_case(
                scenario, table, holding_enabled, time_limit, output / name, env, warm_start)
            results.append(result)
            schedules[name] = data
    return write_comparison_output(scenario, results, schedules, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-limit", type=float, default=60.0, help="Gurobi seconds per case (default: 60).")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing directories are never overwritten.")
    parser.add_argument("--flight-count", type=int, choices=(1, 3), default=1)
    parser.add_argument("--weather-regions", type=int, choices=(1, 3), default=1)
    parser.add_argument("--holding-cost-factor", type=float, default=1.0,
                        help="Multiplier for holding time/distance/fuel objective costs only (default: 1).")
    args = parser.parse_args()
    if not math.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be finite and positive.")
    if args.flight_count == 1 and args.weather_regions == 3:
        parser.error("--weather-regions 3 requires --flight-count 3.")
    if not math.isfinite(args.holding_cost_factor) or args.holding_cost_factor < 0:
        parser.error("--holding-cost-factor must be finite and nonnegative.")
    run_comparison(args.time_limit, args.output_dir, args.flight_count, args.weather_regions, args.holding_cost_factor)


if __name__ == "__main__":
    main()
