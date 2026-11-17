"""Free-location holding constraints and interval-aware reporting."""

import json
from itertools import combinations

import gurobipy as gp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from Functions.fuel_emission_analysis_main_holding import entry_hold_track, POLLUTANTS
from Functions.fuel_emission_analysis_computation import compute_fuel_and_emission


HOLD_ALTITUDE_KNOTS = 401


def add_hold_interval(model, selected, start, finish, fuel_table, speed_mps, duration, name):
    for dimension, (previous, current) in enumerate(zip(start, finish)):
        model.addConstr((selected == 1) >> (current == previous), f'{name}_closed_{dimension}')
    altitudes = np.linspace(fuel_table.axes[1][0], fuel_table.axes[1][-1], HOLD_ALTITUDE_KNOTS)
    flows = fuel_table.exact(speed_mps, altitudes, 0.0)
    flow = model.addVar(lb=float(flows.min()), ub=float(flows.max()), name=f'{name}_flow')
    model.addGenConstrPWL(start[2], flow, altitudes.tolist(), flows.tolist(), name=f'{name}_altitude_pwl')
    fuel = model.addVar(lb=0.0, ub=float(duration * flows.max()), name=f'{name}_kg')
    model.addConstr((selected == 1) >> (fuel == duration * flow), f'{name}_active_fuel')
    model.addConstr((selected == 0) >> (fuel == 0), f'{name}_inactive_fuel')
    return fuel


def racetrack_bounds(speed_mps, duration):
    leg = speed_mps * duration / 4
    radius = leg / np.pi
    return np.array([-radius, 0.0]), np.array([leg + radius, 2 * radius])


def weather_windows(scenario):
    """Per-cell shared-clock windows; older scenarios use one common window."""
    windows = scenario.get("weather_windows_s")
    if windows is None:
        windows = [(scenario["weather_start_s"], scenario["weather_end_s"])] * len(scenario["weather_boxes_uv_m"])
    if len(windows) != len(scenario["weather_boxes_uv_m"]):
        raise ValueError("Each weather region must have exactly one active time window.")
    for window in windows:
        if len(window) != 2 or not np.isfinite(window).all() or window[0] >= window[1]:
            raise ValueError(f"Invalid weather window: {window}; expected finite start < end.")
    return windows


def active_weather_cells(start_s, end_s, scenario):
    """Conservatively protect the whole flight interval for any positive overlap."""
    windows = weather_windows(scenario)
    return [
        (index, bounds)
        for index, (bounds, (start, end)) in enumerate(zip(scenario["weather_boxes_uv_m"], windows))
        if scenario["weather_enabled"] and start_s < end and end_s > start
    ]


def weather_active(start_s, end_s, scenario):
    return bool(active_weather_cells(start_s, end_s, scenario))


def hold_bounds_xy(scenario):
    lower, upper = racetrack_bounds(scenario["hold_speed_mps"], scenario["time_step_seconds"])
    corners = np.array([[lower[0], lower[1]], [lower[0], upper[1]],
                        [upper[0], lower[1]], [upper[0], upper[1]]]) @ np.asarray(scenario["basis"]).T
    return corners.min(axis=0), corners.max(axis=0)


def overlapping_intervals(first, second, scenario):
    dt, n = scenario["time_step_seconds"], scenario["horizon_intervals"]
    for k in range(1, n + 1):
        for j in range(1, n + 1):
            start = max(first["entry_seconds"] + (k - 1) * dt, second["entry_seconds"] + (j - 1) * dt)
            end = min(first["entry_seconds"] + k * dt, second["entry_seconds"] + j * dt)
            if start < end:
                yield k, j, start, end


def position_at(xyz, aircraft, step, time_s, dt):
    fraction = (time_s - aircraft["entry_seconds"] - (step - 1) * dt) / dt
    return [(1 - fraction) * axis[step - 1] + fraction * axis[step] for axis in xyz]


def add_separation(model, scenario, variables):
    """A common separating side over each overlap protects straight legs and hold envelopes."""
    lower, upper = hold_bounds_xy(scenario)
    for a, b in combinations(range(len(variables)), 2):
        first, second = scenario["aircraft"][a], scenario["aircraft"][b]
        xyz_a, arrived_a, hold_a, _, _ = variables[a]
        xyz_b, arrived_b, hold_b, _, _ = variables[b]
        for k, j, start, end in overlapping_intervals(first, second, scenario):
            sides = model.addVars(6, vtype=gp.GRB.BINARY, name=f"separation_{a}_{b}_{k}_{j}")
            model.addConstr(sides.sum() >= 1 - arrived_a[k - 1] - arrived_b[j - 1])
            for time_s in (start, end):
                pa = position_at(xyz_a, first, k, time_s, scenario["time_step_seconds"])
                pb = position_at(xyz_b, second, j, time_s, scenario["time_step_seconds"])
                for axis in range(3):
                    lo, hi = (lower[axis], upper[axis]) if axis < 2 else (0.0, 0.0)
                    clearance = scenario["horizontal_separation_m"] if axis < 2 else scenario["vertical_separation_m"]
                    model.addConstr((sides[2 * axis] == 1) >> (
                        pa[axis] + lo * hold_a[k] - pb[axis] - hi * hold_b[j] >= clearance))
                    model.addConstr((sides[2 * axis + 1] == 1) >> (
                        pb[axis] + lo * hold_b[j] - pa[axis] - hi * hold_a[k] >= clearance))


def seed_geometric_start(model, time_limit, log_path):
    """Seed the copied v16-style model without its named transit fuel constraints."""
    model.update()
    with model.copy() as initializer:
        initializer.Params.TimeLimit = time_limit
        initializer.Params.LogFile = str(log_path)
        initializer.Params.MIPFocus = 1
        initializer.NumStart = 0
        initializer.remove([constraint for constraint in initializer.getConstrs()
                            if constraint.ConstrName.startswith("fuel_")])
        initializer.remove([constraint for constraint in initializer.getGenConstrs()
                            if constraint.GenConstrName.startswith("fuel_")])
        initializer.optimize()
        if initializer.SolCount == 0:
            raise RuntimeError(f"No feasible geometric start; status {initializer.Status}. See {log_path}.")
        model.NumStart = 1
        model.Params.StartNumber = 0
        for variable in model.getVars():
            # Let Gurobi complete the fuel interpolation at the hinted flight states.
            variable.Start = (gp.GRB.UNDEFINED if variable.VarName.startswith("fuel_")
                              else initializer.getVarByName(variable.VarName).X)
        return initializer.Runtime


def add_weather_interval(model, positions, selected, cells, origin, basis, speed_mps, duration, margin, name):
    lower, upper = racetrack_bounds(speed_mps, duration)
    for cell_index, (low_u, low_v, high_u, high_v) in enumerate(cells):
        outside = model.addVars(4, vtype=gp.GRB.BINARY, name=f'{name}_{cell_index}_side')
        model.addConstr(outside.sum() == 1)
        for east, north in positions:
            along = basis[0, 0] * (east - origin[0]) + basis[1, 0] * (north - origin[1])
            across = basis[0, 1] * (east - origin[0]) + basis[1, 1] * (north - origin[1])
            model.addConstr((outside[0] == 1) >> (along + upper[0] * selected <= low_u - margin))
            model.addConstr((outside[1] == 1) >> (along + lower[0] * selected >= high_u + margin))
            model.addConstr((outside[2] == 1) >> (across + upper[1] * selected <= low_v - margin))
            model.addConstr((outside[3] == 1) >> (across + lower[1] * selected >= high_v + margin))


def interior_hold_track(data, endpoint, scenario):
    rows = data.iloc[[endpoint - 1, endpoint, endpoint]].copy().reset_index(drop=True)
    rows.loc[2, 't'] += rows.t.iloc[1] - rows.t.iloc[0]
    rows['f1_status'] = [1, 1, 2]
    rows['f1_hold'] = [0, 1, 0]
    direction = np.asarray(scenario['basis'])[:, 0]
    rows.loc[2, ['f1_x', 'f1_y']] += 1000 * direction
    return entry_hold_track(rows, 1, scenario['hold_speed_mps'])


def report_free_holding(data, aircraft, scenario, output, mass_fraction=0.8):
    origin = np.asarray(scenario['origin_xy_m'])
    basis = np.asarray(scenario['basis'])
    cells = [box(*bounds) for bounds in scenario['weather_boxes_uv_m']]
    tracks, states, checks = [], [], []
    for endpoint in range(1, len(data)):
        if data.f1_status.iloc[endpoint - 1] != 1:
            continue
        previous, current = data.iloc[endpoint - 1], data.iloc[endpoint]
        duration = current.t - previous.t
        holding = bool(round(current.f1_hold))
        if holding:
            if not np.allclose(current[['f1_x', 'f1_y', 'f1_z']].to_numpy(dtype=float),
                               previous[['f1_x', 'f1_y', 'f1_z']].to_numpy(dtype=float), rtol=0, atol=0.01):
                raise ValueError('A selected hold must return to its start position and altitude.')
            track = interior_hold_track(data, endpoint, scenario)
            track['end_step'] = endpoint
            tracks.append(track)
            line_xy = track[['x', 'y']].to_numpy()
            speed = scenario['hold_speed_mps']
        else:
            line_xy = data[['f1_x', 'f1_y']].iloc[endpoint - 1:endpoint + 1].to_numpy()
            speed = np.linalg.norm(line_xy[1] - line_xy[0]) / duration
        active_cells = active_weather_cells(previous.t, current.t, scenario)
        active_weather = bool(active_cells)
        line = LineString((line_xy - origin) @ basis)
        # With no active weather, retain the historical distance-to-footprints diagnostic.
        checked_cells = [cells[index] for index, _ in active_cells] if active_weather else cells
        clearance = min(line.distance(cell) for cell in checked_cells)
        if active_weather and clearance < scenario['clearance_m'] - 0.01:
            raise ValueError(f"Weather clearance failed in interval ending at {current.t}: {clearance}")
        checks.append({'end_t': current.t, 'holding': holding, 'weather_active': active_weather,
                       'active_regions': ','.join(f'W{index + 1}' for index, _ in active_cells),
                       'clearance_m': clearance})
        states.append({'t': current.t, 'd_ts': duration, 'phase': 'hold' if holding else 'route',
                       'groundSpeed': speed / (1852 / 3600),
                       'alt': (previous.f1_z + current.f1_z) / (2 * 0.3048),
                       'rateOfClimb': (current.f1_z - previous.f1_z) / duration / (0.3048 / 60)})
    evaluated = compute_fuel_and_emission(pd.DataFrame(states), aircraft['acType'], mass_fraction)
    evaluated.to_csv(output / 'fuel_emissions_segments.csv', index=False)
    summary = {'acId': aircraft['acId'], 'route_source': 'same_schedule_openap'}
    for phase in ('route', 'hold', 'total'):
        selected = evaluated if phase == 'total' else evaluated[evaluated.phase == phase]
        summary[f'{phase}_seconds'] = float(selected.d_ts.sum())
        summary[f'{phase}_fuel_kg'] = float((selected.fuel_flow * selected.d_ts).sum())
        for pollutant in POLLUTANTS:
            summary[f'{phase}_{pollutant}_g'] = float((selected[f'{pollutant}_flow'] * selected.d_ts).sum())
    result = pd.DataFrame([summary])
    result.to_csv(output / 'fuel_emissions_summary.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'weather_checks.csv', index=False)
    track = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame(columns=['t', 'x', 'y', 'z', 'end_step'])
    track.to_csv(output / 'hold_tracks.csv', index=False)
    scenario = dict(scenario, hold_seconds=summary['hold_seconds'], selected_hold_end_times=data.loc[data.f1_hold > 0.5, 't'].tolist())
    (output / 'free_holding_scenario.json').write_text(json.dumps(scenario, indent=2))

    figure, axis = plt.subplots(figsize=(10, 6))
    windows = weather_windows(scenario)
    for cell_index, (cell, (start, end)) in enumerate(zip(cells, windows)):
        vertices = np.array(cell.exterior.coords) @ basis.T + origin
        axis.add_patch(Polygon(vertices / 1000, facecolor='mistyrose', edgecolor='firebrick', alpha=0.65,
                               label=f"W{cell_index + 1}: {start:g}-{end:g} s" if scenario['weather_enabled']
                               else f"W{cell_index + 1}: disabled"))
    plotted_phases = set()
    for endpoint in range(1, len(data)):
        previous, current = data.iloc[endpoint - 1], data.iloc[endpoint]
        if previous.f1_status != 1 or current.f1_hold:
            continue
        if not scenario['weather_enabled']:
            phase, color = 'MILP route', 'teal'
        elif weather_active(previous.t, current.t, scenario):
            phase, color = 'While weather is active', 'crimson'
        elif current.t <= min(start for start, _ in windows):
            phase, color = 'Before weather forms', 'royalblue'
        elif previous.t >= max(end for _, end in windows):
            phase, color = 'After weather clears', 'teal'
        else:
            phase, color = 'Between weather windows', 'teal'
        axis.plot(data.f1_x.iloc[endpoint-1:endpoint+1] / 1000, data.f1_y.iloc[endpoint-1:endpoint+1] / 1000,
                  'o--', color=color, label=phase if phase not in plotted_phases else None)
        plotted_phases.add(phase)
    for track_index, circuit in enumerate(tracks):
        axis.plot(circuit.x / 1000, circuit.y / 1000, color='darkorange', linewidth=2.5,
                  label='Selected holding racetrack' if track_index == 0 else None)
        anchor = circuit[['x', 'y']].iloc[0].to_numpy() / 1000
        axis.annotate(f"{circuit.t.iloc[0]:g}-{circuit.t.iloc[-1]:g} s hold", anchor,
                      xytext=(-20, 20 + 14 * track_index), textcoords='offset points', color='saddlebrown')
    axis.set(xlabel='East of DTW (km)', ylabel='North of DTW (km)', aspect='equal',
             title=f"{aircraft['acId']} | free-location holding | {summary['hold_seconds']:g} s selected")
    axis.legend(loc='best')
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / 'free_holding_2d.png', dpi=160)
    plt.close(figure)
    return result