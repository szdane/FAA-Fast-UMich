"""Interval-aware fuel reporting and geometry for the single-flight interior-hold test."""

import json

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from Functions.fuel_emission_analysis_main_holding import entry_hold_track, POLLUTANTS
from Functions.fuel_emission_analysis_computation import compute_fuel_and_emission


def interior_hold_track(data, endpoint, scenario):
    rows = data.iloc[[endpoint - 1, endpoint, endpoint]].copy().reset_index(drop=True)
    rows.loc[2, 't'] += rows.t.iloc[1] - rows.t.iloc[0]
    rows['f1_status'] = [1, 1, 2]
    rows['f1_hold'] = [0, 1, 0]
    direction = np.asarray(scenario['basis'])[:, 0]
    rows.loc[2, ['f1_x', 'f1_y']] += 1000 * direction
    return entry_hold_track(rows, 1, scenario['hold_speed_mps'])


def report_interior_holding(data, aircraft, scenario, output, mass_fraction=0.8):
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
            track = interior_hold_track(data, endpoint, scenario)
            tracks.append(track)
            line_xy = track[['x', 'y']].to_numpy()
            speed = scenario['hold_speed_mps']
        else:
            line_xy = data[['f1_x', 'f1_y']].iloc[endpoint - 1:endpoint + 1].to_numpy()
            speed = np.linalg.norm(line_xy[1] - line_xy[0]) / duration
        active_weather = scenario['weather_enabled'] and previous.t < scenario['weather_end_s'] and current.t > scenario['weather_start_s']
        clearance = min(LineString((line_xy - origin) @ basis).distance(cell) for cell in cells)
        if active_weather and clearance < scenario['clearance_m'] - 0.01:
            raise ValueError(f"Weather clearance failed in interval ending at {current.t}: {clearance}")
        checks.append({'end_t': current.t, 'holding': holding, 'weather_active': active_weather, 'clearance_m': clearance})
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
    track = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame(columns=['t', 'x', 'y', 'z'])
    track.to_csv(output / 'interior_hold_track.csv', index=False)
    scenario = dict(scenario, hold_seconds=summary['hold_seconds'], selected_hold_end_times=data.loc[data.f1_hold > 0.5, 't'].tolist())
    (output / 'interior_scenario.json').write_text(json.dumps(scenario, indent=2))

    figure, axis = plt.subplots(figsize=(10, 6))
    for cell_index, cell in enumerate(cells):
        vertices = np.array(cell.exterior.coords) @ basis.T + origin
        axis.add_patch(Polygon(vertices / 1000, facecolor='mistyrose', edgecolor='firebrick', alpha=0.65,
                               label=('Weather cells (400-800 s only)' if scenario['weather_enabled'] else 'Weather footprint (disabled)') if cell_index == 0 else None))
    active = pd.concat([data[data.f1_status == 1], data[data.f1_status == 2].head(1)])
    if tracks:
        before = active[active.t <= scenario['weather_start_s']]
        after = active[active.t >= scenario['weather_end_s']]
        axis.plot(before.f1_x / 1000, before.f1_y / 1000, 'o--', color='royalblue', label='Before weather forms')
        axis.plot(after.f1_x / 1000, after.f1_y / 1000, 'o--', color='teal', label='After weather clears')
    else:
        axis.plot(active.f1_x / 1000, active.f1_y / 1000, 'o--', color='teal', label='MILP route')
    if not track.empty:
        axis.plot(track.x / 1000, track.y / 1000, color='darkorange', linewidth=2.5, label='Interior holding racetrack')
        anchor = np.asarray(scenario['hold_anchor_xy_m']) / 1000
        axis.annotate(f"{track.t.iloc[0]:g}-{track.t.iloc[-1]:g} s hold", anchor, xytext=(-20, 20), textcoords='offset points', color='saddlebrown')
    axis.set(xlabel='East of DTW (km)', ylabel='North of DTW (km)', aspect='equal',
             title=f"{aircraft['acId']} | synthetic weather | holding not forced")
    axis.legend(loc='best')
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / 'interior_holding_2d.png', dpi=160)
    plt.close(figure)
    return result