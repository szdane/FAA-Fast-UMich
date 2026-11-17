"""Analyze saved holding/weather MILP results without rerunning Gurobi."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.path import Path as MarkerPath
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from Functions.fuel_emission_analysis_main_weather_nlp import (
    analyze_optimized_trajectory_xyz, split_entry_hold,
)


def plot_trajectories(data, aircraft_list, results, weather, output_path):
    figure, axis = plt.subplots(figsize=(10, 6))
    right_angles = np.linspace(-np.pi / 2, np.pi / 2, 24)
    left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 24)
    marker_points = np.column_stack((
        np.r_[0.8 + np.cos(right_angles), -0.8 + np.cos(left_angles)],
        np.r_[np.sin(right_angles), np.sin(left_angles)],
    ))
    racetrack = MarkerPath(np.vstack([marker_points, marker_points[0]]))
    if weather is not None and weather['weather_enabled']:
        lower = np.asarray(weather['weather_min_xy_m']) / 1000
        width, height = (np.asarray(weather['weather_max_xy_m']) / 1000 - lower)
        axis.add_patch(Rectangle(lower, width, height, facecolor='mistyrose', edgecolor='firebrick', label='Synthetic weather'))
    hold_labels = []
    for number, aircraft in enumerate(aircraft_list, start=1):
        flight_id = aircraft['acId']
        route, hold_seconds = split_entry_hold(data, number)
        prefix = f'f{number}_'
        axis.plot(route[prefix + 'x'] / 1000, route[prefix + 'y'] / 1000, 'o--',
                  color='gray', linewidth=1, markersize=4, label=f'{flight_id.split("_")[0]} MILP')
        if flight_id in results['optimized']:
            trajectory = results['optimized'][flight_id]
            axis.plot(trajectory.x / 1000, trajectory.y / 1000, color='teal', linewidth=2, label='NLP route')
        if hold_seconds > 0:
            anchor = route[[prefix + 'x', prefix + 'y']].iloc[0].to_numpy() / 1000
            axis.scatter(*anchor, marker=racetrack, s=850, facecolors='none', edgecolors='darkorange',
                         linewidths=2, zorder=6, label='Holding (schematic symbol)')
            axis.annotate(f'{hold_seconds:g} s hold', anchor, xytext=(8, 17), textcoords='offset points', color='saddlebrown')
            hold_labels.append(f'{hold_seconds:g} s holding')
        else:
            hold_labels.append('no holding')
    status = 'NLP generated' if len(results['optimized']) == len(aircraft_list) else 'NLP unavailable; MILP shown'
    axis.set(xlabel='East of DTW (km)', ylabel='North of DTW (km)', aspect='equal',
             title=f'{status} | {", ".join(hold_labels)}')
    axis.margins(x=0.12, y=0.22)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).parent / 'Output/v17_weather_test/bad_weather')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--nlp-time-limit', type=float, default=30.0)
    args = parser.parse_args()
    if args.nlp_time_limit <= 0:
        parser.error('nlp-time-limit must be positive.')
    output = args.output_dir or args.input_dir / ('trajectory_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input_dir / 'df_wide.csv')
    holding = pd.read_csv(args.input_dir / 'holding_summary.csv')
    aircraft_list = [{'acId': flight_id, 'acType': 'B737'} for flight_id in holding.acId]
    weather_path = args.input_dir / 'weather_scenario.json'
    weather = json.loads(weather_path.read_text()) if weather_path.exists() else None
    results = analyze_optimized_trajectory_xyz(
        data, aircraft_list, output_dir=output, weather=weather,
        hold_speed_mps=float(holding.hold_speed_mps.iloc[0]), nlp_time_limit=args.nlp_time_limit,
    )
    plot_trajectories(data, aircraft_list, results, weather, output / 'trajectories_2d.png')
    print(f'Results saved to {output}', flush=True)
    if len(results['optimized']) != len(aircraft_list):
        raise SystemExit('NLP failed for one or more flights; labeled same-schedule fallback and plot saved.')


if __name__ == '__main__':
    main()