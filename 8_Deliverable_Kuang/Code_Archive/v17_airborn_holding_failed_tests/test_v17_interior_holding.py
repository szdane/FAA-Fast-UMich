"""Interior-hold reporting checks and opt-in, time-limited weather controls."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from Functions.interior_holding import interior_hold_track, report_interior_holding


class InteriorHoldingTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            't': [0, 400, 800, 1200, 1600, 2000],
            'f1_x': [0., 50000., 50000., 100000., 150000., 150000.],
            'f1_y': [0.] * 6, 'f1_z': [8200., 6400., 6400., 4800., 3350., 3350.],
            'f1_status': [1, 1, 1, 1, 2, 2], 'f1_hold': [0, 0, 1, 0, 0, 0],
        })
        self.scenario = {
            'origin_xy_m': [0., 0.], 'basis': [[1., 0.], [0., 1.]],
            'hold_anchor_xy_m': [50000., 0.], 'hold_speed_mps': 210 * 1852 / 3600,
            'weather_enabled': True, 'weather_start_s': 400, 'weather_end_s': 800,
            'clearance_m': 1000., 'weather_boxes_uv_m': [[20000, -70000, 44000, 70000],
                [67000, -70000, 160000, 70000], [44000, -70000, 67000, -3000],
                [44000, 11000, 67000, 70000]],
        }

    def test_track_is_closed_level_and_interior(self):
        track = interior_hold_track(self.data, 2, self.scenario)
        np.testing.assert_allclose(track[['x', 'y', 'z']].iloc[0], [50000, 0, 6400])
        np.testing.assert_allclose(track[['x', 'y', 'z']].iloc[-1], track[['x', 'y', 'z']].iloc[0])
        self.assertEqual((track.t.iloc[0], track.t.iloc[-1]), (400, 800))
        self.assertTrue((track.z == 6400).all())
        self.assertAlmostEqual(LineString(track[['x', 'y']]).length,
                               self.scenario['hold_speed_mps'] * 400, delta=30)

    def test_report_counts_prehold_and_posthold_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = report_interior_holding(self.data, {'acId': 'test', 'acType': 'B737'}, self.scenario, output).iloc[0]
            segments = pd.read_csv(output / 'fuel_emissions_segments.csv')
            self.assertEqual(segments.phase.tolist(), ['route', 'hold', 'route', 'route'])
            self.assertEqual(segments.t.tolist(), [400, 800, 1200, 1600])
            self.assertEqual((report.route_seconds, report.hold_seconds, report.total_seconds), (1200, 400, 1600))
            self.assertAlmostEqual(report.total_fuel_kg, report.route_fuel_kg + report.hold_fuel_kg)
            self.assertGreater(report.hold_fuel_kg, 0)

    def test_reconstructed_hold_weather_collision_is_rejected(self):
        self.scenario['weather_boxes_uv_m'] = [[49000, -1000, 51000, 1000]]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, 'Weather clearance failed'):
            report_interior_holding(self.data, {'acId': 'test', 'acType': 'B737'}, self.scenario, Path(directory))

    @unittest.skipUnless(os.environ.get('RUN_INTERIOR_INTEGRATION') == '1', 'Opt-in: three MILP solves capped at 30 seconds each.')
    def test_weather_selects_optional_interior_hold_with_feasible_controls(self):
        script = Path(__file__).with_name('main_v17_interior_holding_test.py')
        costs = {}
        with tempfile.TemporaryDirectory() as directory:
            for name, options in [('weather', []), ('clear', ['--clear-weather']), ('no_hold', ['--max-hold-steps', '0'])]:
                output = Path(directory) / name
                process = subprocess.run([sys.executable, str(script), '--time-limit', '30', '--output-dir', str(output), *options],
                                         capture_output=True, text=True, errors='replace')
                self.assertEqual(process.returncode, 0, process.stdout[-3000:] + process.stderr[-2000:])
                data = pd.read_csv(output / 'df_wide.csv')
                scenario = json.loads((output / 'interior_scenario.json').read_text())
                solver = json.loads((output / 'run_summary.json').read_text())
                self.assertFalse(scenario['holding_forced'])
                self.assertEqual(solver['quadratic_constraints'], 0)
                self.assertEqual(int(data.f1_hold.sum()), int(name == 'weather'))
                costs[name] = solver['objective']
                origin, basis = np.asarray(scenario['origin_xy_m']), np.asarray(scenario['basis'])
                cells = [box(*bounds) for bounds in scenario['weather_boxes_uv_m']]
                for endpoint in range(1, len(data)):
                    previous, current = data.iloc[endpoint - 1], data.iloc[endpoint]
                    if previous.f1_status != 1:
                        continue
                    if current.f1_hold:
                        self.assertGreater(endpoint, 1)
                        self.assertEqual(current.f1_status, 1)
                        self.assertGreater(data.f1_status.eq(2).idxmax(), endpoint)
                        track = pd.read_csv(output / 'interior_hold_track.csv')
                        line_xy = track[['x', 'y']].to_numpy()
                        self.assertEqual((track.t.iloc[0], track.t.iloc[-1]), (previous.t, current.t))
                        np.testing.assert_allclose(data[['f1_x', 'f1_y', 'f1_z']].iloc[endpoint],
                                                   data[['f1_x', 'f1_y', 'f1_z']].iloc[endpoint - 1])
                    else:
                        line_xy = data[['f1_x', 'f1_y']].iloc[endpoint - 1:endpoint + 1].to_numpy()
                    if scenario['weather_enabled'] and previous.t < scenario['weather_end_s'] and current.t > scenario['weather_start_s']:
                        route = LineString((line_xy - origin) @ basis)
                        self.assertGreaterEqual(min(route.distance(cell) for cell in cells), scenario['clearance_m'] - 0.01)
                fuel = pd.read_csv(output / 'fuel_emissions_summary.csv').iloc[0]
                holding = pd.read_csv(output / 'holding_summary.csv').iloc[0]
                self.assertAlmostEqual(fuel.hold_fuel_kg, holding.hold_fuel_kg, places=5)
                self.assertAlmostEqual(fuel.total_fuel_kg, fuel.route_fuel_kg + fuel.hold_fuel_kg)
            self.assertLess(costs['clear'], costs['weather'])
            self.assertLess(costs['weather'], costs['no_hold'])


if __name__ == '__main__':
    unittest.main()