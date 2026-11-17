"""Free holding geometry/fuel checks and opt-in full-route feasibility tests."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import gurobipy as gp
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from Functions.free_holding import HOLD_ALTITUDE_KNOTS, add_hold_interval, add_weather_interval
from Functions.fuel_pwl import FuelFlowTable


class FreeHoldingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = FuelFlowTable()
        cls.speed = 210 * 1852 / 3600

    def test_holding_at_different_positions_and_altitudes(self):
        for anchor in [(-210000, 110000, 6400.8), (-140000, 80000, 7178.04), (20000, -50000, 8229.6)]:
            for holding in (0, 1):
                with self.subTest(anchor=anchor, holding=holding), gp.Model() as model:
                    model.Params.OutputFlag = 0
                    model.Params.NonConvex = 0
                    start = [model.addVar(lb=value, ub=value) for value in anchor]
                    finish_values = np.asarray(anchor) + (0 if holding else 100)
                    finish = [model.addVar(lb=value, ub=value) for value in finish_values]
                    selected = model.addVar(vtype=gp.GRB.BINARY, lb=holding, ub=holding)
                    fuel = add_hold_interval(model, selected, start, finish, self.table, self.speed, 400, 'hold')
                    model.optimize()
                    self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                    self.assertEqual(model.NumQConstrs, 0)
                    expected = 400 * float(self.table.exact(self.speed, anchor[2], 0)) * holding
                    self.assertAlmostEqual(fuel.X, expected, delta=max(0.01, expected * 0.005))

    def test_hold_cannot_change_position_or_altitude(self):
        for dimension in range(3):
            with self.subTest(dimension=dimension), gp.Model() as model:
                model.Params.OutputFlag = 0
                anchor = np.array([1000., 2000., 6000.])
                finish_values = anchor.copy()
                finish_values[dimension] += 100
                start = [model.addVar(lb=value, ub=value) for value in anchor]
                finish = [model.addVar(lb=value, ub=value) for value in finish_values]
                selected = model.addVar(vtype=gp.GRB.BINARY, lb=1, ub=1)
                add_hold_interval(model, selected, start, finish, self.table, self.speed, 400, 'hold')
                model.optimize()
                self.assertEqual(model.Status, gp.GRB.INFEASIBLE)

    def test_altitude_interpolation_accuracy(self):
        knots = np.linspace(0, self.table.axes[1][-1], HOLD_ALTITUDE_KNOTS)
        samples = np.linspace(0, self.table.axes[1][-1], 4001)
        exact = self.table.exact(self.speed, samples, 0)
        approximation = np.interp(samples, knots, self.table.exact(self.speed, knots, 0))
        self.assertLess(float(np.max(np.abs(approximation - exact) / exact)), 0.005)

    def test_weather_checks_full_circuit_and_full_transit(self):
        cases = [
            ((0, 0), (0, 0), 1, [0, 500, 20000, 2000], gp.GRB.INFEASIBLE),
            ((0, 0), (0, 0), 0, [0, 500, 20000, 2000], gp.GRB.OPTIMAL),
            ((-30000, 0), (-30000, 0), 1, [0, 500, 20000, 2000], gp.GRB.OPTIMAL),
            ((-5000, 0), (25000, 0), 0, [0, -100, 20000, 100], gp.GRB.INFEASIBLE),
        ]
        basis = np.array([[0., -1.], [1., 0.]])
        origin = np.array([12345., -67890.])
        for start_uv, finish_uv, holding, cell, expected in cases:
            with self.subTest(start=start_uv, holding=holding), gp.Model() as model:
                model.Params.OutputFlag = 0
                positions = [[model.addVar(lb=value, ub=value) for value in np.asarray(point) @ basis.T + origin]
                             for point in (start_uv, finish_uv)]
                selected = model.addVar(vtype=gp.GRB.BINARY, lb=holding, ub=holding)
                add_weather_interval(model, positions, selected, [cell], origin, basis, self.speed, 400, 100, 'weather')
                model.optimize()
                self.assertEqual(model.Status, expected)

    def check_output(self, output):
        data = pd.read_csv(output / 'df_wide.csv')
        scenario = json.loads((output / 'free_holding_scenario.json').read_text())
        solver = json.loads((output / 'run_summary.json').read_text())
        intervals = pd.read_csv(output / 'holding_intervals.csv')
        tracks = pd.read_csv(output / 'hold_tracks.csv')
        fuel = pd.read_csv(output / 'fuel_emissions_summary.csv').iloc[0]
        segments = pd.read_csv(output / 'fuel_emissions_segments.csv')
        self.assertTrue(scenario['hold_location_free'])
        self.assertTrue(scenario['hold_altitude_free'])
        self.assertTrue(scenario['entry_holding_allowed'])
        self.assertFalse(scenario['holding_forced'])
        self.assertEqual(solver['quadratic_constraints'], 0)
        self.assertLessEqual(int(data.f1_hold.sum()), scenario['max_hold_steps'])
        self.assertEqual(len(intervals), int(data.f1_hold.sum()))
        self.assertEqual(fuel.hold_seconds, 400 * len(intervals))
        self.assertAlmostEqual(fuel.hold_fuel_kg, intervals.hold_fuel_kg.sum(), places=5)
        self.assertAlmostEqual(fuel.total_fuel_kg, fuel.route_fuel_kg + fuel.hold_fuel_kg)
        self.assertEqual(fuel.total_seconds, segments.d_ts.sum())
        self.assertEqual(len(segments), int((data.f1_status == 1).sum()))
        origin, basis = np.asarray(scenario['origin_xy_m']), np.asarray(scenario['basis'])
        cells = [box(*bounds) for bounds in scenario['weather_boxes_uv_m']]
        for endpoint in range(1, len(data)):
            previous, current = data.iloc[endpoint - 1], data.iloc[endpoint]
            if current.f1_hold:
                self.assertEqual(previous.f1_status, 1)
                self.assertEqual(current.f1_status, 1)
                np.testing.assert_allclose(previous[['f1_x', 'f1_y', 'f1_z']].to_numpy(dtype=float),
                                           current[['f1_x', 'f1_y', 'f1_z']].to_numpy(dtype=float), atol=0.01, rtol=0)
                circuit = tracks[tracks.end_step == endpoint]
                self.assertEqual((circuit.t.iloc[0], circuit.t.iloc[-1]), (previous.t, current.t))
                np.testing.assert_allclose(circuit[['x', 'y', 'z']].iloc[0], circuit[['x', 'y', 'z']].iloc[-1], atol=0.01, rtol=0)
                self.assertAlmostEqual(LineString(circuit[['x', 'y']]).length, self.speed * 400, delta=30)
                line_xy = circuit[['x', 'y']].to_numpy()
            else:
                line_xy = data[['f1_x', 'f1_y']].iloc[endpoint - 1:endpoint + 1].to_numpy()
            if previous.f1_status == 1 and scenario['weather_enabled'] and previous.t < scenario['weather_end_s'] and current.t > scenario['weather_start_s']:
                line = LineString((line_xy - origin) @ basis)
                self.assertGreaterEqual(min(line.distance(cell) for cell in cells), scenario['clearance_m'] - 0.01)
        return data, solver

    @unittest.skipUnless(os.environ.get('RUN_FREE_HOLDING_INTEGRATION') == '1', 'Three optional-choice solves, 30 seconds each.')
    def test_optional_weather_and_clear_controls(self):
        script = Path(__file__).with_name('main_v17_free_holding_test.py')
        with tempfile.TemporaryDirectory() as directory:
            for name, options in [('weather', []), ('clear', ['--clear-weather']), ('no_hold', ['--max-hold-steps', '0'])]:
                output = Path(directory) / name
                completed = subprocess.run([sys.executable, str(script), '--time-limit', '30', '--output-dir', str(output), *options],
                                           capture_output=True, text=True, errors='replace')
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-2000:])
                data, solver = self.check_output(output)
                self.assertEqual(int(data.f1_hold.sum()), int(name == 'weather'))

    @unittest.skipUnless(os.environ.get('RUN_FREE_HOLDING_INTEGRATION') == '1', 'Full-model schedule feasibility, 15 seconds per fixture.')
    def test_entry_late_consecutive_and_separate_holds(self):
        script = Path(__file__).with_name('main_v17_free_holding_test.py')
        fixture = '''
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
import gurobipy as gp
import numpy as np
script, output, pattern_text = sys.argv[1:]
pattern = {int(value) for value in pattern_text.split(',')}
sys.path.insert(0, str(Path(script).parent))
sys.argv = [script, '--clear-weather', '--time-limit', '15', '--max-hold-steps', str(len(pattern)), '--output-dir', output]
original_optimize = gp.Model.optimize
def check_schedule(model, callback=None):
    model.update()
    coordinates = [[model.getVarByName(f'f1_{dimension}[{step}]') for dimension in ('x', 'y', 'z')] for step in range(7)]
    start = np.array([variable.Start for variable in coordinates[0]])
    finish = np.array([variable.Start for variable in coordinates[-1]])
    route_steps = 6 - len(pattern)
    progress = 0
    for step in range(7):
        holding = int(step in pattern)
        selected = model.getVarByName(f'hold_interval[0,{step}]')
        selected.LB = selected.UB = selected.Start = holding
        if step and not holding:
            progress += 1
        point = start + progress / route_steps * (finish - start)
        for variable, value in zip(coordinates[step], point):
            variable.Start = value
        model.getVarByName(f'is_end[0,{step}]').Start = int(step == 6)
        model.getVarByName(f'delta_end[0,{step}]').Start = int(step == 6)
    model.getVarByName('exit_k[0]').Start = 6
    model.Params.SolutionLimit = 1
    return original_optimize(model, callback)
with patch.object(gp.Model, 'optimize', check_schedule):
    runpy.run_path(script, run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as directory:
            for pattern in [(1,), (4,), (2, 3), (2, 4)]:
                with self.subTest(pattern=pattern):
                    output = Path(directory) / '_'.join(map(str, pattern))
                    completed = subprocess.run([sys.executable, '-c', fixture, str(script), str(output), ','.join(map(str, pattern))],
                                               capture_output=True, text=True, errors='replace')
                    self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-2000:])
                    data, solver = self.check_output(output)
                    self.assertEqual(tuple(data.index[data.f1_hold > 0.5]), pattern)


if __name__ == '__main__':
    unittest.main()