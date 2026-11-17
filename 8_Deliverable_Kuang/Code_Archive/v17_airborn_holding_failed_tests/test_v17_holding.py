"""Run: python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_holding.py -v."""

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
from itertools import combinations
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import openap

from Functions.fuel_emission_analysis_main_holding import (
    POLLUTANTS, entry_hold_track, split_entry_hold, write_holding_fuel_report,
)


class HoldingTests(unittest.TestCase):
    def waypoints(self, loops):
        entry = 1
        release = entry + loops
        arrival = release + 2
        steps = np.arange(arrival + 3)
        progress = np.clip(steps - release, 0, 2)
        return pd.DataFrame({
            "t": steps * 400.0,
            "f1_x": progress * 72000.0, "f1_y": steps * 0.0,
            "f1_z": 8000.0 - progress * 1000.0,
            "f1_status": np.where(steps < entry, 0, np.where(steps < arrival, 1, 2)),
            "f1_hold": ((steps > entry) & (steps <= release)).astype(int),
        })

    def test_hold_release_and_phase_accounting(self):
        for loops in (0, 1, 2):
            with self.subTest(loops=loops), tempfile.TemporaryDirectory() as directory:
                data = self.waypoints(loops)
                route, seconds = split_entry_hold(data, 1)
                self.assertEqual(seconds, 400 * loops)
                self.assertEqual(route.t.iloc[0], 400 * (1 + loops))
                self.assertEqual(len(route), 3)
                result = write_holding_fuel_report(data, [{"acId": "TEST", "acType": "B737"}], directory).iloc[0]
                mass = 0.8 * openap.prop.aircraft("B737")["mtow"]
                flow = openap.FuelFlow("B737").enroute(mass=mass, tas=210, alt=8000 / 0.3048, vs=0)
                self.assertAlmostEqual(result.hold_fuel_kg, flow * seconds, places=8)
                self.assertEqual(result.route_seconds, 800)
                for key in ("seconds", "fuel_kg", *(f"{name}_g" for name in POLLUTANTS)):
                    self.assertAlmostEqual(result[f"total_{key}"], result[f"route_{key}"] + result[f"hold_{key}"])
                    if loops == 0:
                        self.assertEqual(result[f"hold_{key}"], 0)
                segments = pd.read_csv(Path(directory) / "fuel_emissions_segments.csv")
                self.assertEqual(segments.loc[segments.phase == 'route', 'd_ts'].sum(), 800)

    def test_racetrack_returns_to_release_anchor(self):
        speed = 210 * 1852 / 3600
        track = entry_hold_track(self.waypoints(2), 1, speed)
        self.assertEqual(track.t.iloc[-1] - track.t.iloc[0], 800)
        np.testing.assert_allclose(track[['x', 'y', 'z']].iloc[0], track[['x', 'y', 'z']].iloc[-1])
        length = np.hypot(track.x.diff(), track.y.diff()).sum()
        self.assertLess(abs(length / (speed * 800) - 1), 0.001)
        self.assertLess(np.hypot(track.x, track.y).max(), speed * 400 / 2)
        self.assertEqual(len(entry_hold_track(self.waypoints(0), 1, speed)), 0)

    def test_reject_interior_holds(self):
        data = self.waypoints(1)
        data.loc[2, 'f1_hold'] = 0
        data.loc[3, 'f1_hold'] = 1
        with self.assertRaisesRegex(ValueError, 'contiguous entry'):
            split_entry_hold(data, 1)

    @unittest.skipUnless(os.environ.get("RUN_HOLDING_INTEGRATION") == "1", "Opt-in Gurobi integration")
    def test_six_flight_forced_hold(self):
        script = Path(__file__).with_name("main_v17_holding_test.py")
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(script), "--force-hold-steps", "1", "--time-limit", "20", "--output-dir", directory],
                capture_output=True, text=True, errors="replace", check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout[-4000:] + completed.stderr[-2000:])
            output = Path(directory)
            stats = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(stats["quadratic_constraints"], 0)
            self.assertIn(stats["status"], ("OPTIMAL", "TIME_LIMIT"))
            data = pd.read_csv(output / "df_wide.csv")
            summary = pd.read_csv(output / "fuel_emissions_summary.csv")
            self.assertEqual(len(summary), 6)
            self.assertEqual(summary.hold_seconds.iloc[0], 400)
            for flight_number in range(1, 7):
                prefix = f"f{flight_number}_"
                flags = data[prefix + 'hold'].to_numpy(dtype=int)
                route, seconds = split_entry_hold(data, flight_number)
                self.assertLessEqual(flags.sum(), 1)
                displacement = route[[prefix + axis for axis in ('x', 'y', 'z')]].diff().iloc[1:].to_numpy()
                speed = np.hypot(displacement[:, 0], displacement[:, 1]) / 400
                self.assertTrue((speed >= 120 * 1852 / 3600 - 1e-4).all())
                self.assertTrue((speed <= 300 + 1e-4).all())
                self.assertTrue((abs(displacement[:, 2]) <= 1000 / 60 * 0.3048 * 400 + 1e-3).all())
                for step in np.flatnonzero(flags):
                    np.testing.assert_allclose(data[[prefix + axis for axis in ('x', 'y', 'z')]].iloc[step],
                                               data[[prefix + axis for axis in ('x', 'y', 'z')]].iloc[step - 1], atol=1e-3)
                    self.assertEqual(data[prefix + 'status'].iloc[step], 1)
                self.assertAlmostEqual(summary.total_seconds.iloc[flight_number - 1], seconds + route.t.iloc[-1] - route.t.iloc[0])
            for step in range(len(data)):
                for first, second in combinations(range(1, 7), 2):
                    statuses = [data[f'f{number}_status'].iloc[step] for number in (first, second)]
                    after_arrival = step > 0 and any(data[f'f{number}_status'].iloc[step - 1] == 2 for number in (first, second))
                    if 0 in statuses or after_arrival:
                        continue
                    holding = sum(data[f'f{number}_hold'].iloc[min(step + 1, len(data) - 1)] for number in (first, second))
                    horizontal = 500 * 0.3048 + holding * (210 * 1852 / 3600) * 400 / 2
                    differences = [abs(data[f'f{first}_{axis}'].iloc[step] - data[f'f{second}_{axis}'].iloc[step]) for axis in ('x', 'y', 'z')]
                    self.assertTrue(differences[0] >= horizontal - 1e-3 or differences[1] >= horizontal - 1e-3 or differences[2] >= 100 * 0.3048 - 1e-3)


if __name__ == "__main__":
    unittest.main()