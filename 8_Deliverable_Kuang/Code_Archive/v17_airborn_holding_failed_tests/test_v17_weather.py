"""Integration test: two bounded MILP solves, with holding optional in both."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from shapely.geometry import LineString, box


class WeatherScenarioTests(unittest.TestCase):
    def test_weather_requires_rerouting_without_forcing_holding(self):
        script = Path(__file__).with_name("main_v17_weather_test.py")
        scenarios = {}
        with tempfile.TemporaryDirectory() as directory:
            for name, options in (("clear", ["--clear-weather"]), ("blocked", [])):
                output = Path(directory) / name
                completed = subprocess.run(
                    [sys.executable, str(script), "--time-limit", "20", "--output-dir", str(output), *options],
                    capture_output=True, text=True, errors="replace", check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-2000:])
                report = json.loads((output / "weather_scenario.json").read_text())
                solver = json.loads((output / "run_summary.json").read_text())
                self.assertEqual(solver["quadratic_constraints"], 0)
                self.assertFalse(report["holding_forced"])
                self.assertEqual(report["hold_seconds"], 0)
                self.assertTrue(report["direct_route_blocked"])
                data = pd.read_csv(output / "df_wide.csv")
                active = pd.concat([data[data.f1_status == 1], data[data.f1_status == 2].head(1)])
                obstacle = box(*report["weather_min_xy_m"], *report["weather_max_xy_m"])
                route = LineString(active[['f1_x', 'f1_y']].to_numpy())
                self.assertEqual(route.intersects(obstacle), name == "clear")
                if name == "blocked":
                    self.assertGreaterEqual(route.distance(obstacle), report["clearance_margin_m"] - 0.01)
                self.assertEqual(int(data.f1_hold.sum()), 0)
                fuel = pd.read_csv(output / "fuel_emissions_summary.csv").iloc[0]
                self.assertEqual(fuel.hold_fuel_kg, 0)
                self.assertEqual(fuel.route_source, "same_schedule_openap")
                scenarios[name] = report
            self.assertEqual(scenarios['clear']['fixed_star'], scenarios['blocked']['fixed_star'])
            self.assertGreater(scenarios['blocked']['route_distance_km'], scenarios['clear']['route_distance_km'])


if __name__ == "__main__":
    unittest.main()