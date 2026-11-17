"""Focused regression tests; run with python -m unittest discover in this folder."""

from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import pandas as pd

from Functions.fuel_pwl_v1 import FuelFlowTable, write_fuel_report, horizontal_speed_pwl, add_horizontal_speed, NORM_COSINE


class FuelPWLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = FuelFlowTable()
        cls.environment = gp.Env(empty=True)
        cls.environment.setParam("OutputFlag", 0)
        cls.environment.start()

    @classmethod
    def tearDownClass(cls):
        cls.environment.dispose()

    def test_grid_vertices_match_openap(self):
        for point, flow in zip(self.table.points, self.table.flows):
            self.assertAlmostEqual(self.table.interpolate(*point), flow, places=9)
        self.assertTrue((self.table.flows > 0).all())

    def test_held_out_accuracy(self):
        errors = self.table.validation_samples()["relative_error_percent"].abs()
        self.assertLess(errors.quantile(0.95), 5.0)
        self.assertLess(errors.max(), 20.0)

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            self.table.interpolate(350, 6500, 0)
        with self.assertRaises(ValueError):
            self.table.interpolate(180, 6500, 10)

    def test_linear_speed_norm_bound(self):
        for angle in np.linspace(0, 2 * np.pi, 121):
            estimate = horizontal_speed_pwl(180 * np.cos(angle), 180 * np.sin(angle))
            self.assertGreaterEqual(estimate, 180 - 1e-9)
            self.assertLessEqual(estimate, 180 / NORM_COSINE + 1e-9)
        with gp.Model(env=self.environment) as model:
            speed = add_horizontal_speed(model, 80.0, -120.0, "norm_test")
            model.setObjective(speed, gp.GRB.MAXIMIZE)
            model.optimize()
            self.assertAlmostEqual(speed.X, horizontal_speed_pwl(80, -120), places=6)

    def test_local_solver_interpolation_and_inactive_fuel(self):
        for point in [(180.0, 6500.0, -2.5), (105.0, 8800.0, 1.6), (280.0, 3200.0, -4.5)]:
            with self.subTest(point=point), gp.Model(env=self.environment) as model:
                inactive = model.addVar(vtype=gp.GRB.BINARY, lb=0, ub=0)
                speed = model.addVar(lb=point[0], ub=point[0])
                altitude = model.addVar(lb=point[1], ub=point[1])
                vertical = model.addVar(lb=point[2], ub=point[2])
                fuel = self.table.add_segment(model, speed, altitude, vertical, inactive, 400,
                                              "test", (point[1], point[1]), start_point=point)
                model.setObjective(fuel)
                model.optimize()
                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                self.assertAlmostEqual(fuel.X, 400 * self.table.interpolate(*point), places=5)
                inactive.LB = inactive.UB = 1
                speed.LB = speed.UB = 0
                vertical.LB = vertical.UB = 0
                model.optimize()
                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                self.assertAlmostEqual(fuel.X, 0.0, places=8)

    def test_report_counts_arrival_segment_but_not_waiting(self):
        model_speed = horizontal_speed_pwl(180, 0)
        waypoints = pd.DataFrame({
            "t": [0, 400, 800, 1200, 1600],
            "f1_x": [0, 0, 72000, 144000, 144000], "f1_y": [0] * 5,
            "f1_z": [8000, 8000, 7000, 6000, 6000], "f1_status": [0, 1, 1, 2, 2],
            "f1_fuel_speed_mps": [0, 0, model_speed, model_speed, 0],
        })
        flights = pd.DataFrame({"acId": ["TEST"]})
        fuel = {(0, 2): 400 * self.table.interpolate(model_speed, 7500, -2.5),
                (0, 3): 400 * self.table.interpolate(model_speed, 6500, -2.5)}
        with tempfile.TemporaryDirectory() as directory:
            segments, summary = write_fuel_report(self.table, waypoints, flights, fuel,
                                                  Path(directory), show_plot=False)
            self.assertEqual(segments["active"].tolist(), [False, True, True, False])
            self.assertEqual(summary.loc["TEST", "flight_seconds"], 800)
            self.assertAlmostEqual(summary.loc["TEST", "pwl_fuel_kg"], sum(fuel.values()))
            reference = 400 * sum(float(self.table.exact(180, altitude, -2.5)) for altitude in (7500, 6500))
            self.assertAlmostEqual(summary.loc["TEST", "openap_fuel_kg"], reference)
            self.assertTrue((segments.loc[~segments["active"], "pwl_fuel_kg"] == 0).all())
            self.assertEqual(len(pd.read_csv(Path(directory) / "fuel_segments.csv")), 4)

    def test_reference_units_and_same_schedule(self):
        point = (180.0, 6500.0, -2.5)
        reference = self.table.reference.enroute(
            mass=self.table.mass_kg, tas=point[0] * 3600 / 1852,
            alt=point[1] / 0.3048, vs=point[2] * 60 / 0.3048,
        )
        self.assertAlmostEqual(float(self.table.exact(*point)), float(reference), places=12)

    def test_optional_nlp_output_forwarding(self):
        from Functions import fuel_emission_analysis_main as legacy

        waypoints = pd.DataFrame({"t": [0, 400], "f1_x": [0, 1000],
                                  "f1_y": [0, 1000], "f1_z": [7000, 6000]})
        destination = Path("separate_nlp_output")
        with patch.object(legacy, "analyze_optimized_trajectory", return_value="sentinel") as downstream:
            self.assertEqual(legacy.analyze_optimized_trajectory_xyz(waypoints, output_dir=destination), "sentinel")
            self.assertEqual(downstream.call_args.kwargs["output_dir"], destination)
            np.testing.assert_array_equal(downstream.call_args.args[0]["t"], waypoints["t"])

    @unittest.skipUnless(os.environ.get("RUN_V16_INTEGRATION") == "1", "set RUN_V16_INTEGRATION=1 for the six-flight solve")
    def test_six_flight_fuel_and_arrival_separation(self):
        directory = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            completed = subprocess.run(
                [sys.executable, str(directory / "main_v16_fuel_test.py"), "--time-limit", "30",
                 "--no-plots", "--output-dir", str(output)],
                cwd=directory, capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=90,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            run = json.loads((output / "run_summary.json").read_text())
            waypoints = pd.read_csv(output / "df_wide.csv")
            segments = pd.read_csv(output / "fuel_segments.csv")
            self.assertEqual(segments["acId"].nunique(), 6)
            self.assertAlmostEqual(run["objective_terms"]["fuel"], 25 * run["pwl_fuel_kg"], places=4)
            self.assertLess(abs(run["pwl_fuel_kg"] / run["openap_fuel_kg"] - 1), 0.05)
            self.assertTrue((segments.loc[~segments["active"], "pwl_fuel_kg"] == 0).all())
            active_speed = segments.loc[segments["active"], "speed_mps"]
            self.assertTrue(active_speed.between(120 * 1852 / 3600 - 0.01, 300.01).all())
            present = {}
            for flight_number in range(1, 7):
                status = waypoints[f"f{flight_number}_status"]
                present[flight_number] = (status == 1) | ((status == 2) & (status.shift() == 1))
                self.assertEqual(((status == 2) & (status.shift() == 1)).sum(), 1)
            for first, second in combinations(range(1, 7), 2):
                east = abs(waypoints[f"f{first}_x"] - waypoints[f"f{second}_x"])
                north = abs(waypoints[f"f{first}_y"] - waypoints[f"f{second}_y"])
                vertical = abs(waypoints[f"f{first}_z"] - waypoints[f"f{second}_z"])
                violation = (present[first] & present[second] & (east < 152.4 - 0.01)
                             & (north < 152.4 - 0.01) & (vertical < 30.48 - 0.01))
                self.assertFalse(violation.any(), f"Unseparated arrival/active waypoints: {first}, {second}")


if __name__ == "__main__":
    unittest.main()