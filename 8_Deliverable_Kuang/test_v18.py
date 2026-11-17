"""Regression checks for the paired v18 experiment (Gurobi license required)."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

import gurobipy as gp
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

from main_v18 import (
    FuelFlowTable, add_separation, build_model, load_scenario, plot_comparison,
    report_aircraft, run_comparison, validate_separation, weather_active,
)


class V18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = FuelFlowTable()
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_shared_weather_and_exact_entry_offsets(self):
        single = load_scenario()
        multi = load_scenario(3, 1)
        storms = load_scenario(3, 3)
        self.assertEqual(single["weather_boxes_uv_m"], multi["weather_boxes_uv_m"])
        self.assertEqual(multi["aircraft"], storms["aircraft"])
        self.assertEqual([flight["entry_seconds"] for flight in multi["aircraft"]], [0, 485, 166])
        self.assertEqual(len(storms["weather_boxes_uv_m"]), 3)
        self.assertEqual((multi["weather_start_s"], multi["weather_end_s"]), (400, 800))
        self.assertFalse(weather_active(0, 400, multi))
        self.assertFalse(weather_active(800, 1200, multi))
        self.assertTrue(weather_active(400, 800, multi))
        self.assertTrue(weather_active(166, 566, multi))
        self.assertTrue(weather_active(485, 885, multi))
        with self.assertRaises(ValueError):
            load_scenario(1, 3)

    def test_only_holding_budget_changes_between_models(self):
        scenario = load_scenario()
        off, _ = build_model(scenario, self.table, False, 1, self.env)
        on, _ = build_model(scenario, self.table, True, 1, self.env)
        with off, on:
            self.assertEqual(off.NumQConstrs, 0)
            self.assertEqual(off.NumVars, on.NumVars)
            self.assertEqual(off.NumGenConstrs, on.NumGenConstrs)
            self.assertEqual(
                [(v.VarName, v.LB, v.UB, v.Obj, v.VType) for v in off.getVars()],
                [(v.VarName, v.LB, v.UB, v.Obj, v.VType) for v in on.getVars()])
            changed = []
            for a, b in zip(off.getConstrs(), on.getConstrs()):
                self.assertEqual(a.ConstrName, b.ConstrName)
                self.assertEqual(a.Sense, b.Sense)
                self.assertEqual(str(off.getRow(a)), str(on.getRow(b)))
                if a.RHS != b.RHS:
                    changed.append((a.ConstrName, a.RHS, b.RHS))
            self.assertEqual(changed, [("f0_holding_budget", 0, 6)])

    def test_holding_factor_changes_only_objective_costs(self):
        baseline = load_scenario()
        discounted = load_scenario(holding_cost_factor=0.25)
        self.assertEqual({key: value for key, value in baseline.items() if key != "holding_cost_factor"},
                         {key: value for key, value in discounted.items() if key != "holding_cost_factor"})
        original, _ = build_model(baseline, self.table, True, 1, self.env)
        reduced, _ = build_model(discounted, self.table, True, 1, self.env)
        with original, reduced:
            self.assertEqual(original.NumVars, reduced.NumVars)
            self.assertEqual(original.NumGenConstrs, reduced.NumGenConstrs)
            for a, b in zip(original.getConstrs(), reduced.getConstrs()):
                self.assertEqual((a.ConstrName, a.RHS, a.Sense), (b.ConstrName, b.RHS, b.Sense))
                self.assertEqual(str(original.getRow(a)), str(reduced.getRow(b)))
            changed = []
            weights, dt = baseline["objective_weights"], baseline["time_step_seconds"]
            for a, b in zip(original.getVars(), reduced.getVars()):
                self.assertEqual((a.VarName, a.LB, a.UB, a.VType), (b.VarName, b.LB, b.UB, b.VType))
                if a.Obj != b.Obj:
                    changed.append(a.VarName)
                    if a.VarName.startswith("f0_hold["):
                        self.assertAlmostEqual(b.Obj - a.Obj, -0.75 * dt * (
                            weights["time"] + weights["distance"] * baseline["hold_speed_mps"]))
                    else:
                        self.assertRegex(a.VarName, r"^f0_hold_[1-6]_kg$")
                        self.assertAlmostEqual(b.Obj, 0.25 * a.Obj)
            self.assertEqual(len(changed), 12)
        for factor in (-1, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                load_scenario(holding_cost_factor=factor)

    def test_entry_late_consecutive_and_separated_holds(self):
        scenario = load_scenario()
        scenario["weather_enabled"] = False
        for steps in ((1,), (2,), (3,), (4,), (5,), (1, 2), (1, 3)):
            with self.subTest(steps=steps):
                model, variables = build_model(
                    scenario, self.table, True, 10, self.env, kinematic_only=True)
                with model:
                    model.Params.OutputFlag = 0
                    model.Params.SolutionLimit = 1
                    hold = variables[0][2]
                    for k in range(7):
                        model.addConstr(hold[k] == int(k in steps))
                    model.optimize()
                    self.assertGreater(model.SolCount, 0, f"status={model.Status}")
                    xyz = variables[0][0]
                    for k in steps:
                        for axis in xyz:
                            self.assertAlmostEqual(axis[k].X, axis[k - 1].X, delta=0.01)
                    if steps == (1, 3):
                        with tempfile.TemporaryDirectory() as directory:
                            output = Path(directory)
                            destination = output / "holding_on" / scenario["aircraft"][0]["acId"]
                            destination.mkdir(parents=True)
                            report, data = report_aircraft(
                                scenario, scenario["aircraft"][0], variables[0], self.table, destination)
                            self.assertEqual(report["hold_seconds"], 800)
                            self.assertGreater(report["hold_fuel_kg"], 0)
                            tracks = pd.read_csv(destination / "hold_tracks.csv")
                            self.assertEqual(set(tracks.end_step), {1, 3})
                            for _, track in tracks.groupby("end_step"):
                                np.testing.assert_allclose(
                                    track[["x", "y", "z"]].iloc[0], track[["x", "y", "z"]].iloc[-1], atol=0.01)
                                self.assertAlmostEqual(LineString(track[["x", "y"]]).length,
                                                       scenario["hold_speed_mps"] * 400, delta=30)
                            plot_comparison(scenario, {"holding_on": [data]}, output)
                            self.assertGreater((output / "comparison.png").stat().st_size, 10000)

    def test_continuous_separation_rejects_crossing_segments(self):
        scenario = load_scenario(3, 1)
        scenario["aircraft"] = deepcopy(scenario["aircraft"][:2])
        scenario["horizon_intervals"] = 1
        for aircraft in scenario["aircraft"]:
            aircraft["entry_seconds"] = 0
        for altitude_difference, expected in ((0, False), (1000, True)):
            with self.subTest(altitude_difference=altitude_difference), gp.Model(env=self.env) as model:
                schedules, variables = [], []
                for i, endpoints in enumerate(((-1000, 1000), (1000, -1000))):
                    data = pd.DataFrame({"t": [0, 400], "f1_x": endpoints, "f1_y": [0, 0],
                                         "f1_z": [6000 + i * altitude_difference] * 2,
                                         "f1_status": [1, 2], "f1_hold": [0, 0]})
                    schedules.append(data)
                    xyz = []
                    for column in ("f1_x", "f1_y", "f1_z"):
                        xyz.append({k: model.addVar(lb=float(value), ub=float(value))
                                    for k, value in enumerate(data[column])})
                    arrived = {k: model.addVar(lb=k, ub=k) for k in range(2)}
                    hold = {k: model.addVar(lb=0, ub=0) for k in range(2)}
                    variables.append((xyz, arrived, hold, {}, {}))
                add_separation(model, scenario, variables)
                model.optimize()
                self.assertEqual(model.SolCount > 0, expected)
                with tempfile.TemporaryDirectory() as directory:
                    if expected:
                        validate_separation(scenario, schedules, Path(directory))
                    else:
                        with self.assertRaisesRegex(ValueError, "Separation check failed"):
                            validate_separation(scenario, schedules, Path(directory))

    def check_comparison(self, output):
        scenario = json.loads((output / "scenario.json").read_text())
        comparison = pd.read_csv(output / "comparison.csv")
        self.assertEqual(comparison.case.tolist(), ["holding_off", "holding_on"])
        self.assertEqual(comparison.hold_seconds.iloc[0], 0)
        self.assertLessEqual(comparison.objective.iloc[1], comparison.objective.iloc[0] + 0.01)
        self.assertGreater((output / "comparison.png").stat().st_size, 10000)
        self.assertEqual(len(pd.read_csv(output / "aircraft_comparison.csv")), 2 * len(scenario["aircraft"]))
        for case in comparison.case:
            schedules = []
            for aircraft in scenario["aircraft"]:
                folder = output / case / aircraft["acId"]
                data = pd.read_csv(folder / "df_wide.csv")
                schedules.append(data)
                self.assertEqual(data.t.iloc[0], aircraft["entry_seconds"])
                intervals = pd.read_csv(folder / "holding_intervals.csv")
                tracks = pd.read_csv(folder / "hold_tracks.csv")
                fuel = pd.read_csv(folder / "fuel_emissions_summary.csv").iloc[0]
                self.assertEqual(len(intervals), int(data.f1_hold.sum()))
                self.assertEqual(fuel.hold_seconds, 400 * len(intervals))
                self.assertAlmostEqual(fuel.total_fuel_kg, fuel.route_fuel_kg + fuel.hold_fuel_kg)
                self.assertAlmostEqual(fuel.hold_fuel_kg, intervals.hold_fuel_kg.sum(), places=5)
                self.assertEqual(fuel.total_seconds, 400 * int((data.f1_status == 1).sum()))
                for k in range(1, len(data)):
                    if data.f1_status.iloc[k - 1] != 1:
                        self.assertEqual(data.f1_hold.iloc[k], 0)
                        continue
                    if data.f1_hold.iloc[k]:
                        self.assertEqual(data.f1_status.iloc[k], 1)
                        np.testing.assert_allclose(
                            data.loc[k, ["f1_x", "f1_y", "f1_z"]].to_numpy(dtype=float),
                            data.loc[k - 1, ["f1_x", "f1_y", "f1_z"]].to_numpy(dtype=float), atol=0.01)
                        xy = tracks.loc[tracks.end_step == k, ["x", "y"]].to_numpy()
                    else:
                        xy = data[["f1_x", "f1_y"]].iloc[k - 1:k + 1].to_numpy()
                        self.assertGreater(np.linalg.norm(xy[1] - xy[0]), 0)
                    windows = scenario.get("weather_windows_s", [
                        [scenario["weather_start_s"], scenario["weather_end_s"]]
                    ] * len(scenario["weather_boxes_uv_m"]))
                    line = LineString((xy - np.asarray(scenario["origin_xy_m"])) @ np.asarray(scenario["basis"]))
                    for bounds, (start, end) in zip(scenario["weather_boxes_uv_m"], windows):
                        if scenario["weather_enabled"] and data.t.iloc[k - 1] < end and data.t.iloc[k] > start:
                            self.assertGreaterEqual(line.distance(box(*bounds)), scenario["clearance_m"] - 0.01)
            validate_separation(scenario, schedules, output / case)

    @unittest.skipUnless(os.environ.get("RUN_V18_INTEGRATION") == "1", "Six solves, plus geometric initialization.")
    def test_paired_scenario_ladder(self):
        with tempfile.TemporaryDirectory() as directory:
            for flights, regions in ((1, 1), (3, 1), (3, 3)):
                with self.subTest(flights=flights, regions=regions):
                    output = Path(directory) / f"{flights}_{regions}"
                    run_comparison(30, output, flights, regions)
                    self.check_comparison(output)

    @unittest.skipUnless(os.environ.get("V18_RESULTS_ROOT"), "Validate previously saved paired runs.")
    def test_saved_scenario_ladder(self):
        root = Path(os.environ["V18_RESULTS_ROOT"])
        for name in ("single", "three_one", "three_three"):
            with self.subTest(name=name):
                self.check_comparison(root / name)

    @unittest.skipUnless(os.environ.get("V18_PENALTY_RESULTS_ROOT"), "Validate a saved holding-penalty sweep.")
    def test_saved_penalty_sweep(self):
        root = Path(os.environ["V18_PENALTY_RESULTS_ROOT"])
        summary = json.loads((root / "penalty_sweep.json").read_text())
        baseline = summary["baseline"]
        reference = json.loads((root / "scenario.json").read_text())
        for result in summary["results"]:
            with self.subTest(factor=result["holding_cost_factor"]):
                folder = root / result["output_directory"]
                scenario = json.loads((folder / "scenario.json").read_text())
                self.assertEqual({key: value for key, value in reference.items() if key != "holding_cost_factor"},
                                 {key: value for key, value in scenario.items() if key != "holding_cost_factor"})
                aircraft_folder = folder / scenario["aircraft"][0]["acId"]
                segments = pd.read_csv(aircraft_folder / "fuel_emissions_segments.csv")
                fuel = self.table.exact(
                    segments.groundSpeed.to_numpy() * 1852 / 3600,
                    segments.alt.to_numpy() * 0.3048,
                    segments.rateOfClimb.to_numpy() * 0.3048 / 60)
                self.assertAlmostEqual(float(np.sum(fuel * segments.d_ts)), result["total_fuel_kg"], delta=0.01)
                intervals = pd.read_csv(aircraft_folder / "holding_intervals.csv")
                self.assertEqual(len(intervals), result["hold_intervals"])
                self.assertEqual(len(intervals) * 400, result["hold_seconds"])
                self.assertAlmostEqual(intervals.hold_fuel_kg.sum(), result["hold_fuel_kg"], delta=0.01)
                weights = scenario["objective_weights"]
                holding_cost = (weights["time"] * result["hold_seconds"]
                                + weights["distance"] * scenario["hold_speed_mps"] * result["hold_seconds"]
                                + weights["fuel"] * intervals.hold_pwl_fuel_kg.sum())
                self.assertAlmostEqual(holding_cost, result["holding_objective_before_discount"], delta=0.01)
                self.assertAlmostEqual(result["objective_at_full_holding_cost"],
                                       result["objective"] + (1 - result["holding_cost_factor"]) * holding_cost)
                self.assertLessEqual(result["objective"], baseline["objective"] + 0.01)
                self.assertEqual(result["holding_beats_all_no_hold_solutions"],
                                 bool(result["hold_intervals"] > 0 and
                                      result["objective"] < baseline["objective_bound"] - 0.01))
                checks = pd.read_csv(aircraft_folder / "weather_checks.csv")
                self.assertTrue((checks.loc[checks.weather_active, "clearance_m"] >= scenario["clearance_m"] - 0.01).all())
                data = pd.read_csv(aircraft_folder / "df_wide.csv")
                tracks = pd.read_csv(aircraft_folder / "hold_tracks.csv")
                for step in intervals.end_step:
                    np.testing.assert_allclose(
                        data.loc[step - 1, ["f1_x", "f1_y", "f1_z"]].to_numpy(dtype=float),
                        data.loc[step, ["f1_x", "f1_y", "f1_z"]].to_numpy(dtype=float), atol=0.01)
                    circuit = tracks.loc[tracks.end_step == step]
                    self.assertAlmostEqual(LineString(circuit[["x", "y"]]).length,
                                           scenario["hold_speed_mps"] * 400, delta=30)
                    if weather_active(data.t.iloc[step - 1], data.t.iloc[step], scenario):
                        uv = (circuit[["x", "y"]].to_numpy() - np.asarray(scenario["origin_xy_m"])) @ np.asarray(scenario["basis"])
                        self.assertGreaterEqual(min(LineString(uv).distance(box(*bounds))
                                                    for bounds in scenario["weather_boxes_uv_m"]),
                                                scenario["clearance_m"] - 0.01)
        self.assertGreater((root / "penalty_sweep.png").stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
