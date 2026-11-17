"""Checks for the minimally extended procedural model and shared output contract."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import pandas as pd

from Functions.holding_output import _plot_weather_timed_path, animation_track, write_case_output
from Functions.free_holding import active_weather_cells, report_free_holding, weather_windows
import test_v18


SCRIPT = Path(__file__).with_name("main_v18.1.py")
BUILD_SOURCE, BUILD_MARKER, _ = SCRIPT.read_text(encoding="utf-8").partition("# 2.6. Initiate optimization")
if not BUILD_MARKER:
    raise RuntimeError("The model-building boundary is missing; do not execute full solves in unit fixtures.")


@contextmanager
def built_model(*options):
    with tempfile.TemporaryDirectory() as directory:
        namespace = {"__file__": str(SCRIPT), "__name__": "v181_fixture"}
        argv = [str(SCRIPT), "--output-dir", str(Path(directory) / "run"), *options]
        try:
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                exec(compile(BUILD_SOURCE, str(SCRIPT), "exec"), namespace)
            namespace["m"].setObjective(namespace["obj"], gp.GRB.MINIMIZE)
            namespace["m"].update()
            yield namespace
        finally:
            if "m" in namespace:
                namespace["m"].dispose()
            if "env" in namespace:
                namespace["env"].dispose()


def fix_route(namespace, holding):
    model = namespace["m"]
    aircraft = namespace["scenario"]["aircraft"][0]
    start = np.asarray(aircraft["entry_xyz_m"])
    finish = np.r_[aircraft["star_xy_m"], np.mean(aircraft["star_altitude_bounds_m"])]
    fractions = [0, 0, 0.2, 0.4, 0.6, 0.8, 1] if holding else [0, 0.2, 0.4, 0.55, 0.7, 0.85, 1]
    namespace["hold_budgets"][0].RHS = 6
    for k, fraction in enumerate(fractions):
        point = start + fraction * (finish - start)
        for axis, value in zip(("f_x", "f_y", "f_z"), point):
            model.addConstr(namespace[axis][0][k] == float(value))
        model.addConstr(namespace["hold"][0, k] == int(holding and k == 1))
        model.addConstr(namespace["fix_reached"][0, k] == int(k == 6))
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = 15


class V181Tests(unittest.TestCase):
    def test_plot_lines_follow_only_corresponding_weather(self):
        from matplotlib import pyplot as plt
        for window in ((400, 800), (800, 1200), (300, 700)):
            with self.subTest(window=window):
                figure, axis = plt.subplots()
                try:
                    times = np.array([166, 566, 966, 1366], dtype=float)
                    _plot_weather_timed_path(axis, times, np.column_stack([times, times * 2]),
                                             window, True, "C0", "flight (W1", set())
                    solid = [line for line in axis.lines if line.get_linestyle() == "-"]
                    self.assertEqual(len(solid), 1)
                    np.testing.assert_allclose(solid[0].get_xdata()[[0, -1]], window)
                    for line in axis.lines:
                        np.testing.assert_allclose(line.get_ydata(), 2 * line.get_xdata())
                        if line.get_linestyle() == "--":
                            self.assertTrue(line.get_xdata()[-1] <= window[0]
                                            or line.get_xdata()[0] >= window[1])
                    axis.clear()
                    _plot_weather_timed_path(axis, times, np.column_stack([times, times]),
                                             window, False, "C0", "flight (W1", set())
                    self.assertTrue(all(line.get_linestyle() == "--" for line in axis.lines))
                    axis.clear()
                    _plot_weather_timed_path(axis, np.array([0., 200., 400.]),
                                             np.array([[0, 0], [1, 1], [0, 0]]),
                                             (400, 800), True, "C0", "flight HOLD (W1", set(), linewidth=3)
                    self.assertEqual(axis.lines[0].get_linestyle(), "--")
                    self.assertEqual(axis.lines[0].get_linewidth(), 3)
                finally:
                    plt.close(figure)

    def test_animation_uses_circuit_samples_and_stops_at_arrival(self):
        data = pd.DataFrame({
            "t": [166, 566, 966, 1366], "f1_x": [0, 0, 10000, 10000],
            "f1_y": [0, 0, 0, 0], "f1_status": [1, 1, 2, 2], "f1_hold": [0, 1, 0, 0]})
        circuit = pd.DataFrame({
            "t": [166, 266, 366, 466, 566], "x": [0, 1000, 1000, 0, 0],
            "y": [0, 0, 1000, 1000, 0], "end_step": [1] * 5})
        track, holds = animation_track(data, circuit)
        self.assertEqual(holds, [(166, 566)])
        np.testing.assert_allclose(track[:5], circuit[["t", "x", "y"]])
        np.testing.assert_allclose(track[-1], [966, 10000, 0])
        self.assertEqual(np.interp(366, track[:, 0], track[:, 1]), 1000)
        self.assertEqual(np.interp(766, track[:, 0], track[:, 1]), 5000)
        with self.assertRaisesRegex(ValueError, "holding track"):
            animation_track(data, circuit.iloc[:0])
        with self.assertRaisesRegex(ValueError, "recorded STAR arrival"):
            animation_track(data.assign(f1_status=1), circuit)

    def test_preserves_copied_parameters_and_star_choices(self):
        with built_model("--flight-count", "1") as ns:
            model = ns["m"]
            self.assertEqual(model.Params.TimeLimit, 1200)
            self.assertEqual(model.Params.MIPGap, 0.01)
            self.assertEqual(model.NumQConstrs, 0)
            self.assertGreater(ns["n_fixes"], 1)
            self.assertEqual(len(ns["fix_sel"][0]), ns["n_fixes"])
            self.assertEqual(model.getConstrByName("one_fix1").RHS, 1)
            self.assertTrue(all(variable.LB == 0 and variable.UB == 1 for variable in ns["fix_sel"][0].values()))
            self.assertTrue(all(variable.Obj == 0 for variable in ns["u_x"][0].values()))
            self.assertEqual(ns["scenario"]["objective_weights"]["smoothness"], 0)
            self.assertEqual(ns["hold_budgets"][0].RHS, 0)
            self.assertEqual(ns["N_steps"], 7)
            row = model.getRow(model.getConstrByName("fpa_up_0_1"))
            coefficient = next(row.getCoeff(j) for j in range(row.size())
                               if row.getVar(j).sameAs(ns["speed_2d"][0, 1]))
            self.assertAlmostEqual(coefficient, -ns["GAMMA_MAX_TAN"] * ns["NORM_COSINE"])

    def test_exact_offsets_and_multiple_weather_regions(self):
        with built_model() as ns:
            scenario = ns["scenario"]
            self.assertEqual(ns["args"].flight_count, 3)
            self.assertEqual(ns["args"].weather_regions, 3)
            self.assertEqual(ns["args"].holding_cost_factor, 1)
            self.assertEqual([flight["entry_seconds"] for flight in scenario["aircraft"]], [0, 485, 166])
            self.assertEqual(len(scenario["weather_boxes_uv_m"]), 3)
            self.assertEqual(scenario["weather_windows_s"], [(400, 800), (800, 1200), (1000, 1500)])
            self.assertEqual((scenario["weather_start_s"], scenario["weather_end_s"]), (400, 1500))
            self.assertTrue((ns["flights"].flight_entry_timestep == 0).all())
            self.assertTrue(any(v.VarName.startswith("separation_") for v in ns["m"].getVars()))
            for i, flight in enumerate(scenario["aircraft"]):
                for k in range(1, ns["N_steps"]):
                    interval_start = flight["entry_seconds"] + (k - 1) * 400
                    interval_end = interval_start + 400
                    for region, (start, end) in enumerate(scenario["weather_windows_s"]):
                        variable = ns["m"].getVarByName(f"weather_{i}_{k}_w{region + 1}_0_side[0]")
                        self.assertEqual(variable is not None, interval_start < end and interval_end > start)

    def test_w3_earlier_window_and_w1_size_preserve_other_inputs(self):
        with built_model("--holding-cost-factor", "0.5") as ns:
            original = ns["scenario"]
        with built_model("--holding-cost-factor", "0.5", "--w3-window", "300", "700",
                         "--w3-match-w1-size") as ns:
            scenario = ns["scenario"]
            changed_keys = {"weather_windows_s", "weather_start_s", "weather_end_s", "weather_boxes_uv_m"}
            self.assertEqual({k: v for k, v in scenario.items() if k not in changed_keys},
                             {k: v for k, v in original.items() if k not in changed_keys})
            self.assertEqual(scenario["weather_windows_s"], [(400, 800), (800, 1200), (300, 700)])
            self.assertEqual((scenario["weather_start_s"], scenario["weather_end_s"]), (300, 1200))
            self.assertEqual(scenario["weather_boxes_uv_m"][:2], original["weather_boxes_uv_m"][:2])
            w1, w3 = np.asarray(scenario["weather_boxes_uv_m"])[[0, 2]]
            old_w3 = np.asarray(original["weather_boxes_uv_m"][2])
            np.testing.assert_allclose(w3[2:] - w3[:2], w1[2:] - w1[:2])
            np.testing.assert_allclose((w3[:2] + w3[2:]) / 2, (old_w3[:2] + old_w3[2:]) / 2)
            for i, flight in enumerate(scenario["aircraft"]):
                for k in range(1, ns["N_steps"]):
                    start = flight["entry_seconds"] + (k - 1) * 400
                    var = ns["m"].getVarByName(f"weather_{i}_{k}_w3_0_side[0]")
                    self.assertEqual(var is not None, start < 700 and start + 400 > 300)
            self.assertEqual(ns["m"].Params.TimeLimit, 1200)

    def test_invalid_w3_options_are_rejected(self):
        for options in (("--w3-window", "700", "300"), ("--w3-window", "nan", "700"),
                        ("--w3-window", "-1", "700"),
                        ("--flight-count", "1", "--w3-match-w1-size"),
                        ("--weather-regions", "1", "--w3-window", "300", "700")):
            with self.subTest(options=options), self.assertRaises(SystemExit) as error:
                with built_model(*options):
                    self.fail("Invalid W3 options should not build a model.")
            self.assertEqual(error.exception.code, 2)

    def test_weather_window_boundaries_and_legacy_scenarios(self):
        scenario = test_v18.load_scenario(3, 3)
        self.assertEqual(weather_windows(scenario), [(400, 800)] * 3)
        scenario["weather_windows_s"] = [(400, 800), (800, 1200), (1000, 1500)]
        for start, end, expected in (
                (0, 400, []), (400, 800, [0]), (800, 1000, [1]),
                (1000, 1200, [1, 2]), (1200, 1500, [2]), (1500, 1600, []),
                (485, 885, [0, 1]), (966, 1366, [1, 2])):
            with self.subTest(start=start, end=end):
                self.assertEqual([i for i, _ in active_weather_cells(start, end, scenario)], expected)
        scenario["weather_enabled"] = False
        self.assertEqual(active_weather_cells(0, 2000, scenario), [])
        for windows in ([(400, 800)], [(400, 800), (800, 800), (1000, 1500)],
                        [(400, 800), (800, float("nan")), (1000, 1500)]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                weather_windows(dict(scenario, weather_windows_s=windows))

    def test_reporting_ignores_inactive_cells_but_rejects_active_crossings(self):
        scenario = test_v18.load_scenario(3, 3)
        scenario.update(
            weather_windows_s=[(400, 800), (800, 1200), (1000, 1500)],
            origin_xy_m=[0, 0], basis=[[1, 0], [0, 1]],
            weather_boxes_uv_m=[[-20000, -20000, 20000, 20000],
                                [100000, -20000, 120000, 20000],
                                [200000, -20000, 220000, 20000]])
        for holding in (False, True):
            with self.subTest(holding=holding), tempfile.TemporaryDirectory() as directory:
                data = pd.DataFrame({
                    "t": [800, 1200], "f1_x": [0, 0] if holding else [-30000, 30000],
                    "f1_y": [0, 0], "f1_z": [3000, 3000], "f1_status": [1, 1],
                    "f1_hold": [0, int(holding)]})
                report_free_holding(data, scenario["aircraft"][0], scenario, Path(directory))
                checks = pd.read_csv(Path(directory) / "weather_checks.csv")
                self.assertEqual(checks.active_regions.iloc[0], "W2,W3")
                self.assertGreaterEqual(checks.clearance_m.iloc[0], 1000)
                data["t"] -= 400
                with self.assertRaisesRegex(ValueError, "Weather clearance failed"):
                    report_free_holding(data, scenario["aircraft"][0], scenario, Path(directory))

    def test_geometric_start_does_not_change_full_model(self):
        with built_model("--flight-count", "1") as ns:
            model = ns["m"]
            counts = (model.NumVars, model.NumConstrs, model.NumGenConstrs)
            objective = [(variable.VarName, variable.Obj) for variable in model.getVars()]
            ns["seed_geometric_start"](model, 15, ns["output_dir"] / "initialization.log")
            model.update()
            self.assertEqual((model.NumVars, model.NumConstrs, model.NumGenConstrs), counts)
            self.assertEqual([(variable.VarName, variable.Obj) for variable in model.getVars()], objective)
            self.assertIsNotNone(model.getConstrByName("fuel_0_1_one_tet"))
            self.assertEqual(ns["hold_budgets"][0].RHS, 0)
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = 15
            model.optimize()
            self.assertGreater(model.SolCount, 0)

    def test_hold_off_rejects_selected_hold_and_reports_failure(self):
        with built_model("--flight-count", "1") as ns:
            model = ns["m"]
            model.Params.OutputFlag = 0
            model.addConstr(ns["hold"][0, 1] == 1)
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.INFEASIBLE)
            folder = ns["output_dir"] / "failed_case"
            folder.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no feasible incumbent"):
                write_case_output(model, ns["scenario"], ns["flight_variables"],
                                  ns["fuel_table"], folder, False)
            report = json.loads((folder / "run_summary.json").read_text())
            self.assertEqual(report["solution_count"], 0)
            self.assertEqual(report["status"], "INFEASIBLE")
            self.assertFalse((folder / "trajectories.csv").exists())

    def test_airborne_hold_has_real_fuel_and_v18_outputs(self):
        with built_model("--flight-count", "1", "--holding-cost-factor", "0.5") as ns:
            fix_route(ns, holding=True)
            ns["m"].optimize()
            self.assertGreater(ns["m"].SolCount, 0)
            self.assertAlmostEqual(ns["fuel_step_kg"][0, 1].X, 0, delta=0.01)
            self.assertGreater(ns["hold_step_kg"][0, 1].X, 0)
            folder = ns["output_dir"] / "holding_on"
            folder.mkdir()
            result, schedules = write_case_output(
                ns["m"], ns["scenario"], ns["flight_variables"], ns["fuel_table"], folder, True)
            self.assertEqual(result["hold_seconds"], 400)
            self.assertEqual(result["airborne_seconds"], 2400)
            self.assertGreater(result["hold_fuel_kg"], 0)
            self.assertAlmostEqual(result["holding_objective_discount"],
                                   0.5 * result["holding_objective_before_discount"])
            np.testing.assert_allclose(schedules[0][["f1_x", "f1_y", "f1_z"]].iloc[0],
                                       schedules[0][["f1_x", "f1_y", "f1_z"]].iloc[1], atol=0.01)
            ac_folder = folder / ns["scenario"]["aircraft"][0]["acId"]
            self.assertGreater((ac_folder / "free_holding_2d.png").stat().st_size, 10000)

    def test_weather_alone_rejects_a_crossing_transit(self):
        with built_model("--flight-count", "1") as ns:
            fix_route(ns, holding=False)
            model = ns["m"]
            model.optimize()
            self.assertEqual(model.Status, gp.GRB.INFEASIBLE)
            weather_constraints = [
                constraint for constraint in model.getGenConstrs()
                if constraint.GenConstrType == gp.GRB.GENCONSTR_INDICATOR
                and model.getGenConstrIndicator(constraint)[0].VarName.startswith("weather_")
            ]
            self.assertTrue(weather_constraints)
            model.remove(weather_constraints)
            model.optimize()
            self.assertGreater(model.SolCount, 0, "The identical route must be feasible without weather.")

    def test_existing_output_directory_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("previous results", encoding="utf-8")
            with patch.object(sys, "argv", [str(SCRIPT), "--flight-count", "1", "--output-dir", directory]):
                with redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
                    exec(compile(BUILD_SOURCE, str(SCRIPT), "exec"), {"__file__": str(SCRIPT)})
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous results")

    @unittest.skipUnless(os.environ.get("V181_RESULTS_ROOT"), "Check the saved v18.1 paired examples.")
    def test_saved_v18_output_contract(self):
        root = Path(os.environ["V181_RESULTS_ROOT"])
        checker = test_v18.V18Tests("test_shared_weather_and_exact_entry_offsets")
        for name, count in (("single", 1), ("three_three", 3), ("six_one", 6)):
            with self.subTest(name=name):
                folder = root / name
                checker.check_comparison(folder)
                scenario = json.loads((folder / "scenario.json").read_text())
                self.assertEqual(len(scenario["aircraft"]), count)
                for case in ("holding_off", "holding_on"):
                    rows = pd.read_csv(folder / case / "aircraft_summary.csv")
                    for _, row in rows.iterrows():
                        data = pd.read_csv(folder / case / row.acId / "df_wide.csv")
                        x, y, (upper, lower) = scenario["star_candidates"][row.star_fix]
                        np.testing.assert_allclose(data[["f1_x", "f1_y"]].iloc[-1], [x, y], atol=0.01)
                        self.assertGreaterEqual(data.f1_z.iloc[-1], lower - 0.01)
                        self.assertLessEqual(data.f1_z.iloc[-1], upper + 0.01)

    @unittest.skipUnless(os.environ.get("V181_STAGGERED_RESULTS"), "Check a saved staggered-weather comparison.")
    def test_saved_staggered_weather_comparison(self):
        folder = Path(os.environ["V181_STAGGERED_RESULTS"])
        test_v18.V18Tests("test_shared_weather_and_exact_entry_offsets").check_comparison(folder)
        scenario = json.loads((folder / "scenario.json").read_text())
        windows = [[400, 800], [800, 1200], [1000, 1500]]
        self.assertEqual(scenario["weather_windows_s"], windows)
        self.assertEqual(len(scenario["aircraft"]), 3)
        if os.environ.get("V181_RESULTS_ROOT"):
            previous = json.loads((Path(os.environ["V181_RESULTS_ROOT"]) / "three_three" / "scenario.json").read_text())
            timing_keys = {"weather_windows_s", "weather_start_s", "weather_end_s"}
            self.assertEqual({k: v for k, v in scenario.items() if k not in timing_keys},
                             {k: v for k, v in previous.items() if k not in timing_keys})
        for case in ("holding_off", "holding_on"):
            case_scenario = json.loads((folder / case / "scenario.json").read_text())
            self.assertEqual(case_scenario["weather_windows_s"], windows)
            for aircraft in scenario["aircraft"]:
                checks = pd.read_csv(folder / case / aircraft["acId"] / "weather_checks.csv").fillna("")
                for row in checks.itertuples():
                    expected = [f"W{i + 1}" for i, (start, end) in enumerate(windows)
                                if row.end_t - 400 < end and row.end_t > start]
                    self.assertEqual(row.active_regions, ",".join(expected))
                    self.assertEqual(row.weather_active, bool(expected))
                    if expected:
                        self.assertGreaterEqual(row.clearance_m, scenario["clearance_m"] - 0.01)


if __name__ == "__main__":
    unittest.main()
