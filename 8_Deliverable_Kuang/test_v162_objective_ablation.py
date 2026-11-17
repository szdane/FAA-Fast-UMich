"""Focused checks for objective-only changes and trustworthy experiment outcomes."""

from contextlib import redirect_stdout
import csv
import gzip
import io
import math
from pathlib import Path
import tempfile
import unittest

import gurobipy as gp

from run_v162_objective_ablation import (
    CASES, SolverTrace, apply_case, built_model, objective_groups, outcome,
    relative_gap, run_experiment, zero_objective_fingerprint,
    write_model,
)


class ObjectiveAblationTests(unittest.TestCase):
    def test_export_uses_portable_compression(self):
        with tempfile.TemporaryDirectory() as directory, gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                model.addVar(vtype=gp.GRB.BINARY)
                folder = Path(directory)
                write_model(model, folder)
                self.assertFalse((folder / "model.mps").exists())
                with gzip.open(folder / "model.mps.gz", "rt") as stream:
                    self.assertIn("ENDATA", stream.read())

    def test_full_model_changes_only_requested_objective_coefficients(self):
        with redirect_stdout(io.StringIO()), built_model() as namespace:
            model = namespace["m"]
            original = model.getObjective()
            original_coefficients = model.getAttr("Obj")
            original_constant = model.ObjCon
            original_fingerprint = model.Fingerprint
            constraint_fingerprint = zero_objective_fingerprint(model)
            original_starts = model.getAttr("Start")
            self.assertEqual(model.Fingerprint, original_fingerprint)
            distance, heading = objective_groups(namespace)
            self.assertTrue(distance)
            self.assertTrue(heading)
            self.assertEqual(namespace["w_descent"], 0)
            for variables, weight in ((distance, namespace["w_dist"]), (heading, namespace["w_accel"])):
                self.assertGreater(weight, 0)
                self.assertTrue(all(variable.Obj == weight for variable in variables))
            for case, keeps in CASES.items():
                with self.subTest(case=case):
                    model.setObjective(original, gp.GRB.MINIMIZE)
                    model.update()
                    apply_case(namespace, case)
                    expected = original_coefficients.copy()
                    for variables, keep in zip((distance, heading), keeps):
                        if not keep:
                            for variable in variables:
                                expected[variable.index] = 0.0
                    self.assertEqual(model.getAttr("Obj"), expected)
                    self.assertEqual(model.ObjCon, original_constant)
                    self.assertEqual(model.getAttr("Start"), original_starts)
                    self.assertEqual(zero_objective_fingerprint(model), constraint_fingerprint)
                    self.assertIsNotNone(model.getConstrByName("accel_x_ub_0_2"))
                    self.assertGreater(model.NumGenConstrs, 0)

    def test_trace_records_actual_incumbent(self):
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            with gp.Model(env=env) as model:
                x = model.addVar(vtype=gp.GRB.BINARY)
                model.addConstr(x == 1)
                model.setObjective(x)
                stream = io.StringIO()
                trace = SolverTrace(stream)
                model.optimize(trace)
                rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                self.assertIsNotNone(trace.first_incumbent_s)
                self.assertEqual(trace.first_incumbent_objective, 1.0)
                self.assertTrue(any(row["event"] == "solution" for row in rows))

    def test_feasible_time_limit_is_not_reported_as_convergence(self):
        self.assertEqual(outcome(gp.GRB.TIME_LIMIT, 1, 0.02, 1200, 900, True), "FEASIBLE_GAP_OPEN")
        self.assertEqual(outcome(gp.GRB.TIME_LIMIT, 0, None, 1200, 900, False), "NO_INCUMBENT")
        self.assertEqual(outcome(gp.GRB.OPTIMAL, 1, 0.005, 800, 900, True), "PASS")
        self.assertEqual(outcome(gp.GRB.OPTIMAL, 1, 0.005, 901, 900, True), "SOLVED_OVER_TARGET")
        self.assertEqual(outcome(gp.GRB.INFEASIBLE, 0, None, 1, 900, False), "INFEASIBLE")
        self.assertEqual(outcome(gp.GRB.NUMERIC, 1, 0.005, 1, 900, True), "NUMERIC")
        self.assertEqual(outcome(gp.GRB.OPTIMAL, 1, 0.005, 1, 900, False),
                         "SOLUTION_TOLERANCE_EXCEEDED")

    def test_gap_handles_zero_and_missing_incumbents(self):
        self.assertIsNone(relative_gap(None, 10))
        self.assertIsNone(relative_gap(10, None))
        self.assertIsNone(relative_gap(0, -1))
        self.assertEqual(relative_gap(0, 0), 0)
        self.assertEqual(relative_gap(-100, -110), 0.1)

    def test_invalid_settings_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not_created"
            for limit, acceptable in ((math.inf, 900), (1200, math.nan), (10, 20), (0, 0)):
                with self.subTest(limit=limit, acceptable=acceptable), self.assertRaises(ValueError):
                    run_experiment(["baseline"], [0], limit, acceptable, output)
            self.assertFalse(output.exists())
            for cases, seeds in ((["unknown"], [0]), (["baseline", "baseline"], [0]),
                                 (["baseline"], [-1]), (["baseline"], [0, 0])):
                with self.subTest(cases=cases, seeds=seeds), self.assertRaises(ValueError):
                    run_experiment(cases, seeds, output_dir=output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
