"""Offline regression tests; no downloads or Gurobi license required."""

import ast
import argparse
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import numpy as np

# The debug script builds and solves its model at module scope. Load only its
# weather functions so these offline checks never execute the optimization.
_milp_path = (Path(__file__).resolve().parents[1] / "3_MILP_Serra" /
              "2.1_MILP" / "2.1.2_MILP" / "milp_multiple_Debug.py")
_weather_names = {"utc_naive", "load_weather_frames", "add_weather_constraints",
                  "build_arg_parser", "plot_weather_results"}
_tree = ast.parse(_milp_path.read_text(), filename=str(_milp_path))
_functions = [node for node in _tree.body
              if isinstance(node, ast.FunctionDef) and node.name in _weather_names]
assert {node.name for node in _functions} == _weather_names
_namespace = {"bisect_right": bisect_right, "timedelta": timedelta,
              "timezone": timezone, "math": math, "Path": Path,
              "np": np, "argparse": argparse}
exec(compile(ast.Module(body=_functions, type_ignores=[]), str(_milp_path), "exec"),
     _namespace)
add_weather_constraints = _namespace["add_weather_constraints"]
load_weather_frames = _namespace["load_weather_frames"]


class FixedTrajectoryModel:
    """Evaluate generated indicator disjunctions for fixed numeric positions."""

    def __init__(self):
        self.groups = []

    def addVars(self, count, **kwargs):
        group = [[] for _ in range(count)]
        self.groups.append(group)

        class Sides(dict):
            def sum(self):
                # addConstr's OR requires at least one of these sides.
                return 1

        return Sides(enumerate(group))

    def addConstr(self, condition, **kwargs):
        assert condition

    def addGenConstrIndicator(self, side, enabled, condition, **kwargs):
        assert enabled is True
        side.append(bool(condition))

    def feasible(self):
        return all(any(all(side) for side in group) for group in self.groups)


class WeatherTests(unittest.TestCase):
    start = datetime(2023, 12, 22, 18, 9, 12)
    square = (0.0, 1.0, 0.0, 1.0)

    def test_square_scenario_command(self):
        args = _namespace["build_arg_parser"]().parse_args([
            "--shape", "square", "--horizon-minutes", "60", "--plot",
            "--weather-start", "2026-09-03T19:00:00Z", "--time-limit", "300"])
        self.assertEqual(args.shape, "square")
        self.assertEqual(args.horizon_minutes, 60)
        self.assertTrue(args.plot)
        self.assertEqual(args.time_limit, 300)

    def test_weather_plot_artifacts(self):
        solution = pd.DataFrame({"t": [0, 300, 600],
            "f1_lon": [-2, -1, -1], "f1_lat": [0.5, 0.5, 1.5],
            "f1_alt_ft": [30000, 25000, 20000],
            "f2_lon": [-2, -2, -1], "f2_lat": [2, 2, 2],
            "f2_alt_ft": [31000, 31000, 26000]})
        frames = [(self.start - timedelta(seconds=252), (self.square,)),
                  (self.start + timedelta(seconds=48), ()),
                  (self.start + timedelta(seconds=348), (self.square,))]
        with tempfile.TemporaryDirectory() as tmp:
            _namespace["plot_weather_results"](solution, [["Flight A"], ["Flight B"]],
                [0, 1], frames, self.start, tmp, "square", 0.05,
                tracon_polygon=[(-0.5, -0.5), (0.5, -0.5), (0, 0.5)])
            self.assertEqual((Path(tmp) / "trajectories.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual((Path(tmp) / "weather_trajectories.gif").read_bytes()[:6], b"GIF89a")

    def evaluate(self, lat, lon, frames=None, times=None, entry=0):
        model = FixedTrajectoryModel()
        add_weather_constraints(
            model, [lat], [lon], [entry], times or [0, 300], self.start,
            frames or [(self.start, (self.square,)),
                       (self.start + timedelta(minutes=5), (self.square,))])
        return model.feasible()

    def test_all_four_exterior_sides(self):
        for lat, lon in [([-1, -1], [0, 1]), ([2, 2], [0, 1]),
                         ([0, 1], [-1, -1]), ([0, 1], [2, 2])]:
            with self.subTest(lat=lat, lon=lon):
                self.assertTrue(self.evaluate(lat, lon))

    def test_inside_and_boundary_rejected(self):
        self.assertFalse(self.evaluate([0.5, 0.5], [0.5, 0.5]))
        self.assertFalse(self.evaluate([0, 0], [0.5, 0.5]))

    def test_clear_endpoints_crossing_square_rejected(self):
        self.assertFalse(self.evaluate([-1, 2], [0.5, 0.5]))

    def test_narrow_crossing_between_old_sample_points_rejected(self):
        rect = (0.1, 0.11, 0.0, 1.0)
        self.assertFalse(self.evaluate([0, 1], [0.5, 0.5],
                                      frames=[(self.start, (rect,))], times=[0, 299]))

    def test_clear_weather_has_no_constraints(self):
        self.assertTrue(self.evaluate([0.5, 0.5], [0.5, 0.5],
                                     frames=[(self.start, ())], times=[0, 299]))

    def test_pre_entry_positions_ignored(self):
        self.assertTrue(self.evaluate([0.5, 2], [0.5, 2], entry=1))

    def test_non_aligned_weather_boundary_splits_segment(self):
        # Clear frame first; storm starts after the aircraft has passed it.
        frames = [(self.start - timedelta(seconds=252), ()),
                  (self.start + timedelta(seconds=48), (self.square,))]
        self.assertTrue(self.evaluate([0.5, 0.5], [0.5, 10], frames=frames))
        # Reversing the trip now intersects weather during the storm frame.
        self.assertFalse(self.evaluate([0.5, 0.5], [10, 0.5], frames=frames))

    def test_no_holding_stale_weather_forever(self):
        with self.assertRaisesRegex(ValueError, "does not cover"):
            self.evaluate([2, 2], [2, 2], frames=[(self.start, ())])

    def test_gapped_frames_rejected(self):
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.evaluate([2, 2], [2, 2], frames=[(self.start, ()),
                (self.start + timedelta(minutes=10), ())])

    def runner(self, **kwargs):
        self.calls.append(kwargs)
        start = datetime.fromisoformat(kwargs["date"] + "T" + kwargs["start_time"])
        end = datetime.fromisoformat(kwargs["date"] + "T" + kwargs["end_time"])
        data = {}
        while start <= end:
            for level, flag in [("FL110", True), ("FL350", False), ("FL390", True)]:
                data[start, level] = pd.DataFrame([dict(zip(
                    ["infeasible", "min_lat", "max_lat", "min_lon", "max_lon"],
                    [flag, *self.square]))])
            start += timedelta(minutes=kwargs["step_minutes"])
        return {"flight_levels": ["FL110", "FL350", "FL390"], "data": data}

    def test_load_midnight_union_deduplication_and_alignment(self):
        self.calls = []
        frames = load_weather_frames(datetime(2023, 12, 22, 23, 58, 12),
                                     datetime(2023, 12, 23, 0, 6), runner=self.runner)
        self.assertEqual([f[0] for f in frames], [datetime(2023, 12, 22, 23, 55),
            datetime(2023, 12, 23), datetime(2023, 12, 23, 0, 5)])
        self.assertTrue(all(rects == (self.square,) for _, rects in frames))
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(call["return_data"] for call in self.calls))

    def test_timezone_is_normalized_to_utc(self):
        self.calls = []
        local = self.start.replace(tzinfo=timezone(timedelta(hours=-5)))
        frames = load_weather_frames(local, local, runner=self.runner)
        self.assertEqual(frames[0][0], datetime(2023, 12, 22, 23, 5))

    def test_missing_level_fails_closed(self):
        self.calls = []

        def incomplete(**kwargs):
            result = self.runner(**kwargs)
            result["data"].pop(next(iter(result["data"])))
            return result

        with self.assertRaisesRegex(ValueError, "Missing CWAM weather"):
            load_weather_frames(self.start, self.start, runner=incomplete)


if __name__ == "__main__":
    unittest.main()
