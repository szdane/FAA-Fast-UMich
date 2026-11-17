import csv
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from animate_zob_day import DAY_SECONDS, Flight, load_flights
from compute_fuel_history import (
    aggregate_day, compute_file, compute_flight, create_model, fuel_events,
    load_daily_fuel, validate_day,
)


class ConstantFuel:
    def enroute(self, *, mass, tas, alt, vs):
        return 2.0


class MassDependentFuel:
    def __init__(self):
        self.masses = []

    def enroute(self, *, mass, tas, alt, vs):
        self.masses.append(mass)
        return mass / 1000


class FuelHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.input = self.directory / "tracks.csv"
        self.output = self.directory / "fuel"
        self.day = date(2023, 12, 22)
        self.aircraft = {"mtow": 1250, "oew": 100, "ceiling": 13100,
                         "engine": {"default": "test-engine"}}

    def flight(self, times=(0, 10, 30), speeds=None, altitudes=None, key="1"):
        n = len(times)
        samples = np.column_stack((times, np.full(n, -80), np.full(n, 42),
                                   np.full(n, 30000) if altitudes is None else altitudes))
        kinematics = np.column_stack((np.full(n, 250) if speeds is None else speeds, np.zeros(n)))
        return Flight(key, "SAME", "KBUF", "KCMH", samples, "ZOB", kinematics)

    def calculate(self, flight, model=None):
        return compute_flight(flight, model or ConstantFuel(), 1000, 100, 43000)

    def write_input(self, speed=250):
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["recType", "recTime", "fltKey", "Source", "acId",
                             "coord1", "coord2", "alt", "dep", "arrival", "groundSpeed", "rateOfClimb"])
            for key, instant in (("1", "2023-12-21 23:59:50.000"), ("1", "2023-12-22 00:00:10.000"),
                                 ("2", "2023-12-22 12:00:00.000"), ("2", "2023-12-22 12:00:20.000")):
                writer.writerow(["3", instant, key, "ZOB", "SAME_KBUFtoKCMH",
                                 "42", "-80", "300", "KBUF", "KCMH", speed, "0"])

    def compute_output(self):
        self.write_input()
        with patch("compute_fuel_history.create_model", return_value=(ConstantFuel(), self.aircraft)):
            with self.assertLogs("compute_fuel_history", level="INFO") as messages:
                metadata = compute_file(self.input, self.output, self.day)
        self.assertTrue(any("INITIAL ESTIMATE from 2 flights" in line for line in messages.output))
        return metadata

    def test_exact_units_right_endpoint_mass_update_and_no_fictitious_first_interval(self):
        model = MassDependentFuel()
        history = self.calculate(self.flight(), model)
        np.testing.assert_allclose(model.masses, [1000, 1000, 990])
        np.testing.assert_allclose(history.flow, [1, 1, 0.99])
        np.testing.assert_allclose(history.interval_fuel, [0, 10, 19.8])
        np.testing.assert_allclose(history.cumulative_fuel, [0, 10, 29.8])
        np.testing.assert_allclose(history.mass, [1000, 990, 970.2])
        self.assertEqual(history.status, ["first_observation", "modeled", "modeled"])

    def test_long_gaps_are_excluded_without_resetting_mass(self):
        history = self.calculate(self.flight(times=(0, 10, 200, 210)))
        np.testing.assert_array_equal(history.modeled_duration, [0, 10, 0, 10])
        np.testing.assert_array_equal(history.cumulative_fuel, [0, 20, 20, 40])
        self.assertEqual(history.status[2], "excluded_gap_over_120s")

    def test_exactly_120_second_intervals_are_included(self):
        history = self.calculate(self.flight(times=(0, 120)))
        self.assertEqual(history.interval_fuel[-1], 240)

    def test_bad_speed_excludes_both_touching_intervals_and_is_not_clamped(self):
        for bad_speed in (0, -1, 651, 7572):
            with self.subTest(speed=bad_speed):
                history = self.calculate(self.flight(times=(0, 10, 20, 30), speeds=(250, bad_speed, 250, 250)))
                self.assertTrue(np.isnan(history.flow[1]))
                np.testing.assert_array_equal(history.interval_fuel, [0, 0, 0, 20])
                self.assertEqual(history.status[1:3], ["excluded_speed_endpoint"] * 2)
                self.assertIn("invalid_reported_speed", history.flags[1])

    def test_above_ceiling_scenario_extrapolations_are_flagged(self):
        history = self.calculate(self.flight(times=(0, 10), altitudes=(45000, 45000)))
        self.assertEqual(history.cumulative_fuel[-1], 20)
        self.assertIn("above_B788_ceiling", history.flags[1])

    def test_ground_speed_states_cannot_overflow_the_airborne_model(self):
        model, aircraft = create_model()
        flight = self.flight(times=(0, 10, 20, 30), speeds=(250, 12, 250, 250),
                             altitudes=(600, 600, 600, 600))
        history = compute_flight(flight, model, 0.8 * aircraft["mtow"], aircraft["oew"],
                                 aircraft["ceiling"] / 0.3048)
        self.assertTrue(np.isnan(history.flow[1]))
        self.assertIn("below_100kt", history.flags[1])
        np.testing.assert_array_equal(history.modeled_duration, [0, 0, 0, 10])
        self.assertGreater(history.cumulative_fuel[-1], 0)

    def test_speed_threshold_is_inclusive_and_low_speeds_are_not_taxi_estimates(self):
        history = self.calculate(self.flight(times=(0, 10, 20, 30), speeds=(99, 100, 650, 651)))
        np.testing.assert_array_equal(history.interval_fuel, [0, 0, 20, 0])
        self.assertIn("below_100kt", history.flags[0])

    def test_no_negative_or_nonfinite_model_results_are_silently_accepted(self):
        for result in (-1, 0, float("nan"), float("inf")):
            with self.subTest(result=result), patch.object(ConstantFuel, "enroute", return_value=result):
                with self.assertRaisesRegex(ValueError, "Invalid OpenAP"):
                    self.calculate(self.flight())

    def test_empty_weight_and_invalid_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty weight"):
            compute_flight(self.flight(times=(0, 120)), ConstantFuel(), 300, 100, 43000)
        for times in ((0, 0), (10, 0)):
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "strictly increasing"):
                self.calculate(self.flight(times=times))

    def test_single_point_has_zero_duration_not_extrapolated_fuel(self):
        flight = self.flight(times=(5,))
        history = self.calculate(flight)
        self.assertEqual(history.cumulative_fuel[-1], 0)
        curve = aggregate_day([fuel_events(flight, history)])
        np.testing.assert_array_equal(curve, [[0, 0], [DAY_SECONDS, 0]])

    def test_midnight_clipping_excludes_other_days_but_preserves_full_history(self):
        flight = self.flight(times=(-10, 10, DAY_SECONDS - 10, DAY_SECONDS + 10))
        history = self.calculate(flight)
        np.testing.assert_array_equal(history.day_duration, [0, 10, 0, 10])
        self.assertEqual(history.cumulative_fuel[-1], 80)
        self.assertEqual(history.day_fuel.sum(), 40)
        curve = aggregate_day([fuel_events(flight, history)])
        self.assertEqual(curve[-1, 1], 40)
        self.assertEqual(np.interp(5, curve[:, 0], curve[:, 1]), 10)
        self.assertEqual(np.interp(43200, curve[:, 0], curve[:, 1]), 20)

    def test_fleet_total_is_sum_of_concurrent_intervals_without_future_fuel(self):
        flights = [self.flight(times=(0, 60)), self.flight(times=(30, 90), key="2")]
        curve = aggregate_day([fuel_events(f, self.calculate(f)) for f in flights])
        np.testing.assert_allclose(np.interp([0, 15, 30, 60, 90, DAY_SECONDS],
                                            curve[:, 0], curve[:, 1]), [0, 30, 60, 180, 240, 240])

    def test_full_history_loader_preserves_callsign_legs_and_correct_input_units(self):
        self.write_input()
        flights = load_flights(self.input, self.day, include_kinematics=True, full_history=True)
        self.assertEqual([f.key for f in flights], ["1", "2"])
        self.assertEqual(flights[0].samples[0, 3], 30000)
        np.testing.assert_array_equal(flights[0].kinematics, [[250, 0], [250, 0]])
        self.assertEqual(flights[0].samples[0, 0], -10)

    def test_outputs_cover_every_flight_and_row_and_reconcile_day_total(self):
        metadata = self.compute_output()
        self.assertEqual(metadata["flight_count"], 2)
        self.assertEqual(metadata["observation_count"], 4)
        self.assertEqual(metadata["full_record_fuel_kg"], 80)
        self.assertEqual(metadata["day_fuel_kg"], 60)
        with (self.output / "flight_fuel_history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual([r["fltKey"] for r in rows], ["1", "1", "2", "2"])
        self.assertEqual(float(rows[1]["cumulative_fuel_kg"]), 40)
        self.assertEqual(float(rows[1]["day_cumulative_fuel_kg"]), 20)
        curve = load_daily_fuel(self.output, self.input, self.day, "America/Detroit")
        self.assertEqual(curve[-1, 1], 60)
        self.assertEqual(metadata["pilot_timing"]["observations"], 4)

    def test_preexisting_outputs_and_mismatched_fuel_are_rejected(self):
        self.compute_output()
        with self.assertRaises(FileExistsError):
            compute_file(self.input, self.output, self.day)
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_daily_fuel(self.output, self.input, date(2023, 12, 21), "America/Detroit")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_daily_fuel(self.output, self.input, self.day, "UTC")
        with self.input.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_daily_fuel(self.output, self.input, self.day, "America/Detroit")

    def test_flights_without_usable_intervals_are_explicitly_unavailable(self):
        self.write_input(speed=12)
        with patch("compute_fuel_history.create_model", return_value=(ConstantFuel(), self.aircraft)), \
                self.assertLogs("compute_fuel_history", level="WARNING") as messages:
            metadata = compute_file(self.input, self.output, self.day)
        self.assertEqual(metadata["flights_without_modeled_intervals"], 2)
        self.assertTrue(any("NOT a fuel-consumption estimate" in line for line in messages.output))
        with (self.output / "flight_fuel_summary.csv").open(newline="", encoding="utf-8") as stream:
            summaries = list(csv.DictReader(stream))
        self.assertEqual({row["coverage"] for row in summaries}, {"unavailable_no_modeled_intervals"})

    def test_tampered_curve_total_is_rejected(self):
        self.compute_output()
        metadata_path = self.output / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["day_fuel_kg"] += 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recorded total"):
            load_daily_fuel(self.output, self.input, self.day, "America/Detroit")

    def test_failed_computation_does_not_publish_partial_results(self):
        self.write_input()
        with patch("compute_fuel_history.create_model", return_value=(ConstantFuel(), self.aircraft)), \
                patch.object(ConstantFuel, "enroute", return_value=-1):
            with self.assertRaises(ValueError):
                compute_file(self.input, self.output, self.day)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.iterdir()), [self.input])

    def test_dst_day_requires_utc_input(self):
        with self.assertRaisesRegex(ValueError, "DST-transition"):
            validate_day(date(2023, 3, 12), "America/Detroit")
        validate_day(date(2023, 3, 12), "UTC")

    def test_real_b788_model_is_positive_and_depletes_mass(self):
        model, aircraft = create_model()
        flight = self.flight(times=(0, 10, 20), speeds=(450, 450, 450), altitudes=(35000, 35000, 35000))
        history = compute_flight(flight, model, 0.8 * aircraft["mtow"], aircraft["oew"],
                                 aircraft["ceiling"] / 0.3048)
        self.assertTrue(np.all(history.flow > 0))
        self.assertTrue(np.all(history.flow < 20))
        self.assertGreater(history.cumulative_fuel[-1], 0)
        self.assertLess(history.mass[-1], history.mass[0])


if __name__ == "__main__":
    unittest.main()
