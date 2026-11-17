import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from convert_iff_to_filtered_rows import OUTPUT_COLUMNS, convert_iff


def header(key="1", acid="TEST1", departure="KBUF", arrival="KCMH"):
    row = [""] * 16
    row[0], row[2], row[5], row[7] = "2", key, "ZOB", acid
    row[10], row[11] = departure, arrival
    return row


def point(key="1", acid="TEST1", timestamp="1703218248.000", width=45):
    row = [""] * width
    row[0], row[1], row[2], row[5], row[7] = "3", timestamp, key, "ZOB", acid
    row[9:12] = ["42.88167", "-78.67500", "42.00"]
    row[16:19] = ["228", "201", "1200"]
    return row


class IffConversionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.input = self.directory / "input.csv"
        self.output = self.directory / "output.csv"

    def write_input(self, rows):
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows([["1", "IFF", "2.15"], *rows])

    def read_output(self):
        with self.output.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, OUTPUT_COLUMNS)
            self.assertEqual(len(reader.fieldnames), 47)
            return list(reader)

    def test_exact_schema_and_unmodified_track_fields(self):
        track = point()
        track[23] = "quoted, scratch pad"
        track[44] = "track-id"
        self.write_input([header(), track])
        self.assertEqual(convert_iff(self.input, self.output), (1, 0))
        rows = self.read_output()
        self.assertEqual(rows[0]["recTime"], "2023-12-21 23:10:48.000")
        self.assertEqual(rows[0]["acId"], "TEST1_KBUFtoKCMH")
        self.assertEqual((rows[0]["dep"], rows[0]["arrival"]), ("KBUF", "KCMH"))
        for index, name in enumerate(OUTPUT_COLUMNS[:45]):
            if index not in (1, 7):
                self.assertEqual(rows[0][name], track[index])

    def test_summer_timezone_and_utc_override(self):
        self.write_input([header(), point(timestamp="1786422609.123")])
        convert_iff(self.input, self.output)
        self.assertEqual(self.read_output()[0]["recTime"], "2026-08-11 00:30:09.123")
        utc_output = self.directory / "utc.csv"
        convert_iff(self.input, utc_output, timezone_name="UTC")
        with utc_output.open(newline="", encoding="utf-8") as stream:
            self.assertEqual(next(csv.DictReader(stream))["recTime"],
                             "2026-08-11 04:30:09.123")

    def test_fractional_seconds_are_not_float_rounded(self):
        self.write_input([header(), point(timestamp="1703218248.123456")])
        convert_iff(self.input, self.output, timezone_name="UTC")
        self.assertEqual(self.read_output()[0]["recTime"], "2023-12-22 04:10:48.123456")

    def test_join_uses_flight_key_and_accepts_headers_after_points(self):
        self.write_input([
            point(), point(key="2"),
            header(), header(key="2", departure="KCMH", arrival="KBUF"),
        ])
        self.assertEqual(convert_iff(self.input, self.output), (2, 0))
        self.assertEqual([row["acId"] for row in self.read_output()],
                         ["TEST1_KBUFtoKCMH", "TEST1_KCMHtoKBUF"])

    def test_legacy_36_field_records_are_padded(self):
        self.write_input([header(), point(width=36)])
        convert_iff(self.input, self.output)
        row = self.read_output()[0]
        self.assertEqual(row["alt"], "42.00")
        self.assertTrue(all(row[name] == "" for name in OUTPUT_COLUMNS[36:45]))
        self.assertEqual(row["arrival"], "KCMH")

    def test_unknown_airports_are_not_replaced_with_estimates_or_plans(self):
        metadata = header(departure="?", arrival="?")
        metadata[13:15] = ["BUF", "CMH"]
        plan = header()
        plan[0] = "4"
        self.write_input([metadata, plan, point()])
        with self.assertRaisesRegex(ValueError, "lack reported origin/destination"):
            convert_iff(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_explicit_skip_reports_missing_flights(self):
        self.write_input([
            header(), point(),
            header(key="2", acid="TEST2", departure="?", arrival="?"),
            point(key="2", acid="TEST2"),
        ])
        with self.assertLogs("convert_iff_to_filtered_rows", level="WARNING") as logs:
            self.assertEqual(convert_iff(self.input, self.output, skip_missing_airports=True),
                             (1, 1))
        self.assertIn("Flight keys: 2", logs.output[0])
        self.assertEqual(len(self.read_output()), 1)

    def test_missing_header_and_all_skipped_files_fail(self):
        self.write_input([point()])
        for skip in (False, True):
            with self.subTest(skip=skip), self.assertRaises(ValueError):
                convert_iff(self.input, self.output, skip_missing_airports=skip)
            self.assertFalse(self.output.exists())

    def test_conflicting_headers_or_callsigns_fail(self):
        cases = [
            [header(), header(arrival="KDTW"), point()],
            [header(), point(acid="OTHER1")],
            [header(), point(), point(acid="OTHER1")],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.write_input(rows)
                convert_iff(self.input, self.output)
            self.assertFalse(self.output.exists())

    def test_identical_headers_do_not_multiply_points(self):
        self.write_input([header(), header(), point()])
        self.assertEqual(convert_iff(self.input, self.output), (1, 0))
        self.assertEqual(len(self.read_output()), 1)

    def test_source_duplicates_and_blank_fields_are_preserved(self):
        first = point()
        first[3] = "1234"
        self.write_input([header(), first, point(), point()])
        self.assertEqual(convert_iff(self.input, self.output), (3, 0))
        rows = self.read_output()
        self.assertEqual([row["bcnCode"] for row in rows], ["1234", "", ""])
        self.assertEqual(rows[1], rows[2])

    def test_unsupported_record_width_fails_before_output(self):
        self.write_input([header(), point(width=44)])
        with self.assertRaisesRegex(ValueError, "44 fields"):
            convert_iff(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_timestamps_leave_no_partial_output(self):
        for value in ("bad", "NaN", "Infinity", "1703218248.1234567"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "invalid recTime"):
                self.write_input([header(), point(), point(timestamp=value)])
                convert_iff(self.input, self.output)
            self.assertFalse(self.output.exists())
            self.assertEqual(list(self.directory.glob(".iff-conversion-*")), [])

    def test_input_and_existing_output_are_protected(self):
        self.write_input([header(), point()])
        original = self.input.read_bytes()
        with self.assertRaises(ValueError):
            convert_iff(self.input, self.input)
        self.assertEqual(self.input.read_bytes(), original)
        self.output.write_text("existing output", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            convert_iff(self.input, self.output)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output")

    def test_lfs_pointer_and_no_track_points_fail(self):
        self.input.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not a raw IFF"):
            convert_iff(self.input, self.output)
        self.write_input([header()])
        with self.assertRaisesRegex(ValueError, "no type-3"):
            convert_iff(self.input, self.output)
        self.assertFalse(self.output.exists())

    def test_cli_errors_and_successful_conversion(self):
        script = Path(__file__).with_name("convert_iff_to_filtered_rows.py")
        self.write_input([header(departure="?"), point()])
        command = [sys.executable, str(script), str(self.input), str(self.output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--skip-missing-airports", result.stderr)
        self.assertFalse(self.output.exists())
        self.write_input([header(), point()])
        result = subprocess.run(command + ["--timezone", "UTC"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Saved 1 rows and 47 columns", result.stdout)
        self.assertEqual(self.read_output()[0]["recTime"], "2023-12-22 04:10:48.000")


if __name__ == "__main__":
    unittest.main()
