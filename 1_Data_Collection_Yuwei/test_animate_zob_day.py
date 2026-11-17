import csv
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from animate_zob_day import (
    DAY_SECONDS, Flight, GEOGRAPHIC, fetch_json, frame_data, frame_times,
    load_artcc_boundary, load_flights, make_figure, plt, prepare_playback,
    read_boundary_dataset, render_day, reporting_facility,
)


class DailyAnimationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.input = self.directory / "tracks.csv"
        self.day = date(2023, 12, 22)
        self.boundaries = self.directory / "boundaries.geojson"
        self.boundary_data = {
            "type": "FeatureCollection",
            "features": [
                self.feature(),
                self.feature(chart="ARTCC_L", bounds=(-85.2, 39, -76, 44)),
                self.feature("ZAU", bounds=(-94, 40, -85, 45)),
            ],
            "faa_metadata": {"effective_from": "2023-11-30", "effective_to": "2023-12-28"},
        }
        self.save_boundaries()

    def feature(self, facility="ZOB", chart="ARTCC_H", bounds=(-85, 39, -76, 44)):
        west, south, east, north = bounds
        return {
            "type": "Feature",
            "properties": {"IDENT": facility, "NAME": {"ZOB": "CLEVELAND", "ZAU": "CHICAGO"}[facility],
                           "TYPE_CODE": "ARTCC", "LOCAL_TYPE": chart},
            "geometry": {"type": "Polygon", "coordinates": [
                [[west, south], [east, south], [east, north], [west, north], [west, south]],
            ]},
        }

    def save_boundaries(self):
        self.boundaries.write_text(json.dumps(self.boundary_data), encoding="utf-8")

    def boundary(self, facility="ZOB"):
        return load_artcc_boundary(facility, self.boundaries)

    def write_rows(self, rows):
        with self.input.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["recType", "recTime", "fltKey", "Source", "acId",
                             "coord1", "coord2", "alt", "dep", "arrival"])
            writer.writerows(rows)

    def point(self, timestamp="2023-12-22 12:00:00.000", key="1", latitude="42",
              source="ZOB", departure="KBUF", arrival="KCMH"):
        return ["3", timestamp, key, source, f"TEST1_{departure}to{arrival}",
                latitude, "-80", "300", departure, arrival]

    def flight(self, key="1", times=(0, 60), longitudes=(-80, -79), facility="ZOB"):
        samples = np.column_stack((times, longitudes, np.full(len(times), 42),
                                   np.full(len(times), 30000)))
        return Flight(key, "TEST1", "KBUF", "KCMH", samples, facility)

    def test_same_callsign_different_keys_remain_separate(self):
        self.write_rows([self.point(key="1"), self.point(key="2", departure="KCMH", arrival="KBUF")])
        flights = load_flights(self.input, self.day)
        self.assertEqual(len(flights), 2)
        self.assertEqual([flight.key for flight in flights], ["1", "2"])
        self.assertEqual([flight.departure for flight in flights], ["KBUF", "KCMH"])
        self.assertEqual(flights[0].samples[0, 3], 30000)

    def test_day_is_half_open_and_adjacent_samples_do_not_add_flights(self):
        self.write_rows([
            self.point("2023-12-21 23:59:30.000"),
            self.point("2023-12-22 00:00:30.000"),
            self.point("2023-12-23 00:00:00.000", key="2"),
        ])
        flights = load_flights(self.input, self.day)
        self.assertEqual(len(flights), 1)
        np.testing.assert_array_equal(flights[0].samples[:, 0], [-30, 30])
        playback = prepare_playback(flights, self.boundary())
        self.assertEqual(playback.day_point_count, 1)
        self.assertEqual(len(frame_data(playback, 0, 600)[4]), 1)

    def test_sort_and_identical_duplicates(self):
        early = self.point("2023-12-22 10:00:00.000")
        self.write_rows([self.point(), early, early])
        with self.assertLogs("animate_zob_day", level="WARNING"):
            flights = load_flights(self.input, self.day)
        np.testing.assert_array_equal(flights[0].samples[:, 0], [36000, 43200])

    def test_conflicting_duplicates_or_metadata_fail(self):
        for second in (self.point(latitude="43"), self.point(arrival="KDTW")):
            with self.subTest(second=second):
                self.write_rows([self.point(), second])
                with self.assertRaises(ValueError):
                    load_flights(self.input, self.day)

    def test_invalid_data_and_empty_day_fail(self):
        for rows in ([self.point(latitude="nan")], [self.point(latitude="95")],
                     [self.point(source="INVALID-FACILITY")], [self.point("2023-12-21 12:00:00.000")]):
            with self.subTest(rows=rows):
                self.write_rows(rows)
                with self.assertRaises(ValueError):
                    load_flights(self.input, self.day)

    def test_midpoint_interpolation_and_no_extrapolation(self):
        flight = self.flight()
        playback = prepare_playback([flight], self.boundary())
        heads = frame_data(playback, 30, 600)[4]
        projected = playback.projection.transform_points(GEOGRAPHIC, flight.samples[:, 1], flight.samples[:, 2])
        np.testing.assert_allclose(heads[0, :2], projected[:, :2].mean(axis=0))
        self.assertEqual(heads[0, 2], 30000)
        self.assertEqual(len(frame_data(playback, 61, 600)[4]), 0)

    def test_long_gaps_are_not_connected(self):
        playback = prepare_playback([self.flight(times=(0, 300))], self.boundary())
        self.assertEqual(len(playback.segments), 0)
        trails, _, _, observed, heads = frame_data(playback, 150, 600)
        self.assertEqual((len(trails), len(heads)), (0, 0))
        self.assertEqual(len(observed), 1)

    def test_exact_sample_time_has_only_one_head(self):
        playback = prepare_playback([self.flight(times=(0, 60, 120), longitudes=(-80, -79.5, -79))], self.boundary())
        self.assertEqual(len(frame_data(playback, 60, 600)[4]), 1)

    def test_short_and_single_point_legs_appear_as_observed_dots(self):
        flights = [self.flight(times=(5, 10)), self.flight(key="2", times=(17,), longitudes=(-80,))]
        playback = prepare_playback(flights, self.boundary())
        observed = frame_data(playback, 30, 600)[3]
        self.assertEqual(set(observed[:, 4].astype(int)), {0, 1})
        self.assertEqual(len(frame_data(playback, 700, 600)[3]), 0)

    def test_trails_stop_at_clock_and_start_at_trailing_window(self):
        flight = self.flight(times=(0, 120))
        playback = prepare_playback([flight], self.boundary())
        trails = frame_data(playback, 90, 30)[0]
        projected = playback.projection.transform_points(GEOGRAPHIC, flight.samples[:, 1], flight.samples[:, 2])[:, :2]
        np.testing.assert_allclose(trails[0, 0], projected[0] + 0.5 * (projected[1] - projected[0]))
        np.testing.assert_allclose(trails[0, 1], projected[0] + 0.75 * (projected[1] - projected[0]))

    def test_default_frames_cover_every_observation_and_midnight(self):
        times = frame_times(30, 600)
        self.assertEqual(len(times), 2881)
        self.assertEqual((times[0], times[-1]), (0, DAY_SECONDS))
        for timestamp in (0, 0.001, 17, 43199.999, 86399.999):
            next_frame = times[np.searchsorted(times, timestamp)]
            self.assertLessEqual(next_frame - timestamp, 30)
        for step, trail in ((0, 600), (-1, 600), (30, 10), (30, float("nan"))):
            with self.assertRaises(ValueError):
                frame_times(step, trail)

    def test_marker_colors_match_fixed_altitude_legend(self):
        playback = prepare_playback([self.flight()], self.boundary())
        with patch("animate_zob_day.shapereader.natural_earth"), \
                patch("animate_zob_day.shapereader.Reader") as reader:
            reader.return_value.records.return_value = []
            figure, artists, update = make_figure(playback, self.day, "America/Detroit", 600)
        try:
            update(30)
            for scatter in artists[1:3]:
                self.assertEqual(scatter.get_cmap().name, "turbo")
                self.assertEqual(scatter.get_clim(), (0, 50000))
            self.assertEqual(artists[4].get_text(), "1 active tracked flights")
            self.assertIn("KBUF > KCMH", artists[5].get_text())
            self.assertEqual(artists[7].get_text(), "1")
            self.assertEqual(sum(label.get_visible() for label in artists[7:]), 1)
            gids = {collection.get_gid() for collection in figure.axes[0].collections}
            self.assertTrue({"ARTCC_ZOB_ARTCC_H", "ARTCC_ZOB_ARTCC_L"}.issubset(gids))
            self.assertTrue(any("ZOB | CLEVELAND ARTCC" in text.get_text() for text in figure.texts))
        finally:
            plt.close(figure)

    def test_artcc_detection_accepts_another_center_and_sensor_prefixes(self):
        self.write_rows([self.point(source="0/ZAU"),
                         self.point("2023-12-22 12:01:00.000", source=" zau ")])
        flights = load_flights(self.input, self.day)
        self.assertEqual(flights[0].facility, "ZAU")
        self.assertEqual(self.boundary("ZAU").name, "CHICAGO")
        self.assertEqual(reporting_facility("12/zob"), "ZOB")
        for value in ("", "ZOB/ZAU", "../ZOB", " ZOB OR ZAU "):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reporting_facility(value)

    def test_fuel_sidebar_replaces_flight_ids_and_stops_at_current_time(self):
        playback = prepare_playback([self.flight()], self.boundary())
        curve = np.array([[0, 0], [3600, 2000], [43200, 8000], [DAY_SECONDS, 12000]])
        with patch("animate_zob_day.shapereader.natural_earth"), \
                patch("animate_zob_day.shapereader.Reader") as reader:
            reader.return_value.records.return_value = []
            figure, artists, update = make_figure(playback, self.day, "America/Detroit", 600, curve)
        try:
            self.assertEqual(len(artists), 9)
            fuel_line = artists[7]
            self.assertEqual(fuel_line.get_gid(), "daily_cumulative_fuel")
            for elapsed in (0, 17, 3600, 43200, DAY_SECONDS):
                update(elapsed)
                expected = np.interp(elapsed, curve[:, 0], curve[:, 1]) / 1000
                self.assertAlmostEqual(fuel_line.get_xdata()[-1], elapsed / 3600)
                self.assertAlmostEqual(fuel_line.get_ydata()[-1], expected)
                self.assertTrue(np.all(fuel_line.get_xdata() <= elapsed / 3600))
                self.assertEqual(artists[5].get_text(), f"{expected:,.1f} tonnes")
            labels = "\n".join(text.get_text() for text in figure.texts)
            self.assertIn("CUMULATIVE FLEET FUEL", labels)
            self.assertNotIn("LOWEST 8 ACTIVE FLIGHTS", labels)
            self.assertNotIn("TEST1", labels)
            self.assertEqual(len(figure.axes[0].texts), 0)
            gids = {collection.get_gid() for collection in figure.axes[0].collections}
            self.assertTrue({"ARTCC_ZOB_ARTCC_H", "ARTCC_ZOB_ARTCC_L"}.issubset(gids))
        finally:
            plt.close(figure)

    def test_mixed_facilities_cannot_merge_the_same_flight_key(self):
        self.write_rows([self.point(), self.point(source="ZAU")])
        with self.assertRaisesRegex(ValueError, "Multiple reporting facilities"):
            load_flights(self.input, self.day)

    def test_boundary_selection_is_by_facility_and_artcc_type(self):
        extra = self.feature()
        extra["properties"]["TYPE_CODE"] = "FIR"
        self.boundary_data["features"].append(extra)
        self.save_boundaries()
        boundary = self.boundary()
        self.assertEqual(boundary.name, "CLEVELAND")
        self.assertEqual({chart for chart, _ in boundary.contours}, {"ARTCC_H", "ARTCC_L"})
        self.assertEqual(len(boundary.contours), 2)
        self.assertEqual(boundary.effective_from, date(2023, 11, 30))
        for source in ("D21", "USA", "ZZZ"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "No ARTCC outline"):
                self.boundary(source)

    def test_complete_contour_fits_extent_and_projection_recenters(self):
        chicago = self.boundary("ZAU")
        playback = prepare_playback([self.flight(facility="ZAU", longitudes=(-89, -88))], chicago)
        west, east, south, north = playback.extent
        self.assertLess(west, -94)
        self.assertGreater(east, -85)
        self.assertLess(south, 40)
        self.assertGreater(north, 45)
        self.assertAlmostEqual(playback.projection.proj4_params["lon_0"], (west + east) / 2)
        with self.assertRaisesRegex(ValueError, "do not match"):
            prepare_playback([self.flight()], chicago)

    def test_invalid_missing_or_nonpolygon_outlines_fail(self):
        for geometry in (
            None,
            {"type": "Point", "coordinates": [-80, 42]},
            {"type": "Polygon", "coordinates": []},
            {"type": "Polygon", "coordinates": [[[-85, 39], [-76, 44], [-85, 44], [-76, 39], [-85, 39]]]},
        ):
            with self.subTest(geometry=geometry):
                self.boundary_data["features"][0]["geometry"] = geometry
                self.save_boundaries()
                with self.assertRaises(ValueError):
                    self.boundary()

    def test_multiple_features_and_multipolygons_are_preserved(self):
        first = self.boundary_data["features"][0]
        second = self.feature(bounds=(-87, 39, -86, 40))
        first["geometry"] = {
            "type": "MultiPolygon",
            "coordinates": [first["geometry"]["coordinates"], second["geometry"]["coordinates"]],
        }
        self.boundary_data["features"].append(self.feature(bounds=(-90, 39, -89, 40)))
        self.save_boundaries()
        boundary = self.boundary()
        self.assertEqual(len(boundary.contours), 3)
        self.assertEqual(boundary.contours[0][1].geom_type, "MultiPolygon")
        self.assertEqual(len(boundary.contours[0][1].geoms), 2)

    def test_invalid_geometry_for_another_facility_does_not_block_this_artcc(self):
        self.boundary_data["features"][2]["geometry"] = None
        self.save_boundaries()
        self.assertEqual(len(self.boundary().contours), 2)
        with self.assertRaisesRegex(ValueError, "requires polygon"):
            self.boundary("ZAU")

    def test_unrecognized_chart_classification_is_not_guessed(self):
        self.boundary_data["features"][0]["properties"]["LOCAL_TYPE"] = "UNKNOWN"
        self.save_boundaries()
        with self.assertRaisesRegex(ValueError, "unsupported ARTCC chart classification"):
            self.boundary()

    def test_projected_or_truncated_geojson_is_rejected(self):
        self.boundary_data["crs"] = {"type": "name", "properties": {"name": "EPSG:3857"}}
        self.save_boundaries()
        with self.assertRaisesRegex(ValueError, "WGS84"):
            self.boundary()
        del self.boundary_data["crs"]
        self.boundary_data["exceededTransferLimit"] = True
        self.save_boundaries()
        with self.assertRaisesRegex(ValueError, "complete GeoJSON"):
            self.boundary()

    def test_explicit_missing_boundary_path_never_downloads_a_replacement(self):
        with patch("animate_zob_day.fetch_json") as fetch:
            with self.assertRaises(FileNotFoundError):
                load_artcc_boundary("ZOB", self.directory / "missing.geojson")
            fetch.assert_not_called()

    def test_api_error_is_not_treated_as_an_empty_boundary(self):
        response = io.StringIO(json.dumps({"error": {"code": 429, "message": "Retry after 60 sec."}}))
        with patch("animate_zob_day.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "429"):
                fetch_json("https://example.invalid")

    def test_current_faa_download_is_cached_with_effective_dates(self):
        cache = self.directory / "cache" / "artcc.geojson"
        item = {"description": "<p>Current Effective Date: 0901Z 29 Oct 2026 to 0901Z 24 Dec 2026</p>"}
        with patch("animate_zob_day.DEFAULT_BOUNDARIES", cache), \
                patch("animate_zob_day.fetch_json", side_effect=[item, self.boundary_data]) as fetch:
            first = load_artcc_boundary("ZOB")
            second = load_artcc_boundary("ZAU")
            self.assertEqual(fetch.call_count, 2)
        self.assertEqual(first.effective_from, date(2026, 10, 29))
        self.assertEqual(second.effective_to, date(2026, 12, 24))
        self.assertTrue(cache.exists())

    def test_incomplete_download_never_creates_a_cache(self):
        cache = self.directory / "cache" / "artcc.geojson"
        item = {"description": "Current Effective Date: 0901Z 29 Oct 2026 to 0901Z 24 Dec 2026"}
        self.boundary_data["exceededTransferLimit"] = True
        with patch("animate_zob_day.DEFAULT_BOUNDARIES", cache), \
                patch("animate_zob_day.fetch_json", side_effect=[item, self.boundary_data]):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_boundary_dataset()
        self.assertFalse(cache.exists())

    def test_another_artcc_title_and_boundary_vintage_warning_are_visible(self):
        boundary = self.boundary("ZAU")
        boundary.effective_from, boundary.effective_to = date(2026, 10, 29), date(2026, 12, 24)
        playback = prepare_playback([self.flight(facility="ZAU", longitudes=(-89, -88))], boundary)
        with patch("animate_zob_day.shapereader.natural_earth"), \
                patch("animate_zob_day.shapereader.Reader") as reader:
            reader.return_value.records.return_value = []
            figure, _, _ = make_figure(playback, self.day, "America/Chicago", 600)
        try:
            text = "\n".join(label.get_text() for label in figure.texts)
            self.assertIn("ZAU | CHICAGO ARTCC", text)
            self.assertIn("2026-10-29", text)
            self.assertIn("does NOT cover 2023-12-22", text)
        finally:
            plt.close(figure)

    def test_unspecified_and_invalid_boundary_periods_are_not_guessed(self):
        self.boundary_data.pop("faa_metadata")
        self.save_boundaries()
        self.assertIsNone(self.boundary().effective_from)
        for metadata in (
            {"effective_from": "2023-11-30"},
            {"effective_from": "2023-12-28", "effective_to": "2023-11-30"},
        ):
            self.boundary_data["faa_metadata"] = metadata
            self.save_boundaries()
            with self.assertRaises(ValueError):
                self.boundary()

    def test_output_protection_and_invalid_settings(self):
        self.write_rows([self.point()])
        output = self.directory / "video.mp4"
        output.write_bytes(b"existing video")
        with self.assertRaises(FileExistsError):
            render_day(self.input, output, self.day)
        self.assertEqual(output.read_bytes(), b"existing video")
        for options in ({"fps": 0}, {"trail_minutes": 0}, {"seconds_per_frame": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                render_day(self.input, self.directory / "new.mp4", self.day, **options)
        with self.assertRaisesRegex(ValueError, "DST-transition"):
            render_day(self.input, self.directory / "new.mp4", date(2023, 11, 5))


if __name__ == "__main__":
    unittest.main()
