# TRACON Arrival Flow Analysis Tools

A collection of Python scripts for analyzing terminal-area (TRACON) arrival traffic using FAA/NASA SHERLOCK data exports (RD and IFF daily CSV files). The tools quantify how arrivals distribute across TRACON entry fixes, which airlines use which fixes, and how long flights spend transiting the airspace just outside the TRACON boundary. Developed for the LATTICE / FAST research effort (AERO 590, University of Michigan), with case studies of C90 (Chicago) and D21 (Detroit) TRACONs using December 2023 data.

## Contents

| File | Purpose |
|---|---|
| `fix_concentration.py` | Count arrivals per TRACON entry fix and plot a bar chart |
| `fix_concentration_per_day.py` | Break down entry-fix usage by hour of day (stacked bar charts) |
| `fix_concentration_airlien.py` | Break down entry-fix usage by airline (stacked bar charts) |
| `airline_counter.py` | Rank airlines by arrival flight count (top-5 bar chart) |
| `preTRACON_transit_time_calc.py` | Compute pre-TRACON transit times from IFF trajectory data (config-driven, multi-scenario) |
| `preTRACON_time_calc_config.toml` | Configuration file for the transit-time calculator |

The first four scripts are standalone one-off analyses of RD (flight record) data. The last two form a configurable batch pipeline that joins RD flight records to IFF track-point data.

## Requirements

Python 3.11+ (the transit-time calculator uses the standard-library `tomllib`), plus:

```
pandas
numpy
matplotlib
shapely        (transit-time calculator only)
pyproj         (transit-time calculator only)
```

## Common inputs and conventions

All scripts read SHERLOCK daily CSV exports named like `RD_C90_20231201_..._.csv` (flight records) and, for the transit-time pipeline, matching `IFF_<facility>_<YYYYMMDD>_...csv` trajectory files. Shared conventions:

- **Part 121 filter** — the three `fix_concentration*` scripts keep only aircraft types listed in an external text file (`part_121_acft_list.txt`, one ICAO type per line, `#` comments allowed), restricting the analysis to scheduled air-carrier equipment.
- **Airline cleanup** — rows with `Airline == "?"` (unknown) or airlines starting with `GA.` (general aviation) are dropped everywhere.
- **Entry fix** — the RD `Orig` column is treated as the TRACON entry fix for arrivals; each script excludes a set of non-fix / satellite-airport values via an `ORIG_EXCLUDE` set.
- **Paths are hard-coded** in a "USER SETTINGS" block at the top of each script (or supplied via the TOML config for the transit-time pipeline). Edit those before running.

---

## fix_concentration.py

**What it does:** Counts how many filtered arrivals entered the TRACON over each entry fix, across a whole month of RD files.

Workflow: recursively reads every CSV under `root_folder`, concatenates them, normalizes the `Airline`/`AcType`/`Dest`/`Orig` string columns, then filters to Part 121 aircraft types, drops unknown/GA airlines, keeps a single destination airport (currently `Dest == "ORD"`), and removes excluded `Orig` values. It then counts occurrences of each remaining `Orig` value.

Outputs (written to `output_folder`):

- `combined_all_csvs.csv` — the raw concatenated dataset
- `filtered_dataset.csv` — rows surviving all filters
- `orig_counts.csv` — arrivals per entry fix
- `orig_counts_bar_chart.png` — bar chart of arrivals per entry fix

It also prints row counts after each filter stage and a debug list of aircraft types not found in the Part 121 list.

## fix_concentration_per_day.py

**What it does:** Same data loading and filtering as `fix_concentration.py`, but adds a temporal dimension: it parses `StartDate` + `StartTime` into a datetime, extracts the UTC hour, and pivots the data into a 24-hour × entry-fix table.

Outputs:

- `orig_counts_by_hour.csv` — arrivals per entry fix for each hour of day (all 24 hours present, zero-filled)
- `stacked_counts_by_hour.png` — stacked bar chart of arrival counts by hour, colored by entry fix
- `stacked_percent_by_hour.png` — same chart normalized to percent share per hour

Configured for C90 / ORD arrivals; axis labels note that hours are UTC (local Chicago time is UTC−6).

## fix_concentration_airlien.py

**What it does:** Same loading/filtering pattern, but pivots by entry fix × airline to show which carriers use which entry fixes. Configured for D21 TRACON / DTW arrivals (December 2023), with a D21-specific `ORIG_EXCLUDE` set (satellite fields and `DA0`–`DA9` codes).

Fixes are sorted by total traffic; only the top 10 airlines are kept individually, with everything else lumped into an `OTHER` column.

Outputs:

- `fix_airline_counts.csv` / `fix_airline_percent.csv` — counts and percent share per fix × airline
- `fix_airline_counts.png` — stacked bar chart (counts) of airline usage by entry fix
- `fix_airline_distribution.png` — stacked bar chart (percent) of airline mix at each fix


## airline_counter.py

**What it does:** Aggregates RD files matching a glob pattern (e.g. `RD_C90_*.csv`) and ranks airlines by arrival flight count, reporting the top 5.

Unlike the other RD scripts, its `load_rd()` helper handles both headered files (first cell `lKey`) and raw headerless exports (columns addressed by index and renamed: 13 → `AcId`, 21 → `Airline`, 26 → `Dest`). Filtering keeps arrivals to the configured destination set (currently `MDW`/`ORD`), dropping GA and unknown airlines. No aircraft-type filter is applied here.

Outputs and console report:

- Per-file row counts before/after filtering
- Combined totals: flights, unique callsigns, unique airlines, full per-airline counts
- `airline_distribution_c90.png` — top-5 airlines bar chart with value labels and a totals box

## preTRACON_transit_time_calc.py + preTRACON_time_calc_config.toml

**What it does:** Measures how long arrivals spend in the "pre-TRACON" region — a geodesic circle (default 300 km radius) around the arrival airport, minus the TRACON polygon defined by an ordered ring of STAR entry fixes. For each flight it finds the first and last track point inside that annulus and reports the elapsed transit time.

### Pipeline

1. **Config load & validation** — the TOML file defines base settings plus optional `[[scenarios]]` blocks that override any subset of them (data/output paths, region center, fix ring and coordinates, destination airports, airline include/exclude prefixes, landing-runway prefixes, unknown-runway handling). Unknown keys and inconsistent fix definitions fail fast with clear errors.
2. **File pairing** — IFF and RD files in each `data_dir` are paired by facility + date parsed from the filename; unpaired files produce warnings and are skipped.
3. **Efficient grouped passes** — scenarios sharing the same `data_dir` and region geometry are grouped so each day's RD file is read once and the IFF file is scanned once (in chunks, default 500k rows) for the union of all scenarios' flight keys. This makes running ~30 scenarios (e.g. per-airline × per-flow) roughly as cheap as one.
4. **Per-flight transit computation** — track points (`recType == 3`) inside the pre-TRACON annulus are grouped by flight key; entry/exit times, positions, and transit minutes are computed. Flights with a single in-region point are dropped.
5. **Full-trajectory filter** — a flight is kept for statistics only if it *entered* within `boundary_tolerance_km` of the outer circle **and** *exited* within the same tolerance of the TRACON boundary, excluding partial tracks.
6. **Outputs per scenario:**
   - `pre_tracon_transit_times_all_days.csv` — per-flight entry/exit records and transit times (all flights, flagged full/partial)
   - `pre_tracon_transit_stats_all_days.csv` — box-and-whisker statistics (mean, std, quartiles, whiskers, outlier count)
   - `pre_tracon_transit_per_day_summary.csv` — daily n/mean/median/Q1/Q3
   - `pre_tracon_transit_boxplot_all_days.png` — annotated boxplot of transit times
   - `pre_tracon_entry_exit_verification_all_days.png` — map of the region, fix ring, and kept/excluded entry and exit points, for sanity-checking the geometry

### Usage

```bash
python preTRACON_transit_time_calc.py [path/to/config.toml]
```

With no argument, the `CONFIG_PATH` constant at the top of the script is used (so it also runs from an IDE like Spyder).

### The supplied config

`preTRACON_time_calc_config.toml` defines a **DTW north flow** base case (ZOB center data, 7-fix D21 boundary ring, runways 21/22) plus ~30 ORD scenarios on ZAU data with a 6-fix C90 ring (JVL, WYNDE, WATSN, VEECK, TRTLL, SHAIN): all-traffic ORD/MDW, then ORD split by flow direction (west 27/28, east 09/10, north 04, south 22), and the same flow split repeated for individual carriers — United (UAL), SkyWest (SKW), American (AAL), Air Wisconsin (AWI), and Envoy (ENY). All scenarios share one data directory and region, so the whole set runs in a single data pass.

## Notes and caveats

- Hard-coded paths reference two different machines (Windows `C:\Users\anomi\...` in the fix-concentration scripts, macOS `/Users/anomi/...` in the others); update them for your environment.
- In `fix_concentration.py`, the comment above the destination filter says `DTW` but the code filters `Dest == "ORD"` — the code is authoritative; edit the constant for the airport you want.
- Similarly, `airline_counter.py`'s comments mention "DTW arrivals" and "ATL arrivals" while the code keeps `MDW`/`ORD`; trust the code.
- All timestamps are UTC. Chicago local time is UTC−6 (CST) for the December 2023 datasets.
- Chart windows open via `plt.show()`; the scripts are meant to be run interactively or from an IDE.
