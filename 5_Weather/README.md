# CWAM pre-TRACON weather grid

`cwam_weather.py` fetches N0Q reflectivity and MRMS EchoTop data, tiles the
pre-TRACON region around an airport into equal-size cells, and computes a
convective-weather deviation probability for every cell at every flight level.
Output is one CSV per (timestamp, flight level).

Times are UTC throughout — no timezone conversion happens anywhere.

## MILP integration

`3_MILP_Serra/2.1_MILP/2.1.2_MILP/milp_multiple_Debug.py` contains the
weather loading and exclusion constraints directly. Its `load_weather_frames`
function invokes `cwam_weather.run_cwam` with
`return_data=True`. This replaces the fake weather CSV inputs. The weather
window follows the earliest selected flight entry through the **full model
horizon** (48,000 seconds by default), rather than only the recorded
flight exits. Naive flight timestamps are assumed UTC. Frames use five-minute
UTC boundaries and requests are split across calendar dates when necessary.

For a shorter square-separation test with plots, run from the repository root
using a Python environment with the MILP/weather dependencies and a working
Gurobi license:

```bash
python3 3_MILP_Serra/2.1_MILP/2.1.2_MILP/milp_multiple_Debug.py \
  --shape square --horizon-minutes 60 --time-limit 300 --plot \
  --weather-start 2026-09-03T19:00:00Z
```

`--weather-start` replays weather from that UTC time while keeping the input
flights' positions and relative entry times. Omit it to use weather matching
the input flight dates. The input remains the first two rows of
`2.1.1_Input_Data/entry_exit_points.csv`. The horizon is extended if necessary
to cover their recorded exits. `--shape square` selects square aircraft
waypoint/end-cap clearance in the existing capsule formulation; weather
exclusion always uses the CWAM cell rectangles.

Outputs default to `3_MILP_Serra/2.3_Outputs_and_Results/weather_debug/`;
override with `--output-dir`. A feasible solution produces `solution.csv`.
`--plot` also produces `trajectories.png` (route map with the union of weather
across all timestamps, plus altitude profiles) and `weather_trajectories.gif`
(current weather and moving aircraft, with square clearance outlines).
Both maps include a blue, lightly shaded TRACON polygon using the same
STAR-fix convex hull returned by CWAM for its weather region.
The GIF ends at the first model waypoint when all aircraft have reached the
TRACON polygon, including its boundary. Each displayed aircraft stops at its
own arrival waypoint. If a track never intersects the polygon, its first
arrival at its selected terminal fix is used instead. Later holding steps
remain in the solution CSV and static overview but are omitted from the GIF.
A time-limited solve exports a feasible incumbent if one exists and reports
that optimality has not been proven. Without an incumbent, no new results or
plots are written; existing files in the output directory remain from prior
runs. `--time-limit` limits solving, not weather downloading or model building.

Configure `WEATHER_AIRPORT`, `WEATHER_STEP_MINUTES`, `WEATHER_EPS_DEG`, and
`CWAM_OPTIONS` near the top of the MILP script. Other CWAM settings, including
the evaluated flight levels and the probability table, retain their defaults
unless supplied in `CWAM_OPTIONS`.

For each timestamp, the MILP script takes the union of cells marked `infeasible`
at **any evaluated flight level**, removes duplicate bounds, and blocks their
latitude/longitude rectangles at every altitude. This is a conservative
horizontal exclusion; the optimizer cannot climb over a blocked cell. Only
cells supplied by CWAM's pre-TRACON grid are covered.

Gurobi indicator constraints require a waypoint to lie south, north, west,
or east of each blocked rectangle, with `WEATHER_EPS_DEG` clearance. Each
straight segment is split at weather-frame boundaries, and both ends of each
portion must satisfy the **same exterior side**. This prevents crossings
between waypoints but can exclude some clear diagonal paths around corners.
Snapshots are held constant until the next frame. Constraints apply from
each flight's entry through the horizon, including positions at the final fix.
Missing snapshots or flight levels stop the run; they never silently disable
weather constraints. Individual missing raster cells follow CWAM's configured
`missing_data_policy` (default `infeasible`).

Run the offline weather/geometry checks from the repository root:

```bash
python3 -m unittest discover -s 5_Weather -p 'test_cwam_milp.py' -v
```

These checks evaluate the generated indicator disjunctions on fixed
trajectories and mock CWAM results, without downloads or a Gurobi license.

## Quick start

```python
from cwam_weather import run_cwam

r = run_cwam(date="2026-09-03", start_time="19:00", end_time="20:00", apt_code="DTW")
r["summary"]      # one row per (timestamp, flight level)
r["csv_files"]    # paths written to disk
```

Keep the per-cell DataFrames in memory instead of re-reading the CSVs:

```python
r = run_cwam("2026-09-03", "19:00", "20:00", "DTW", return_data=True)
df = r["data"][(pd.Timestamp("2026-09-03 19:00:00"), "FL350")]
```

From the command line:

```bash
python cwam_weather.py --date 2026-09-03 --start 19:00 --end 20:00 --apt DTW
```

Only the first four arguments (`date`, `start_time`, `end_time`, `apt_code`)
can be given positionally / without a flag. Every other argument is
keyword-only in Python (`name=value`) and `--flag value` on the CLI — run
`python cwam_weather.py --help` for the full list.

## Required inputs

| Name | Type | Example | Meaning |
|---|---|---|---|
| `date` | str | `"2026-09-03"` | `"YYYY-MM-DD"`, UTC |
| `start_time` | str | `"19:00"` | `"HH:MM"`, UTC, inclusive |
| `end_time` | str | `"20:00"` | `"HH:MM"`, UTC, inclusive |
| `apt_code` | str | `"DTW"` | Airport id. Also tested: `ORD`, `ATL`, `DFW`, `MSP` |

## Other inputs (all keyword-only, all optional)

**Time stepping**
| Name | Default | Meaning |
|---|---|---|
| `step_minutes` | `5` | Spacing between frames within the time window |

**Region of interest**
| Name | Default | Meaning |
|---|---|---|
| `radius_km` | `300` | Pre-TRACON ROI radius around the airport |
| `cell_size_km` | `16` | Analysis grid cell size (CWAM2 uses 16; some configs use 8) |
| `min_cell_area_fraction` | `0.05` | Drop cells with less than this fraction of their area inside the ROI |

**Predictors**
| Name | Default | Meaning |
|---|---|---|
| `dbz_threshold` | `35.0` | Reflectivity cutoff (dBZ) defining `coverage_pct` |
| `echo_top_stat` | `"p90"` | How per-cell echo-top height is summarized: `"p90"` (90th percentile, robust to outlier pixels) or `"max"` |

**Altitude sweep** — either min/max/step, or an explicit list
| Name | Default | Meaning |
|---|---|---|
| `altitude_min_ft` | `11000` | Sweep start |
| `altitude_max_ft` | `40000` | Sweep end |
| `altitude_step_ft` | `2000` | Sweep spacing → FL110...FL390 (15 levels) |
| `altitudes_ft` | `None` | Explicit list, e.g. `[30000, 35000, 39000]` — overrides min/max/step |

**Feasibility rule**
| Name | Default | Meaning |
|---|---|---|
| `mask_probability_threshold` | `0.6` | `p_deviation >=` this ⇒ cell marked `infeasible` |
| `missing_data_policy` | `"infeasible"` | `"infeasible"`: no radar data ⇒ blocked (conservative). `"passable"`: no data ⇒ treated as clear |

**Data sourcing**
| Name | Default | Meaning |
|---|---|---|
| `echo_top_time_tolerance_min` | `5` | Max time gap allowed when pairing an EchoTop file to an N0Q timestamp |
| `download_workers` | `8` | Parallel download threads |
| `prob_table_csv` | `None` | Path to a *calibrated* P(deviation) table. If omitted, uses the uncalibrated placeholder formula below (a warning prints) |

**Paths** — all default to folders next to `cwam_weather.py`, not your working directory
| Name | Default | Meaning |
|---|---|---|
| `data_dir` | `None` | → `data/` (downloaded N0Q/EchoTop + NASR reference tables) |
| `output_dir` | `None` | → `output/` (CSVs + manifest) |
| `plot_dir` | `None` | → `output_dir/plots` |

**Plots**
| Name | Default | Meaning |
|---|---|---|
| `save_plots` | `False` | Write PNGs (N0Q, EchoTop, coverage, P(deviation), infeasible mask, ROI/hull) |
| `plot_flight_levels` | `None` | Restrict per-level plots to specific FLs, e.g. `["FL350"]`. `None` = every level |

**Output shape**
| Name | Default | Meaning |
|---|---|---|
| `return_data` | `False` | Also return every per-cell DataFrame in memory, keyed by `(timestamp, flight_level)` |
| `verbose` | `True` | Progress printing |

## Output

One CSV per timestamp × flight level, named so the filename alone fully
describes the run that produced it:

```
DTW_20260903T1900Z_FL350_r300km_c16km_35dBZ_p90_pt060_infeas.csv
     └time (UTC) └FL   └radius └cell  └dBZ  └stat └P thr └missing-data policy
```

Each CSV has one row per grid cell, with columns including `cell_id`,
`centroid_lat/lon`, `coverage_pct`, `etop_stat_ft`, `dz_ft`, `p_deviation`,
`data_missing`, and `infeasible`. A `manifest_*.json` alongside the CSVs
records every parameter used, the cell grid dimensions, and every file
written — full provenance for reproducing the run later.

## The probability model

Each cell gets two altitude-independent predictors, computed once per
timestamp:

- **`coverage_pct`** — % of N0Q pixels in the cell with reflectivity ≥ `dbz_threshold`
- **`etop_stat_ft`** — the `echo_top_stat` (p90 or max) of 18 dBZ echo-top height across the cell's radar-covered pixels

At each flight level, altitude is compared against the storm tops:

```
dz_ft = altitude_ft - etop_stat_ft      # negative = flying at/below the storm top
```

`p_deviation`, the modeled probability a pilot deviates around weather in
that cell at that altitude, is a 2D lookup table over `(dz_kft, coverage_pct)`,
bilinearly interpolated and clamped at the table edges. The **default table**
(`make_default_prob_table()`) is an analytic placeholder:

```
P = (coverage / 100)^0.35 * sigmoid(-dz_kft / 3.5)
```

- `(coverage/100)^0.35` — rises with more storm coverage, but sub-linearly: even modest coverage pushes probability up substantially
- `sigmoid(-dz_kft / 3.5)` — near 1 when flying at/below the storm top (`dz` very negative), drops toward 0 well above it, with the transition spread over roughly ±3.5 kft around `dz = 0`

**The probability model is swappable.** This formula is the current default —
an analytic approximation shaped to qualitatively resemble Figure 8c in
DeLaura et al.'s CWAM2 paper, not fit to real deviation data. Pass
`prob_table_csv=` with a differently-defined table (any `(dz_kft, coverage_pct) -> P`
surface, e.g. one calibrated against real encounter data) to use it instead,
with no other code changes required — `p_deviation` is always a bilinear
lookup against whatever table is loaded. Worth noting: the paper's own
coverage predictor is echo-top-based (% of echo tops ≥ 30 kft), not
reflectivity-based like `coverage_pct` here, so the `0.35` exponent and the
`35 dBZ` reflectivity threshold are unrelated numbers, not a tuned pair.

A cell is marked **`infeasible`** at a flight level if:

```
infeasible = (p_deviation >= mask_probability_threshold)
             OR (data_missing AND missing_data_policy == "infeasible")
```

`p_deviation` is written to every CSV alongside the binary `infeasible` flag,
so a downstream optimizer (e.g. a MILP) can use either the continuous
probability as a cost, or the binary flag as a hard constraint.
