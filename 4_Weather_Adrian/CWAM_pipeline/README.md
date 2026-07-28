# CWAM Weather Pipeline (N0Q + MRMS Echo Tops, pre-TRACON)

A configurable pipeline implementing the two-predictor structure of the MIT
Lincoln Laboratory Convective Weather Avoidance Model (CWAM2, DeLaura et al.),
combining:

1. **N0Q composite reflectivity** from the Iowa Environmental Mesonet archive
   (same PNG + WLD source used in `wx_grid_creator_3.py`), and
2. **NOAA MRMS `EchoTop_18`** — height of the 18 dBZ echo top, the echo-top
   definition used by CWAM — from the public NOAA S3 archive
   (`noaa-mrms-pds`, ~2 min cadence, CONUS, archived back to Oct 2020).

The pre-TRACON region (geodesic circle around the airport minus the TRACON
polygon from ordered STAR fixes) is tiled with **equally sized square cells**
(default 16×16 km, per CWAM2). For each cell and timestamp the pipeline
computes:

| Predictor | Meaning |
|---|---|
| `coverage_pct` | % of N0Q pixels in the cell with reflectivity ≥ `dbz_threshold` |
| `dz_ft` | altitude − echo-top statistic (90th percentile by default) |

and looks up **P(deviation)** in a 2D probability table
`P(dz, coverage)` with bilinear interpolation.

The WAF is evaluated at **every altitude level** in a manually configured
range (`altitude_min_ft` / `altitude_max_ft` / `altitude_step_ft`, or an
explicit `altitudes_ft` list), producing one full product set per level per
timestamp (tagged `FLxxx`), so a downstream MILP can trade off routes across
altitudes. Coverage % and echo-top statistics are altitude-independent and
computed once per timestamp; only the dz → probability lookup repeats.

## Install

```bash
pip install -r requirements.txt
```

(`rasterio` wheels include GDAL with the GRIB driver, which reads the MRMS
`.grib2` files directly — no eccodes/pygrib needed.)

## Run

```bash
python -m cwam_pipeline.run --config config.yaml
```

Options:

- `--skip-download` — reuse files already in `data_dir`
- `--echotop-local-dir DIR` — use pre-downloaded EchoTop files instead of S3
  (filenames must contain `_YYYYMMDD-HHMMSS`)

## Configuration (`config.yaml`)

- `case` — date, UTC time window, 5-min step (N0Q archive cadence)
- `region` — pre-TRACON definition: center lat/lon, radius (km), STAR fixes,
  ordered boundary fix names. Defaults reproduce the DTW region from
  `wx_grid_creator_3.py`. Point at any airport by editing these values.
- `analysis` — cell size, dBZ threshold, altitude limits (min/max/step or
  explicit list), echo-top statistic (`p90`/`max`), N0Q↔EchoTop pairing
  tolerance, probability-table path, mask threshold, `overlay_binary`,
  `missing_data_policy` (see below)
- `sources`/`paths` — URLs, download workers, data/output directories

## Outputs (per timestamp and per altitude level, under `output_dir`)

`<t>` is the timestamp (`YYYYMMDDHHMM`), `FLxxx` the altitude tag
(FL350 = 35,000 ft):

| File | Contents |
|---|---|
| `cells/cells_<t>_FLxxx.csv` | one row per grid square: cell id/row/col, centroid & bounds, `n_pixels_n0q`, `coverage_pct`, `etop_stat_ft`, `altitude_ft`, `dz_ft`, `p_deviation`, `data_missing`, **`infeasible`** |
| `waf/waf_<t>_FLxxx.tif` | WAF probability raster painted onto the N0Q grid (EPSG:4326, float32, LZW). NaN where a pixel is outside the grid *or* its cell had no valid EchoTop data — these two cases are not distinguishable from this file alone |
| `masks/mask_<t>_FLxxx.npz` | keys `binary_mask` (per `missing_data_policy`, see below), `altitude_ft`, `data_missing_mask` (raw per-pixel data-gap flag, independent of policy) — drop-in for the Gurobi/MILP workflow |
| `regions/regions_<t>_FLxxx.csv` | one row per **infeasible grid square** (no merging): `region_id, n_pixels, min/max lat/lon, centroid, coverage_pct, dz_ft, p_deviation, data_missing` (first 8 columns match the legacy `t00.csv` schema) |
| `overlays/overlay_<t>_FLxxx.png` | black-background quick-look with pre-TRACON boundary (blue), TRACON polygon (red), and data-gap cells in **mid-gray** (see below) |

## Missing-data handling

A cell has no valid EchoTop statistic when it sits in an MRMS radar-coverage
gap, or is a thin sliver at the pre-TRACON boundary too small to contain any
MRMS pixel center. That cell's `dz_ft` and `p_deviation` are correctly `NaN`
in `cells_*.csv` — but a *decision* still has to be made for the mask the
MILP consumes, and "no data" is not the same claim as "confirmed clear."

`analysis.missing_data_policy` controls that decision:

- `infeasible` (default) — missing-data cells are folded into the binary
  mask and the regions CSV as blocked, on the principle that an unknown
  weather state should not be assumed safe to fly through.
- `passable` — missing-data cells are excluded from the mask (legacy
  behavior — data gaps silently read as clear weather). Not recommended for
  anything safety-relevant; useful mainly for comparing against earlier
  pipeline output.

Every cell's `data_missing` flag is preserved in `cells_*.csv` and
`regions_*.csv` regardless of policy, and the mask npz also carries a raw
`data_missing_mask` independent of the chosen policy, so you can always
recover which blocked cells were "confirmed weather" versus "no data" if the
MILP needs to treat them differently downstream. The overlay PNGs paint
data-gap cells mid-gray (0.4) rather than black, so a gap is never visually
identical to confirmed-clear (black, P≈0) or confirmed-blocked (white, P≈1).

## The probability table

`prob_table_cwam2.csv` — rows are `dz` bin centers in **kft** (aircraft
altitude minus echo top; negative = flying below the tops), columns are
coverage-% bin centers. Values are P(deviation). Edit freely; any monotonic
bin spacing works, lookups are bilinearly interpolated and clamped at edges.

> **Important:** the shipped table is an *analytic approximation* shaped to
> match CWAM2 Figure 8c qualitatively
> (`P = (cov/100)^0.35 · sigmoid(−dz/3.5 kft)`). The paper does not publish
> numeric tables, and its coverage predictor is echo-top-based
> (% echo tops ≥ 30 kft) while this pipeline uses reflectivity coverage per
> your spec. Calibrate the table against your own encounter data (or digitized
> paper figures) before drawing research conclusions.

## Notes & conventions

- **N0Q index → dBZ**: `dBZ = index/2 − 32` (index 154 ⇔ 45 dBZ, matching
  `wx_grid_creator_3`). Index 0 = no echo; kept in the coverage denominator.
- **EchoTop_18 units**: MRMS encodes km MSL; converted to feet. Negative
  sentinel values: `−999` (no radar coverage) → NaN; other negatives
  (scanned, no 18 dBZ echo) → 0 ft, so `dz` is large and P ≈ 0.
- **Grid cells** are exact squares in a local azimuthal-equidistant projection
  centered on the airport, clipped to the pre-TRACON region; cells with < 5%
  of their area in the region are dropped (configurable).
- **Time pairing**: for each 5-min N0Q frame, the nearest EchoTop file within
  `echo_top_time_tolerance_min` is used (MRMS runs ~every 2 min).
- N0Q and EchoTop grids have different resolutions (~0.005° vs 0.01°); the
  per-cell aggregation makes co-registration unnecessary.
- Lightning is intentionally excluded, per project scope.
- **Unverified against real files.** The N0Q index→dBZ formula and the
  MRMS km-vs-meters unit heuristic in `rasters.py` were never checked
  against an actual downloaded PNG/GRIB2 file (no network in the dev
  sandbox). On your first real run, sanity-check one frame — e.g. print
  `np.unique(raw)` from the un-scaled EchoTop array and compare a known
  storm's reflectivity/top values against a public radar viewer — before
  trusting the numbers for anything decision-relevant.
- A thin ROI-boundary cell can pass the `min_cell_area_fraction` check by
  area yet still sample zero MRMS pixel centers (`all_touched=False`),
  giving it `data_missing=True` for a sampling reason rather than a true
  coverage gap. Harmless under the default `infeasible` policy (it's
  blocked either way) but worth knowing if you switch to `passable`.

## Package layout

```
cwam_pipeline/
  config.py    YAML → dataclasses
  roi.py       pre-TRACON geometry + equal-area cell grid
  fetch.py     IEM N0Q downloader, MRMS S3 listing/downloader
  rasters.py   PNG+WLD and GRIB2 readers, unit conversions, ROI cropping
  metrics.py   per-cell coverage % and echo-top statistics
  waf.py       probability table (load/lookup/default generator)
  products.py  CSV / GeoTIFF / npz / regions / overlay writers
  run.py       CLI orchestrator
```
