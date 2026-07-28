# CWAM Weather Pipeline — README

Implements the two-predictor structure of the MIT Lincoln Laboratory
Convective Weather Avoidance Model  over a configurable
pre-TRACON region, combining IEM N0Q composite reflectivity with NOAA MRMS
EchoTop_18 (NET) data, and producing per-altitude weather-avoidance products
for the MILP route optimizer.

```
cwam_wx_pipeline/
├── config.yaml              all user-configurable settings
├── prob_table_cwam2.csv     2D probability lookup table (editable)
├── requirements.txt         Python dependencies
├── run_pipeline.py          Spyder-friendly launcher (F5)
├── plot_raw_rasters.py      renders raw N0Q / NET images for one timestamp
└── cwam_pipeline/           the package
    ├── __init__.py
    ├── config.py            YAML → typed config objects
    ├── roi.py               pre-TRACON geometry + analysis cell grid
    ├── fetch.py             data downloaders (IEM, NOAA S3)
    ├── rasters.py           raster readers + unit conversions
    ├── metrics.py           per-cell predictor computation
    ├── waf.py               probability table load/lookup
    ├── products.py          output writers (tif/npz/csv/png)
    └── run.py               CLI orchestrator (ties everything together)
```

---

## What each file does

**`config.yaml`** — the only file you normally touch. Defines the case
(date/time window), the region (airport center, radius, STAR fixes), the
analysis parameters (cell size, dBZ threshold, altitude limits, thresholds),
data source URLs, and directories.

**`prob_table_cwam2.csv`** — the P(deviation) lookup table. Rows = Δz bin
centers in kft (aircraft altitude − echo top; negative = below the tops),
columns = coverage-% bin centers. Edit freely; regenerated with default
values only if the file is missing.

**`run_pipeline.py`** — thin launcher so the pipeline can be run from Spyder
with F5. Sets the import path, reads the option constants at the top of the
file (`CONFIG`, `SKIP_DOWNLOAD`, `ECHOTOP_LOCAL_DIR`), and calls
`cwam_pipeline.run.main()`. Equivalent to running
`python -m cwam_pipeline.run --config config.yaml` in a terminal.

**`plot_raw_rasters.py`** — standalone illustration tool. For one timestamp,
renders the raw N0Q frame (NWS reflectivity palette, dBZ colorbar) and the
raw NET frame (echo tops in kft) cropped to the pre-TRACON circle with the
blue/red boundaries overlaid. Same aspect convention as the pipeline
overlays so all image types line up side by side.

**`cwam_pipeline/config.py`** — see next section.

**`cwam_pipeline/roi.py`** — geometry. Builds the TRACON polygon from the
ordered STAR fixes, the geodesic circle around the airport (WGS84, exact
`radius_km` in every direction), and the pre-TRACON region (circle minus
TRACON). Then tiles that region with equally sized square cells: the grid is
constructed in a local azimuthal-equidistant projection centered on the
airport so every cell is a true `cell_size_km` × `cell_size_km` square on
the ground, then each cell is clipped to the region and transformed back to
lon/lat. Cells with less than `min_cell_area_fraction` of their area inside
the region are dropped.

**`cwam_pipeline/fetch.py`** — data acquisition. Downloads N0Q PNG+WLD pairs
from the IEM archive (concurrent, cached, same source as wx_grid_creator_3)
and EchoTop_18 GRIB2 files from the public NOAA S3 bucket (`noaa-mrms-pds`),
listing each day's objects via the S3 REST API and pairing each N0Q
timestamp with the nearest EchoTop file within
`echo_top_time_tolerance_min`. `local_echo_tops()` supports pre-downloaded
files instead of S3.

**`cwam_pipeline/rasters.py`** — raster loading. Reads both products cropped
to the region bounding box and returns `(data, transform, valid)` grids in
lon/lat. Converts N0Q palette index → dBZ (`dBZ = index/2 − 32`; index 154 ⇔
45 dBZ, matching wx_grid_creator_3) and EchoTop km → feet. Handles the MRMS
0–360° longitude convention and its sentinel values: −999 (no radar
coverage) → NaN, other negatives (scanned, no echo) → 0 ft.

**`cwam_pipeline/metrics.py`** — the two CWAM predictors, computed once per
timestamp (both are altitude-independent). Rasterizes the cell polygons onto
each raster's own grid (so the differing N0Q/NET resolutions never need
co-registration), then per cell: `coverage_pct` = % of N0Q pixels ≥
`dbz_threshold` (no-echo pixels count in the denominator), and
`etop_stat_ft` = 90th-percentile (or max) echo top. Returns a DataFrame plus
the cell-label raster used later for painting.

**`cwam_pipeline/waf.py`** — the probability model. Loads the CSV table,
validates monotonic axes, and does clamped bilinear interpolation:
`P = table(dz_ft, coverage_pct)`. NaN inputs (no echo-top data over a cell)
return NaN. Also contains the default-table generator (an analytic
approximation of CWAM2 Fig. 8c — replace with calibrated values for research
use).

**`cwam_pipeline/products.py`** — output writers: paints cell probabilities
onto the N0Q pixel grid, writes GeoTIFFs, thresholds into binary masks,
and renders the black-background overlay PNGs (boundaries in blue/red;
continuous grayscale P, or thresholded black/white if `overlay_binary`).

**`cwam_pipeline/run.py`** — orchestrator. Parses CLI args, loads config,
builds geometry once, downloads data, then loops timestamps × altitudes
writing all products (see "How outputs are generated").

**`tests/test_metrics_logic.py`** — verifies the numeric core (dBZ
conversion, coverage %, p90/max echo tops, Δz, table interpolation, region
extraction) against hand-computed values, with the geo libraries mocked so
it runs anywhere.

---

## How config.py works

`config.py` turns `config.yaml` into typed Python objects. Each YAML section
maps to a dataclass:

| YAML section | Dataclass | Contents |
|---|---|---|
| `case` | `CaseConfig` | date, start/end time, step |
| `region` | `RegionConfig` | center, radius, STAR fixes, boundary order |
| `analysis` | `AnalysisConfig` | cell size, thresholds, altitudes, table path |
| `sources` | `SourcesConfig` | URLs, S3 prefix, worker count |
| `paths` | `PathsConfig` | data/output directories |

`load_config(path)` reads the YAML and constructs a `PipelineConfig` holding
all five. Anything omitted from the YAML falls back to the dataclass
defaults, so a minimal config only needs `case` and `region`.

Two kinds of derived values are computed on demand:

- `CaseConfig.timestamps()` expands date + window + step into the list of
  datetimes to process (inclusive of `end_time`).
- `AnalysisConfig.altitudes()` expands the manual altitude limits
  (`altitude_min_ft`/`altitude_max_ft`/`altitude_step_ft`) into the list of
  flight levels — unless an explicit `altitudes_ft` list is given, which
  overrides the range. Invalid limits (max < min, step ≤ 0) raise
  immediately at startup.

Relative paths (`data_dir`, `output_dir`, `prob_table`) are resolved
relative to the folder containing `config.yaml`, not the shell's working
directory — so runs behave identically from a terminal or Spyder.

---

## How the outputs are generated

Per run, `run.py` executes this sequence:

1. **Setup (once).** Load config → build region geometry and the cell grid
   (`roi.py`) → load or create the probability table (`waf.py`).
2. **Acquire (once).** Download/reuse N0Q files for every timestamp and pair
   each with its nearest EchoTop file (`fetch.py`). Timestamps missing
   either input are skipped with a warning.
3. **Per timestamp:** load both rasters cropped to the region
   (`rasters.py`) → compute the altitude-independent cell predictors
   `coverage_pct` and `etop_stat_ft` once (`metrics.py`).
4. **Per altitude level** (tag `FLxxx`, e.g. FL340): compute
   `dz_ft = altitude − etop_stat_ft`, look up `p_deviation` per cell
   (`waf.py`), then write the product set (`products.py`).

Products per (timestamp `<t>`, altitude `FLxxx`), under `output_dir`:

| File | Contents | Generated how |
|---|---|---|
| `cells/cells_<t>_FLxxx.csv` | one row per grid square: ids, centroid/bounds, `n_pixels_n0q`, `coverage_pct`, `etop_stat_ft`, `altitude_ft`, `dz_ft`, `p_deviation` | the predictor DataFrame + table lookup |
| `waf/waf_<t>_FLxxx.tif` | probability raster, EPSG:4326 float32 | each N0Q pixel painted with its cell's P via the label raster |
| `masks/mask_<t>_FLxxx.npz` | keys `binary_mask` (uint8, 1 = infeasible) + `altitude_ft` | pixels where P ≥ `mask_probability_threshold`; drop-in for the Gurobi/MILP workflow |
| `regions/regions_<t>_FLxxx.csv` | one row per **infeasible cell** (no merging): `region_id`, `n_pixels`, lat/lon bounds, centroid, plus `coverage_pct`, `dz_ft`, `p_deviation` | filter of the cells table at the mask threshold; first 8 columns match the legacy `t00.csv` schema |
| `overlays/overlay_<t>_FLxxx.png` | black-background quick-look, pre-TRACON boundary blue, TRACON red | grayscale P (or black/white mask if `overlay_binary: true`) |

Additionally `plot_raw_rasters.py` writes `raw/n0q_raw_<t>.png` and
`raw/echotop_raw_<t>.png` for illustration.

Pixel (row, col) → lon/lat georeferencing for the npz masks comes from the
matching GeoTIFF: `rasterio.open(waf_tif).transform`.

---

## Running

```bash
pip install -r requirements.txt          # or conda install -c conda-forge rasterio shapely pyproj
python -m cwam_pipeline.run --config config.yaml
python -m cwam_pipeline.run --config config.yaml --skip-download        # reuse data_dir
python -m cwam_pipeline.run --config config.yaml --echotop-local-dir D  # own EchoTop files
```

Or open `run_pipeline.py` in Spyder and press F5.
