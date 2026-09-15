"""Pipeline orchestrator.

Usage:
    python -m cwam_pipeline.run --config config.yaml
    python -m cwam_pipeline.run --config config.yaml --skip-download
    python -m cwam_pipeline.run --config config.yaml --echotop-local-dir ./my_eet

Per timestamp AND per altitude level (FLxxx tag, e.g. FL350 = 35,000 ft),
writes into <output_dir>:
    cells/cells_<stamp>_FLxxx.csv      per-grid-square predictors + P(deviation)
    waf/waf_<stamp>_FLxxx.tif          WAF probability raster (N0Q grid, EPSG:4326)
    masks/mask_<stamp>_FLxxx.npz       binary infeasibility mask (P >= threshold)
    regions/regions_<stamp>_FLxxx.csv  connected infeasible regions
    overlays/overlay_<stamp>_FLxxx.png quick-look plot

Altitude levels come from analysis.altitude_min/max/step_ft (or an explicit
analysis.altitudes_ft list) in the config. Coverage % and echo-top statistics
are altitude-independent and computed once per timestamp; only the
dz -> probability lookup and products repeat per level, so the MILP can
compare candidate routes across all altitudes.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .config import load_config
from .fetch import download_echo_tops, download_n0q, local_echo_tops
from .metrics import compute_cell_metrics
from .products import (
    cell_bool_raster,
    overlay_png,
    probability_raster,
    write_geotiff,
)
from .rasters import load_echo_top, load_n0q
from .roi import PreTraconROI
from .waf import ProbabilityTable, ensure_default_table


def main(argv=None):
    ap = argparse.ArgumentParser(description="CWAM N0Q + EchoTop weather pipeline")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--skip-download", action="store_true",
                    help="Use already-downloaded files in data_dir")
    ap.add_argument("--echotop-local-dir", default=None,
                    help="Directory of pre-downloaded EchoTop files (skips S3)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    a = cfg.analysis
    timestamps = cfg.case.timestamps()
    altitudes = a.altitudes()
    print(f"Processing {len(timestamps)} timestamps: "
          f"{timestamps[0]:%Y-%m-%d %H:%M} .. {timestamps[-1]:%H:%M} UTC")
    print(f"Altitude levels ({len(altitudes)}): "
          + ", ".join(f"FL{int(round(alt / 100)):03d}" for alt in altitudes))

    # ---- Region geometry + analysis grid ----
    roi = PreTraconROI(cfg.region)
    cells, n_rows, n_cols = roi.build_cell_grid(a.cell_size_km, a.min_cell_area_fraction)
    print(f"Pre-TRACON grid: {len(cells)} cells of {a.cell_size_km:g} km "
          f"({n_rows} rows x {n_cols} cols)")
    bounds = roi.bounds_lonlat

    # ---- Probability table ----
    ensure_default_table(cfg.prob_table_path)
    table = ProbabilityTable.from_csv(cfg.prob_table_path)

    # ---- Acquire data ----
    if args.skip_download:
        from .fetch import n0q_filenames
        n0q_files = {}
        for dt in timestamps:
            p = cfg.data_dir / "n0q" / n0q_filenames(dt)[0]
            if p.exists():
                n0q_files[dt] = p
        eet_files = local_echo_tops(
            args.echotop_local_dir or (cfg.data_dir / "echotop"),
            timestamps, a.echo_top_time_tolerance_min,
        )
    else:
        n0q_files = download_n0q(cfg, timestamps)
        if args.echotop_local_dir:
            eet_files = local_echo_tops(
                args.echotop_local_dir, timestamps, a.echo_top_time_tolerance_min
            )
        else:
            eet_files = download_echo_tops(cfg, timestamps)

    # ---- Output dirs ----
    out = cfg.output_dir
    for sub in ("cells", "waf", "masks", "regions", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- Per-timestamp processing ----
    n_done = 0
    for dt in timestamps:
        if dt not in n0q_files or dt not in eet_files:
            print(f"[{dt:%H:%M}] missing input data, skipped")
            continue
        stamp = dt.strftime("%Y%m%d%H%M")
        print(f"[{dt:%H:%M}] N0Q={n0q_files[dt].name}  EET={eet_files[dt].name}")

        n0q = load_n0q(n0q_files[dt], bounds)
        etop = load_echo_top(eet_files[dt], bounds)

        # Altitude-independent predictors: computed once per timestamp
        base_df, n0q_labels = compute_cell_metrics(
            cells, n0q, etop,
            dbz_threshold=a.dbz_threshold,
            echo_top_stat=a.echo_top_stat,
        )

        for alt_ft in altitudes:
            fl = f"FL{int(round(alt_ft / 100)):03d}"
            cell_df = base_df.copy()
            cell_df["altitude_ft"] = alt_ft
            cell_df["dz_ft"] = alt_ft - cell_df["etop_stat_ft"]
            cell_df["p_deviation"] = table.lookup(
                cell_df["dz_ft"].to_numpy(), cell_df["coverage_pct"].to_numpy()
            )
            # A cell has no verdict when its EchoTop stat was NaN (radar
            # coverage gap, thin ROI-boundary sliver never sampled by any
            # MRMS pixel). `missing_data_policy` decides whether that counts
            # as infeasible (default: an unknown weather state should not be
            # assumed safe to fly through) or passable (legacy behavior).
            cell_df["data_missing"] = cell_df["dz_ft"].isna()
            cell_df["infeasible"] = (
                cell_df["p_deviation"] >= a.mask_probability_threshold
            )
            if a.missing_data_policy == "infeasible":
                cell_df["infeasible"] = (
                    cell_df["infeasible"] | cell_df["data_missing"]
                )
            cell_df.insert(0, "timestamp_utc", dt.strftime("%Y-%m-%d %H:%M"))
            cell_df.to_csv(out / "cells" / f"cells_{stamp}_{fl}.csv", index=False)

            p_raster = probability_raster(cell_df, n0q_labels)
            write_geotiff(out / "waf" / f"waf_{stamp}_{fl}.tif",
                          p_raster, n0q.transform)

            missing_raster = cell_bool_raster(cell_df, n0q_labels, "data_missing")

            # Binary mask for the MILP npz: p_deviation >= threshold, plus
            # missing-data cells if missing_data_policy == "infeasible".
            mask = cell_bool_raster(cell_df, n0q_labels, "infeasible").astype(np.uint8)
            np.savez_compressed(
                out / "masks" / f"mask_{stamp}_{fl}.npz",
                binary_mask=mask, altitude_ft=alt_ft,
                data_missing_mask=missing_raster.astype(np.uint8),
            )

            # One row per infeasible grid square (no merging),
            # same column schema as the old t00.csv regions files
            infeasible = cell_df[cell_df["infeasible"]]
            infeasible = infeasible.rename(columns={
                "cell_id": "region_id", "n_pixels_n0q": "n_pixels",
            })[["region_id", "n_pixels", "min_lat", "max_lat",
                "min_lon", "max_lon", "centroid_lat", "centroid_lon",
                "coverage_pct", "dz_ft", "p_deviation", "data_missing"]]
            infeasible.to_csv(out / "regions" / f"regions_{stamp}_{fl}.csv",
                              index=False)

            overlay_png(out / "overlays" / f"overlay_{stamp}_{fl}.png",
                        p_raster, n0q.transform, roi,
                        p_threshold=(a.mask_probability_threshold
                                     if a.overlay_binary else None),
                        missing_raster=missing_raster)

            n_hot = int(cell_df["infeasible"].sum())
            n_missing = int(cell_df["data_missing"].sum())
            print(f"         {fl}: max P={np.nanmax(cell_df['p_deviation']):.2f}  "
                  f"cells infeasible: {n_hot}  cells missing data: {n_missing}")
        n_done += 1

    print(f"Done. {n_done}/{len(timestamps)} timestamps written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
