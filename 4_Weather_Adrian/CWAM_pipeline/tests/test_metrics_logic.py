"""Numeric verification of the per-cell predictor math with mocked geo deps.

Run: python tests/test_metrics_logic.py  (from the package root)
Verifies coverage %, per-cell p90/max echo tops, dz, and the region extractor
against hand-computed values.
"""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- Mock rasterio / affine / scipy so metrics+waf import without GDAL ----
_LABELS = {}


def _fake_rasterize(shapes, out_shape=None, transform=None, fill=0,
                    dtype="uint32", all_touched=False):
    return _LABELS[out_shape]


fake_rasterio = types.ModuleType("rasterio")
fake_features = types.ModuleType("rasterio.features")
fake_features.rasterize = _fake_rasterize
fake_windows = types.ModuleType("rasterio.windows")
fake_windows.Window = object
fake_windows.from_bounds = lambda *a, **k: None
fake_rasterio.features = fake_features
fake_rasterio.windows = fake_windows
fake_rasterio.open = lambda *a, **k: None
fake_affine = types.ModuleType("affine")


class _FakeAffine:
    def __init__(self, *a):
        self.vals = a
        self.a, self.b, self.c, self.d, self.e, self.f = (a + (0,) * 6)[:6]


fake_affine.Affine = _FakeAffine
for name, mod in [("rasterio", fake_rasterio),
                  ("rasterio.features", fake_features),
                  ("rasterio.windows", fake_windows),
                  ("affine", fake_affine)]:
    sys.modules.setdefault(name, mod)

from cwam_pipeline.metrics import compute_cell_metrics  # noqa: E402
from cwam_pipeline.rasters import GeoGrid, n0q_index_to_dbz  # noqa: E402
from cwam_pipeline.waf import ProbabilityTable, make_default_table  # noqa: E402


class FakeGeom:
    bounds = (-84.0, 41.0, -83.0, 42.0)


class FakeCell:
    def __init__(self, cid):
        self.cell_id = cid
        self.row, self.col = 0, cid - 1
        self.geom_lonlat = FakeGeom()
        self.centroid_lon, self.centroid_lat = -83.5, 41.5
        self.area_fraction = 1.0


def main():
    # --- N0Q index -> dBZ convention (45 dBZ == index 154) ---
    assert n0q_index_to_dbz(np.array([154]))[0] == 45.0
    assert n0q_index_to_dbz(np.array([64]))[0] == 0.0

    # --- Build a 10x10 N0Q grid: cell 1 = left half, cell 2 = right half ---
    n0q_labels = np.zeros((10, 10), dtype=np.uint32)
    n0q_labels[:, :5] = 1
    n0q_labels[:, 5:] = 2

    dbz = np.full((10, 10), 10.0, dtype=np.float32)
    dbz[:2, :5] = 50.0      # 10 of 50 pixels in cell 1 >= 45 dBZ -> 20%
    dbz[:, 5:] = 46.0       # all 50 pixels in cell 2 >= 45 dBZ  -> 100%
    valid = np.ones((10, 10), dtype=bool)
    valid[9, 0] = False     # a "no echo" pixel: still in denominator
    dbz[9, 0] = np.nan
    n0q = GeoGrid(data=dbz, transform=None, valid=valid)

    # --- 4x4 EchoTop grid, same split ---
    eet_labels = np.zeros((4, 4), dtype=np.uint32)
    eet_labels[:, :2] = 1
    eet_labels[:, 2:] = 2
    etop = np.zeros((4, 4), dtype=np.float32)
    etop[:, :2] = np.array([[10000, 20000], [30000, 40000],
                            [15000, 25000], [35000, 5000]])
    etop[:, 2:] = 38000.0
    etop_grid = GeoGrid(data=etop, transform=None, valid=np.ones((4, 4), bool))

    _LABELS[(10, 10)] = n0q_labels
    _LABELS[(4, 4)] = eet_labels

    cells = [FakeCell(1), FakeCell(2)]
    df, labels = compute_cell_metrics(
        cells, n0q, etop_grid, dbz_threshold=45.0, echo_top_stat="p90",
    )
    # dz per altitude level, as done in run.py
    df["dz_ft"] = 35000.0 - df["etop_stat_ft"]

    # coverage: cell1 = 10/50 = 20%, cell2 = 50/50 = 100%
    assert abs(df.loc[0, "coverage_pct"] - 20.0) < 1e-9, df.loc[0, "coverage_pct"]
    assert abs(df.loc[1, "coverage_pct"] - 100.0) < 1e-9

    # echo top p90: cell1 values sorted -> p90 of 8 samples
    exp_p90 = np.percentile([10000, 20000, 30000, 40000, 15000, 25000, 35000, 5000], 90)
    assert abs(df.loc[0, "etop_stat_ft"] - exp_p90) < 1e-6
    assert abs(df.loc[1, "etop_stat_ft"] - 38000.0) < 1e-6
    # dz
    assert abs(df.loc[0, "dz_ft"] - (35000 - exp_p90)) < 1e-6
    assert abs(df.loc[1, "dz_ft"] - (-3000.0)) < 1e-6

    # max statistic
    df2, _ = compute_cell_metrics(
        cells, n0q, etop_grid, dbz_threshold=45.0, echo_top_stat="max",
    )
    assert df2.loc[0, "etop_stat_ft"] == 40000.0

    # --- probability lookup end-to-end on these cells ---
    tbl_df = make_default_table()
    table = ProbabilityTable(
        tbl_df["dz_kft"].to_numpy(float),
        np.array([float(c) for c in tbl_df.columns[1:]]),
        tbl_df.iloc[:, 1:].to_numpy(float),
    )
    p = table.lookup(df["dz_ft"].to_numpy(), df["coverage_pct"].to_numpy())
    assert 0 <= p[0] <= 1 and 0 <= p[1] <= 1
    assert p[1] > p[0], "cell 2 (100% cov, below tops) must out-rank cell 1"

    # --- region extraction (pure numpy path via scipy) ---
    try:
        from cwam_pipeline.products import (
            binary_mask_and_regions, cell_bool_raster,
        )
        pr = np.zeros((6, 6), dtype=np.float32)
        pr[1:3, 1:3] = 0.9
        pr[4:6, 4:6] = 0.7
        tr = _FakeAffine(0.1, 0, -84.0, 0, -0.1, 43.0)

        class T:
            def __mul__(self, xy):
                x, y = xy
                return (-84.0 + 0.1 * x, 43.0 - 0.1 * y)
        mask, regions = binary_mask_and_regions(pr, T(), 0.6, 1)
        assert mask.sum() == 8
        assert len(regions) == 2
        print("region extraction: OK")

        # --- missing-data policy: NaN cells must not silently pass ---
        pr_nan = pr.copy()
        pr_nan[3, 3] = np.nan  # a cell with no valid EchoTop data
        # No missing_raster given -> old/legacy fallback: NaN stays passable
        mask_legacy, _ = binary_mask_and_regions(pr_nan, T(), 0.6, 1)
        assert mask_legacy[3, 3] == 0

        missing = np.zeros((6, 6), dtype=bool)
        missing[3, 3] = True
        mask_strict, _ = binary_mask_and_regions(
            pr_nan, T(), 0.6, 1, missing_raster=missing,
            missing_data_policy="infeasible",
        )
        assert mask_strict[3, 3] == 1, "missing-data cell must be infeasible"
        mask_permissive, _ = binary_mask_and_regions(
            pr_nan, T(), 0.6, 1, missing_raster=missing,
            missing_data_policy="passable",
        )
        assert mask_permissive[3, 3] == 0
        print("missing-data policy (binary_mask_and_regions): OK")

        # --- cell_bool_raster paints a boolean cell attribute correctly ---
        import pandas as pd
        cdf = pd.DataFrame({
            "cell_id": [1, 2], "infeasible": [True, False],
            "data_missing": [False, True],
        })
        labels = np.array([[0, 1], [2, 2]], dtype=np.uint32)
        inf_raster = cell_bool_raster(cdf, labels, "infeasible")
        miss_raster = cell_bool_raster(cdf, labels, "data_missing")
        assert inf_raster.tolist() == [[False, True], [False, False]]
        assert miss_raster.tolist() == [[False, False], [True, True]]
        assert not inf_raster[0, 0], "outside-grid pixel (label 0) must be False"
        print("cell_bool_raster: OK")
    except ImportError as e:
        print(f"region extraction: SKIPPED ({e})")

    print("coverage %, echo-top p90/max, dz, P lookup: ALL PASS")
    print(df[["cell_id", "coverage_pct", "etop_stat_ft", "dz_ft"]].to_string(index=False))


if __name__ == "__main__":
    main()
