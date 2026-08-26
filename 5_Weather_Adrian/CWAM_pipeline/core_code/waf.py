"""Weather Avoidance Field: 2D probability lookup table.

P(deviation) = f(dz, coverage) where
  dz       = aircraft altitude - echo top height [kft]  (rows)
  coverage = % of N0Q pixels >= dBZ threshold in cell   (columns)

Table CSV format (editable, any bin spacing, must be monotonic):

    dz_kft,0,5,10,20,...,100      <- coverage % bin centers
    -30,0.000,0.62,...            <- one row per dz bin center
    ...
     20,0.000,0.01,...

Lookup is bilinear interpolation, clamped at the table edges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ProbabilityTable:
    def __init__(self, dz_kft: np.ndarray, coverage_pct: np.ndarray, p: np.ndarray):
        if not (np.all(np.diff(dz_kft) > 0) and np.all(np.diff(coverage_pct) > 0)):
            raise ValueError("Probability table axes must be strictly increasing")
        if p.shape != (dz_kft.size, coverage_pct.size):
            raise ValueError("Probability table shape mismatch")
        self.dz = dz_kft.astype(float)
        self.cov = coverage_pct.astype(float)
        self.p = p.astype(float)

    @classmethod
    def from_csv(cls, path) -> "ProbabilityTable":
        df = pd.read_csv(path)
        dz = df.iloc[:, 0].to_numpy(float)
        cov = np.array([float(c) for c in df.columns[1:]])
        return cls(dz, cov, df.iloc[:, 1:].to_numpy(float))

    def lookup(self, dz_ft, coverage_pct):
        """Bilinear-interpolated P(deviation). Inputs in FEET and %, any shape.

        NaN inputs (e.g. echo top data missing over a cell) return NaN.
        """
        dz_kft = np.asarray(dz_ft, dtype=float) / 1000.0
        cov = np.asarray(coverage_pct, dtype=float)
        nan_mask = ~np.isfinite(dz_kft) | ~np.isfinite(cov)

        x = np.clip(dz_kft, self.dz[0], self.dz[-1])
        y = np.clip(cov, self.cov[0], self.cov[-1])

        i = np.clip(np.searchsorted(self.dz, x) - 1, 0, self.dz.size - 2)
        j = np.clip(np.searchsorted(self.cov, y) - 1, 0, self.cov.size - 2)

        x0, x1 = self.dz[i], self.dz[i + 1]
        y0, y1 = self.cov[j], self.cov[j + 1]
        tx = np.where(x1 > x0, (x - x0) / (x1 - x0), 0.0)
        ty = np.where(y1 > y0, (y - y0) / (y1 - y0), 0.0)

        p00 = self.p[i, j]
        p01 = self.p[i, j + 1]
        p10 = self.p[i + 1, j]
        p11 = self.p[i + 1, j + 1]
        out = (
            p00 * (1 - tx) * (1 - ty)
            + p10 * tx * (1 - ty)
            + p01 * (1 - tx) * ty
            + p11 * tx * ty
        )
        out = np.where(nan_mask, np.nan, out)
        return out


def make_default_table() -> pd.DataFrame:
    """Analytic approximation of the CWAM2 two-predictor deviation probability
    surface (DeLaura et al., Fig. 8c), adapted to reflectivity coverage.

    P = (coverage/100)^0.35 * sigmoid(-dz_kft / 3.5)

    * At dz = 0 (flight at echo top) and high coverage, P ~ 0.5.
    * Flights >= ~10 kft above echo tops rarely deviate (P < 0.06).
    * Flights well below high-coverage weather deviate with P > 0.85.
    * Zero coverage -> P = 0.

    THIS IS AN UNCALIBRATED PLACEHOLDER shaped to match the published figure
    qualitatively. Replace prob_table_cwam2.csv with values calibrated on
    your own encounter data (or digitized from the paper) for research use.
    """
    dz = np.array([-30, -25, -20, -15, -10, -7.5, -5, -2.5, 0,
                   2.5, 5, 7.5, 10, 15, 20], dtype=float)
    cov = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    dzg, covg = np.meshgrid(dz, cov, indexing="ij")
    p = (covg / 100.0) ** 0.35 / (1.0 + np.exp(dzg / 3.5))
    df = pd.DataFrame(np.round(p, 3), columns=[f"{c:g}" for c in cov])
    df.insert(0, "dz_kft", dz)
    return df


def ensure_default_table(path) -> Path:
    path = Path(path)
    if not path.exists():
        make_default_table().to_csv(path, index=False)
        print(f"  [waf] wrote default probability table -> {path}")
    return path
