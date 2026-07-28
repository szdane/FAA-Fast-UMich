"""Configuration loading for the CWAM weather pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml


@dataclass
class CaseConfig:
    date: str
    start_time: str
    end_time: str
    step_minutes: int = 5

    def timestamps(self):
        """List of UTC datetimes to process (inclusive of end_time)."""
        t0 = datetime.strptime(f"{self.date} {self.start_time}", "%Y-%m-%d %H:%M")
        t1 = datetime.strptime(f"{self.date} {self.end_time}", "%Y-%m-%d %H:%M")
        out, t = [], t0
        while t <= t1:
            out.append(t)
            t += timedelta(minutes=self.step_minutes)
        return out


@dataclass
class RegionConfig:
    center_lat: float
    center_lon: float
    radius_km: float
    star_fixes: dict          # {NAME: [lon, lat]}
    ordered_fix_names: list   # boundary order for TRACON polygon


@dataclass
class AnalysisConfig:
    cell_size_km: float = 16.0
    dbz_threshold: float = 45.0
    # Altitude levels at which the WAF is evaluated (manually set limits).
    # Either give an explicit list (altitudes_ft) or a min/max/step range.
    altitude_min_ft: float = 24000.0
    altitude_max_ft: float = 40000.0
    altitude_step_ft: float = 2000.0
    altitudes_ft: list | None = None
    echo_top_stat: str = "p90"
    echo_top_time_tolerance_min: float = 5.0
    min_cell_area_fraction: float = 0.05
    prob_table: str = "prob_table_cwam2.csv"
    mask_probability_threshold: float = 0.6
    min_region_pixels: int = 10
    overlay_binary: bool = False
    # How cells with no valid EchoTop data (missing radar coverage, thin
    # ROI-boundary slivers) are treated in the binary mask / MILP input.
    # "infeasible" (recommended): missing data -> treated as blocked, since
    #   an unknown weather state should not be assumed safe to fly through.
    # "passable": missing data -> treated as clear (legacy behavior).
    missing_data_policy: str = "infeasible"

    def __post_init__(self):
        if self.missing_data_policy not in ("infeasible", "passable"):
            raise ValueError(
                "missing_data_policy must be 'infeasible' or 'passable', "
                f"got {self.missing_data_policy!r}"
            )

    def altitudes(self) -> list:
        """Altitude levels (ft) to evaluate, from the manual limits."""
        if self.altitudes_ft:
            alts = sorted(float(a) for a in self.altitudes_ft)
        else:
            if self.altitude_max_ft < self.altitude_min_ft:
                raise ValueError("altitude_max_ft must be >= altitude_min_ft")
            if self.altitude_step_ft <= 0:
                raise ValueError("altitude_step_ft must be > 0")
            alts, a = [], float(self.altitude_min_ft)
            while a <= self.altitude_max_ft + 1e-6:
                alts.append(a)
                a += self.altitude_step_ft
        return alts


@dataclass
class SourcesConfig:
    n0q_base_url: str = (
        "https://mesonet.agron.iastate.edu/archive/data/{yyyy}/{mm}/{dd}/GIS/uscomp/"
    )
    mrms_bucket_url: str = "https://noaa-mrms-pds.s3.amazonaws.com"
    mrms_prefix: str = "CONUS/EchoTop_18_00.50"
    download_workers: int = 8


@dataclass
class PathsConfig:
    data_dir: str = "./data"
    output_dir: str = "./output"


@dataclass
class PipelineConfig:
    case: CaseConfig
    region: RegionConfig
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    config_dir: Path = Path(".")  # directory of the YAML (for relative paths)

    @property
    def data_dir(self) -> Path:
        return (self.config_dir / self.paths.data_dir).resolve()

    @property
    def output_dir(self) -> Path:
        return (self.config_dir / self.paths.output_dir).resolve()

    @property
    def prob_table_path(self) -> Path:
        p = Path(self.analysis.prob_table)
        return p if p.is_absolute() else (self.config_dir / p).resolve()


def load_config(path) -> PipelineConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(
        case=CaseConfig(**raw["case"]),
        region=RegionConfig(**raw["region"]),
        analysis=AnalysisConfig(**raw.get("analysis", {})),
        sources=SourcesConfig(**raw.get("sources", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        config_dir=path.parent,
    )
