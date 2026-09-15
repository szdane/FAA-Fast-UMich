"""Pre-TRACON region geometry and the equal-size analysis cell grid.

The pre-TRACON region is a geodesic circle around the airport center minus
the TRACON polygon built from ordered STAR fixes (as in wx_grid_creator_3).

The analysis grid is built in a local azimuthal-equidistant (AEQD)
projection centered on the airport, so every cell is an exact
cell_size_km x cell_size_km square on the ground ("equally sized grids").
Cell polygons are transformed back to lon/lat for raster sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyproj
from pyproj import Geod
from shapely.geometry import Polygon, box
from shapely.ops import transform as shp_transform

from .config import RegionConfig


@dataclass
class Cell:
    cell_id: int          # 1-based (0 is reserved for "no cell" in label rasters)
    row: int              # grid row (0 = northernmost)
    col: int              # grid col (0 = westernmost)
    geom_lonlat: Polygon  # cell area clipped to the pre-TRACON region (lon/lat)
    centroid_lon: float
    centroid_lat: float
    area_fraction: float  # fraction of the full square inside the ROI


class PreTraconROI:
    def __init__(self, region: RegionConfig):
        self.region = region

        # --- TRACON polygon from ordered STAR fixes (lon, lat) ---
        self.tracon_polygon = Polygon(
            [tuple(region.star_fixes[name]) for name in region.ordered_fix_names]
        )

        # --- Geodesic circle around the center ---
        geod = Geod(ellps="WGS84")
        az = np.linspace(0, 360, 361)
        lons, lats = [], []
        for a in az:
            lon2, lat2, _ = geod.fwd(
                region.center_lon, region.center_lat, a, region.radius_km * 1000.0
            )
            lons.append(lon2)
            lats.append(lat2)
        self.geodesic_circle = Polygon(zip(lons, lats))

        # --- Pre-TRACON region = circle minus TRACON ---
        self.pre_tracon_area = self.geodesic_circle.difference(self.tracon_polygon)

        # --- Local AEQD projection (meters) centered on the airport ---
        self.crs_lonlat = pyproj.CRS("EPSG:4326")
        self.crs_local = pyproj.CRS.from_proj4(
            f"+proj=aeqd +lat_0={region.center_lat} +lon_0={region.center_lon} "
            "+datum=WGS84 +units=m +no_defs"
        )
        self._to_local = pyproj.Transformer.from_crs(
            self.crs_lonlat, self.crs_local, always_xy=True
        ).transform
        self._to_lonlat = pyproj.Transformer.from_crs(
            self.crs_local, self.crs_lonlat, always_xy=True
        ).transform

        self.pre_tracon_local = shp_transform(self._to_local, self.pre_tracon_area)

    @property
    def bounds_lonlat(self):
        """(minx, miny, maxx, maxy) of the full geodesic circle in lon/lat."""
        return self.geodesic_circle.bounds

    def build_cell_grid(self, cell_size_km: float, min_area_fraction: float = 0.05):
        """Tile the pre-TRACON region with equal-size square cells.

        Returns (cells, n_rows, n_cols). Cells are clipped to the ROI; cells
        whose ROI overlap is < min_area_fraction of a full square are dropped.
        """
        size = cell_size_km * 1000.0
        minx, miny, maxx, maxy = self.pre_tracon_local.bounds

        # Snap grid origin outward to whole cells so the grid is stable
        x0 = np.floor(minx / size) * size
        y1 = np.ceil(maxy / size) * size
        n_cols = int(np.ceil((maxx - x0) / size))
        n_rows = int(np.ceil((y1 - miny) / size))

        full_area = size * size
        cells = []
        cid = 0
        for r in range(n_rows):
            cy1 = y1 - r * size          # top edge of this row
            cy0 = cy1 - size
            for c in range(n_cols):
                cx0 = x0 + c * size
                cx1 = cx0 + size
                square = box(cx0, cy0, cx1, cy1)
                clipped = square.intersection(self.pre_tracon_local)
                if clipped.is_empty:
                    continue
                frac = clipped.area / full_area
                if frac < min_area_fraction:
                    continue
                cid += 1
                # Densify edges before reprojecting so square sides stay accurate
                clipped_ll = shp_transform(
                    self._to_lonlat, clipped.segmentize(size / 8.0)
                )
                cen = shp_transform(self._to_lonlat, clipped.centroid)
                cells.append(
                    Cell(
                        cell_id=cid,
                        row=r,
                        col=c,
                        geom_lonlat=clipped_ll,
                        centroid_lon=cen.x,
                        centroid_lat=cen.y,
                        area_fraction=frac,
                    )
                )
        return cells, n_rows, n_cols
