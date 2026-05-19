"""
Preprocess two-aircraft CSV data into 2D relative trajectories and run ACM
using the `compute_unsafe_cond` function from the Active Corners library.

Expected CSV columns:
    t, f1_lat, f1_lon, f1_alt_ft, f2_lat, f2_lon, f2_alt_ft

Usage:
    python run_acm_two_aircraft.py ac_1sec.csv
"""
import sys, os
sys.path.append(os.path.abspath("../../automatic-safety-proofs"))

from safe_region_utils import *
import sys
import math
import pandas as pd
import sympy as sp
from sympy import Point, Polygon, Interval


FT_PER_NM = 6076.12
FT_PER_DEG_LAT = 60.0 * FT_PER_NM


def latlon_to_local_ft(df):
    """Convert lat/lon to local tangent-plane feet.

    north_ft corresponds to latitude direction.
    east_ft corresponds to longitude direction.
    """
    lat0 = pd.concat([df["f1_lat"], df["f2_lat"]]).mean()
    lon0 = pd.concat([df["f1_lon"], df["f2_lon"]]).mean()
    ft_per_deg_lon = FT_PER_DEG_LAT * math.cos(math.radians(lat0))

    for prefix in ["f1", "f2"]:
        df[f"{prefix}_north_ft"] = (df[f"{prefix}_lat"] - lat0) * FT_PER_DEG_LAT
        df[f"{prefix}_east_ft"] = (df[f"{prefix}_lon"] - lon0) * ft_per_deg_lon

    return df


def rectangle_poly(half_x, half_y):
    """ACM protected zone polygon centered at the moving aircraft."""
    return Polygon(
        Point(-half_x, -half_y),
        Point( half_x, -half_y),
        Point( half_x,  half_y),
        Point(-half_x,  half_y),
    )


def point_inside_box(px, py, half_x, half_y):
    """Degenerate segment check."""
    return abs(px) <= half_x and abs(py) <= half_y


def segment_to_sympy_trajectory(p0, p1, x, y, eps=1e-9):
    """Convert a 2D line segment into the form expected by compute_unsafe_cond.

    Returns:
        trajectory expression, domain, mode

    mode is either:
        "y=f(x)" or "x=f(y)"

    If the segment is vertical, we use x=f(y), because y=f(x) would fail.
    If the segment goes backwards, the domain is still sorted geometrically.
    """
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    dx = x1 - x0
    dy = y1 - y0

    if abs(dx) < eps and abs(dy) < eps:
        return None, None, "point"

    if abs(dx) >= eps:
        m = dy / dx
        b = y0 - m * x0
        trajectory = sp.Float(m) * x + sp.Float(b)
        domain = Interval(sp.Float(min(x0, x1)), sp.Float(max(x0, x1)))
        return trajectory, domain, "y=f(x)"

    # vertical segment: x = constant as a function of y
    trajectory = sp.Float(x0)
    domain = Interval(sp.Float(min(y0, y1)), sp.Float(max(y0, y1)))
    return trajectory, domain, "x=f(y)"


def acm_check_segment(p0, p1, half_x, half_y, add_notches=True):
    """Run ACM for one 2D relative segment.

    In the relative frame, aircraft 2 is fixed at the origin.
    compute_unsafe_cond returns the set of obstacle locations that would be unsafe
    for the moving relative trajectory. So we check whether (0,0) is unsafe.
    """
    x, y = sp.symbols("x y", real=True)
    poly = rectangle_poly(half_x, half_y)

    traj, domain, mode = segment_to_sympy_trajectory(p0, p1, x, y)

    if mode == "point":
        unsafe = point_inside_box(p0[0], p0[1], half_x, half_y)
        return unsafe, "degenerate-point", None

    unsafe_cond = compute_unsafe_cond(
        x=x,
        y=y,
        poly=poly,
        trajectory=traj,
        domain=domain,
        add_notches=add_notches,
        print_latex=False,
    )

    unsafe_at_origin = unsafe_cond.subs({x: 0, y: 0})
    unsafe_bool = bool(sp.simplify(unsafe_at_origin))
    return unsafe_bool, mode, unsafe_cond


def build_relative_planes(df):
    """Create relative trajectories A wrt B in the three 2D planes."""

    # Relative aircraft 1 with respect to aircraft 2.
    df["rel_north_ft"] = df["f1_north_ft"] - df["f2_north_ft"]
    df["rel_east_ft"] = df["f1_east_ft"] - df["f2_east_ft"]
    df["rel_alt_ft"] = df["f1_alt_ft"] - df["f2_alt_ft"]

    return {
        "lat_lon": ("rel_north_ft", "rel_east_ft", 500.0, 500.0),
        "lat_alt": ("rel_north_ft", "rel_alt_ft", 500.0, 100.0),
        "lon_alt": ("rel_east_ft", "rel_alt_ft", 500.0, 100.0),
    }


def run_acm(csv_path):
    df = pd.read_csv(csv_path)
    df = latlon_to_local_ft(df)
    planes = build_relative_planes(df)
    print(df)

    results = []

    for plane_name, (xcol, ycol, half_x, half_y) in planes.items():
        for k in range(len(df) - 1):
            p0 = (df.loc[k, xcol], df.loc[k, ycol])
            p1 = (df.loc[k + 1, xcol], df.loc[k + 1, ycol])

            try:
                unsafe, mode, _ = acm_check_segment(p0, p1, half_x, half_y)
            except Exception as e:
                results.append({
                    "plane": plane_name,
                    "segment": k,
                    "t_start": df.loc[k, "t"],
                    "t_end": df.loc[k + 1, "t"],
                    "status": "ERROR",
                    "mode": None,
                    "details": repr(e),
                })
                continue

            results.append({
                "plane": plane_name,
                "segment": k,
                "t_start": df.loc[k, "t"],
                "t_end": df.loc[k + 1, "t"],
                "status": "UNSAFE" if unsafe else "SAFE",
                "mode": mode,
                "details": "",
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "../2.3_Outputs_and_Results/sols/ac_1sec.csv"
    out = run_acm(csv_path)

    print(out.to_string(index=False))

    if (out["status"] == "UNSAFE").any():
        print("\nPotential conflicts:")
        print(out[out["status"] == "UNSAFE"].to_string(index=False))
    else:
        print("\nNo ACM conflicts detected in the checked 2D projections.")
