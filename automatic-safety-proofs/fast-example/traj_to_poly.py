import numpy as np
import pandas as pd
import sympy as sp

def _build_merged_piecewise(t_vals, y_vals, var_symbol="x",
                            closed_right=True, atol=1e-12, rtol=1e-9,
                            digits=None,
                            ndigits=None,          # <- new: round coeffs (m) and tracked y’s
                            round_knots=True):     # <- new: round the value at each knot
    """
    Create a SymPy Piecewise for linear interpolation through (t_i, y_i),
    after MERGING consecutive segments that represent the SAME LINE y = m*t + b
    (within atol/rtol). Then optionally round coefficients to keep decimals small,
    while preserving continuity at the knots.

    Parameters
    ----------
    ...
    digits : int | None
        If not None, wrap numeric literals as sp.Float(x, digits). (Original behavior.)
    ndigits : int | None
        If not None, round slopes to this many decimals; intercepts are recomputed
        to enforce continuity at each knot. If None, no rounding is applied.
    round_knots : bool
        If True and ndigits is not None, the tracked y-value at each knot is also
        rounded to ndigits before computing the next segment's intercept. This keeps
        what you *see* at breakpoints to e.g. .2f too, without breaking continuity.
    """
    # Clean + sort + dedup times (average duplicates)
    df = pd.DataFrame({"t": t_vals, "y": y_vals}).dropna()
    df = df.groupby("t", as_index=False).mean().sort_values("t")
    t = df["t"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("Need at least two points to build a piecewise-linear function.")

    # Build raw segments: [(t0, t1, m, b)]
    segs = []
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        if dt == 0:
            continue
        m = (y[i+1] - y[i]) / dt
        b = y[i] - m * t[i]  # so that y = m*t + b
        segs.append([t[i], t[i+1], m, b])

    if not segs:
        raise ValueError("Degenerate data after deduplication.")

    # Merge consecutive segments with same (m, b) within tolerance
    def close(a, b):
        return abs(a - b) <= (atol + rtol * max(abs(a), abs(b)))

    merged = [segs[0]]
    for t0, t1, m, b in segs[1:]:
        t0_prev, t1_prev, m_prev, b_prev = merged[-1]
        if close(m, m_prev) and close(b, b_prev) and close(t0, t1_prev):
            merged[-1][1] = t1
        else:
            merged.append([t0, t1, m, b])

    T = sp.symbols(var_symbol, real=True)

    # Helpers to create sympy floats the way you want
    def S_num(x):
        # First apply *coefficient* rounding if requested (for literal emission)
        if digits is None:
            return sp.Float(x)
        else:
            return sp.Float(x, digits)

    pieces = []

    # We will compute rounded slopes and continuity-preserving intercepts.
    # Also track the rounded function value at each knot to keep visible decimals low.
    def r(x):
        return round(x, ndigits) if ndigits is not None else x

    # initial left knot value (from original data)
    t0_first = merged[0][0]
    y0_first = merged[0][2] * t0_first + merged[0][3]  # m*t0 + b
    y_left = r(y0_first) if (ndigits is not None and round_knots) else y0_first

    for k, (t0, t1, m, b) in enumerate(merged):
        # Round the slope; recompute intercept to hit y_left at t0
        m_r = r(m) if ndigits is not None else m
        b_r = y_left - m_r * t0

        expr = S_num(m_r) * T + S_num(b_r)

        # Build half-open intervals except possibly the last
        if (k < len(merged) - 1) or not closed_right:
            cond = sp.And(T >= S_num(t0), T < S_num(t1))
        else:
            cond = sp.And(T >= S_num(t0), T <= S_num(t1))

        pieces.append((expr, cond))

        # Advance y_left to the *end* of this segment and (optionally) round it,
        # so the next segment starts exactly here.
        y_right = m_r * t1 + b_r
        y_left = r(y_right) if (ndigits is not None and round_knots) else y_right

    return sp.Piecewise(*pieces)

# Thin wrapper you already had; just pass through ndigits/round_knots
def piecewise_from_rel_df_merged(rel_df, axis="up", t_col="x", var_symbol="x",
                                 closed_right=True, atol=1e-12, rtol=1e-9,
                                 digits=None, ndigits=None, round_knots=True):
    """
    rel_df must contain time column t_col and one of:
      axis='east'  -> 'relative_lat_ft'
      axis='north' -> 'relative_lon_ft'
      axis='up'    -> 'relative_alt_ft'
    Returns a SymPy Piecewise with merged colinear segments and (optional) rounded coefficients.
    """
    colmap = {"east": "relative_lat_ft", "north": "relative_lon_ft", "up": "relative_alt_ft"}
    if axis not in colmap:
        raise ValueError("axis must be one of {'east','north','up'}")

    df = (rel_df[[t_col, colmap[axis]]]
          .dropna()
          .sort_values(t_col)
          .rename(columns={t_col: "t", colmap[axis]: "y"}))

    return _build_merged_piecewise(df["t"].to_numpy(),
                                   df["y"].to_numpy(),
                                   var_symbol=var_symbol,
                                   closed_right=closed_right,
                                   atol=atol, rtol=rtol,
                                   digits=digits,
                                   ndigits=ndigits,
                                   round_knots=round_knots)


rel = pd.read_csv("relative_trajectories_alt.csv")

E_pw = piecewise_from_rel_df_merged(rel, axis="east",  t_col="t", var_symbol="x",
                                    atol=1e-3, rtol=1e-1, digits=None,
                                    ndigits=0, round_knots=True)
N_pw = piecewise_from_rel_df_merged(rel, axis="north", t_col="t", var_symbol="x",
                                    atol=1e-3, rtol=1e-1, digits=None,
                                    ndigits=0, round_knots=True)
U_pw = piecewise_from_rel_df_merged(rel, axis="up",    t_col="t", var_symbol="x",
                                    atol=1e-3, rtol=1e-1, digits=None,
                                    ndigits=0, round_knots=True)

print("Latitude:",  E_pw)
print("Longitude:", N_pw)
print("Altitude:",  U_pw)
