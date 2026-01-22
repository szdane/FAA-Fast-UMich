# split_and_trim_trailing_equal_to_last_tol.py
import os
import pandas as pd
import numpy as np

def _get_eps_for(col, eps):
    """Allow eps to be a scalar (float) or a per-column dict."""
    if isinstance(eps, dict):
        return float(eps.get(col, 0.0))
    return float(eps)

def _rows_equal_to_last(df, cols, eps=0.0):
    """
    Boolean mask where True means the row's `cols` are equal to the LAST row's `cols`
    within absolute tolerance `eps` (scalar or dict). NaNs are treated as equal.
    """
    last_vals = df.iloc[-1][cols]
    masks = []
    for c in cols:
        tol = _get_eps_for(c, eps)
        a = df[c].astype(float).values
        b = float(last_vals[c])
        bvec = np.full(len(df), b, dtype=float)
        both_nan = np.isnan(a) & np.isnan(bvec)
        if tol == 0.0:
            eq = (a == bvec) | both_nan
        else:
            close = np.abs(a - bvec) <= tol
            eq = close | both_nan
        masks.append(eq)
    eq_all = np.logical_and.reduce(masks)
    return pd.Series(eq_all, index=df.index)

def trim_tail_equal_to_last(df, cols, eps=0.0, keep_one_if_all_trimmed=False):
    """
    Remove the trailing segment where rows are equal (within eps) to the LAST row,
    but KEEP the first row of that trailing plateau.
    Examples:
      A B C C  -> A B C
      A B C    -> A B C     (plateau length 1 => keep all)
      A A A    -> A          (keep one)
    """
    n = len(df)
    if n == 0:
        return df.copy()
    if n == 1:
        # Only one row: keep it (acts like plateau length 1)
        return df.copy()

    eq_to_last = _rows_equal_to_last(df, cols, eps=eps)
    # Count trailing True's (rows equal to the last row)
    r = 0
    for v in reversed(eq_to_last.values):
        if v:
            r += 1
        else:
            break

    # r >= 1 always (the last row equals itself).
    # Keep everything up to the first row of the trailing plateau (inclusive).
    keep_upto = n - r + 1  # index to slice up to (exclusive)
    if keep_upto <= 0:
        # Entire df is the plateau
        return df.iloc[: (1 if keep_one_if_all_trimmed or n >= 1 else 0)].copy()
    return df.iloc[:keep_upto].copy()


def split_and_process(input_csv, output_dir, eps=1e-4, rename_columns=False,
                      keep_one_if_all_trimmed=False):
    """
    Reads wide CSV:
      t, f1_lat,f1_lon,f1_alt_ft, ..., f10_lat,f10_lon,f10_alt_ft
    Produces per-flight CSVs with trailing rows within tolerance of the last row removed.
    `eps` may be a float (applied to all cols) or a dict, e.g.:
      eps = {"f1_lat":1e-6, "f1_lon":1e-6, "f1_alt_ft":1e-3}
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df = df.sort_values('t', kind='stable').reset_index(drop=True)

    for i in range(1, 11):
        lat = f"f{i}_lat"
        lon = f"f{i}_lon"
        alt = f"f{i}_alt_ft"
        sub = df[['t', lat, lon, alt]].copy()

        if rename_columns:
            sub.columns = ['t', 'lat', 'lon', 'alt_ft']
            cols = ['lat', 'lon', 'alt_ft']
            # If you pass a dict for eps using original names, remap here as needed.
        else:
            cols = [lat, lon, alt]

        trimmed = trim_tail_equal_to_last(
            sub, cols=cols, eps=eps, keep_one_if_all_trimmed=keep_one_if_all_trimmed
        )

        out_path = os.path.join(output_dir, f"flight{i}.csv")
        trimmed.to_csv(out_path, index=False)
        print(f"Saved {out_path} (rows: {len(trimmed)})")

if __name__ == "__main__":
    INPUT_CSV = "trial10flights.csv"
    OUTPUT_DIR = "per_flight"

    # Use absolute tolerance 1e-4 across lat/lon/alt. Adjust if needed, or pass a dict.
    split_and_process(
        input_csv=INPUT_CSV,
        output_dir=OUTPUT_DIR,
        eps=1e-4,                 # <= 1e-4 difference counts as "equal" to the last row
        rename_columns=False,
        keep_one_if_all_trimmed=False
    )
