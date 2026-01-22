import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- helper: haversine distance in meters ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0  # meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# --- robust time parsing ---
def to_datetime_series(s, epoch_unit_default='s'):
    """
    Return a pandas datetime64[ns] Series.
    Handles ISO strings, numeric epochs (s/ms/us/ns), and numeric-as-strings.
    """
    s = s.copy()

    # If numeric dtype OR looks numeric-as-strings -> try numeric epochs
    try_numeric = np.issubdtype(s.dtype, np.number) or (
        s.dtype == 'object' and pd.to_numeric(s, errors='coerce').notna().mean() > 0.95
    )
    if try_numeric:
        num = pd.to_numeric(s, errors='coerce')
        # Heuristic: choose epoch unit by magnitude
        # (ns ~1e18, us ~1e15, ms ~1e12, s ~1e9 around 2020s)
        abs_med = np.nanmedian(np.abs(num.values))
        if abs_med > 1e17:
            unit = 'ns'
        elif abs_med > 1e14:
            unit = 'us'
        elif abs_med > 1e11:
            unit = 'ms'
        elif abs_med > 1e6:
            unit = 's'
        else:
            # small numbers -> probably "seconds since start" (relative)
            # interpret as seconds since epoch default (1970-01-01)
            unit = epoch_unit_default
        dt = pd.to_datetime(num, unit=unit, errors='coerce', utc=True)
    else:
        # ISO / general strings
        dt = pd.to_datetime(s, errors='coerce', utc=True)

    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} timestamps in time column; "
            f"ensure values are ISO strings or numeric epochs."
        )
    return dt.tz_convert(None)  # drop UTC tz to simplify downstream

def compute_velocity(df, t_col, lat_col, lon_col, alt_col):
    df = df[[t_col, lat_col, lon_col, alt_col]].copy()
    # --- make sure time is datetime and sorted ---
    if not np.issubdtype(df[t_col].dtype, np.datetime64):
        df[t_col] = to_datetime_series(df[t_col])
    df = df.sort_values(t_col).reset_index(drop=True)

    # --- dt in seconds ---
    dt = df[t_col].diff().dt.total_seconds().to_numpy()

    # --- distances ---
    horiz = haversine(
        df[lat_col].shift(), df[lon_col].shift(),
        df[lat_col],         df[lon_col]
    )
    vert = (df[alt_col].diff() * 0.3048).to_numpy()  # ft -> m
    total = np.sqrt(horiz**2 + vert**2)

    # --- speed with guards (avoid divide-by-zero/NaN on the first row etc.) ---
    with np.errstate(invalid='ignore', divide='ignore'):
        speed_m_s = np.where(dt > 0, total / dt, np.nan)

    out = df.copy()
    out["speed_m_s"] = speed_m_s
    out["speed_knots"] = speed_m_s * 1.94384
    return out

def compare_flight_speeds(df1, df2, label1="Flight A", label2="Flight B",
                          t_col="t", lat_col="f1_lat", lon_col="f1_lon", alt_col="f1_alt_ft",
                          max_nearest_tol=None):
    v1 = compute_velocity(df1, t_col, lat_col, lon_col, alt_col)
    v2 = compute_velocity(df2, t_col, lat_col, lon_col, alt_col)

    # Align by nearest time (optionally with tolerance)
    tol = (pd.to_timedelta(max_nearest_tol) if isinstance(max_nearest_tol, str)
           else max_nearest_tol)
    merged = pd.merge_asof(
        v1[[t_col, "speed_knots"]].sort_values(t_col).rename(columns={"speed_knots": f"speed_knots_{label1}"}),
        v2[[t_col, "speed_knots"]].sort_values(t_col).rename(columns={"speed_knots": f"speed_knots_{label2}"}),
        on=t_col, direction="nearest", tolerance=tol
    ).dropna()

    merged["delta_speed_knots"] = merged[f"speed_knots_{label2}"] - merged[f"speed_knots_{label1}"]

    print(merged[t_col])

    # --- plots ---
    plt.figure(figsize=(9, 5))
    plt.plot(merged[t_col], merged[f"speed_knots_{label1}"], label=label1)
    plt.plot(merged[t_col], merged[f"speed_knots_{label2}"], label=label2)
    plt.xlabel("Time"); plt.ylabel("Speed (knots)")
    plt.title("Flight Speeds Over Time")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    plt.figure(figsize=(9, 4))
    plt.plot(merged[t_col], merged["delta_speed_knots"])
    plt.axhline(0, lw=0.8, linestyle="--")
    plt.xlabel("Time"); plt.ylabel("Δ Speed (knots)")
    plt.title(f"Speed Difference: {label2} − {label1}")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

    # --- text summary ---
    print("\nΔ Speed (knots) summary:")
    print(merged["delta_speed_knots"].describe().to_string())


# Example usage:
df1 = pd.read_csv("solution28_single_CI0.csv")  # must have t, lat, lon, alt_ft
df2 = pd.read_csv("solution28_single_CI999.csv")


df1["t"] = pd.to_datetime(df1["t"], unit="s", errors="coerce")
df2["t"] = pd.to_datetime(df2["t"], unit="s", errors="coerce")


compare_flight_speeds(df1, df2,
                      label1="Lambda = 0", label2="Lambda = MAX",
                      t_col="t", lat_col="f1_lat", lon_col="f1_lon", alt_col="f1_alt_ft")


