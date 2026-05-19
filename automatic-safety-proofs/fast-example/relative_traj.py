import numpy as np
import pandas as pd

def relative_from_csv(csv_path, time_grid="as_is"):
    """
    Compute airplane-2's trajectory relative to airplane-1 from a CSV with columns:
      t, f1_lat, f1_lon, f1_alt_ft, f2_lat, f2_lon, f2_alt_ft

    Returns
    -------
    pd.DataFrame with columns:
        t                 : time
        rel_east_nm       : east offset of A2 relative to A1 (nautical miles)
        rel_north_nm      : north offset of A2 relative to A1 (nautical miles)
        rel_up_nm         : vertical offset of A2 relative to A1 (nautical miles; up positive)
        range_nm          : Euclidean separation (nautical miles)
    """
    df = pd.read_csv(csv_path)

    required = ["t","f1_lat","f1_lon","f1_alt_ft","f2_lat","f2_lon","f2_alt_ft"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("t")

    def interp_to(new_t, t_src, v_src):
        v = np.interp(new_t, t_src, v_src, left=np.nan, right=np.nan)
        if np.isnan(v[0]):  v[0]  = v_src.iloc[0]
        if np.isnan(v[-1]): v[-1] = v_src.iloc[-1]
        if np.isnan(v).any():
            idx = np.flatnonzero(~np.isnan(v))
            v = np.interp(np.arange(len(v)), idx, v[idx])
        return v

    t1 = df["t"].to_numpy()

    if time_grid == "as_is":
        t = t1
        f1_lat, f1_lon, f1_alt_ft = df["f1_lat"], df["f1_lon"], df["f1_alt_ft"]
        f2_lat, f2_lon, f2_alt_ft = df["f2_lat"], df["f2_lon"], df["f2_alt_ft"]

    elif time_grid == "interp_to_f1":
        t = t1
        f1_lat, f1_lon, f1_alt_ft = df["f1_lat"], df["f1_lon"], df["f1_alt_ft"]
        f2_lat = interp_to(t, df["t"], df["f2_lat"])
        f2_lon = interp_to(t, df["t"], df["f2_lon"])
        f2_alt_ft = interp_to(t, df["t"], df["f2_alt_ft"])

    elif time_grid == "union":
        t = np.unique(df["t"].to_numpy())
        for prefix in ("f1_", "f2_"):
            g = df[["t", prefix+"lat", prefix+"lon", prefix+"alt_ft"]].groupby("t").mean().reset_index()
            if prefix == "f1_":
                f1_lat = interp_to(t, g["t"], g[prefix+"lat"])
                f1_lon = interp_to(t, g["t"], g[prefix+"lon"])
                f1_alt_ft = interp_to(t, g["t"], g[prefix+"alt_ft"])
            else:
                f2_lat = interp_to(t, g["t"], g[prefix+"lat"])
                f2_lon = interp_to(t, g["t"], g[prefix+"lon"])
                f2_alt_ft = interp_to(t, g["t"], g[prefix+"alt_ft"])
    else:
        raise ValueError("time_grid must be one of {'as_is','interp_to_f1','union'}")

    # --- Convert per-row lat/lon deltas to local ENU in meters ---
    DEG_TO_RAD = np.pi / 180.0
    m_per_deg_lat = 111_132.0
    phi = f1_lat * DEG_TO_RAD
    m_per_deg_lon = 111_320.0 * np.cos(phi)

    dlat = (f2_lat - f1_lat)
    dlon = (f2_lon - f1_lon)
    east_m  = dlon * m_per_deg_lon
    north_m = dlat * m_per_deg_lat

    # Vertical (feet -> meters)
    FT_TO_M = 0.3048
    up_m = (f2_alt_ft - f1_alt_ft) * FT_TO_M

    # Convert to nautical miles
    M_TO_NM = 1.0 / 1852.0
    east_nm  = east_m * M_TO_NM
    north_nm = north_m * M_TO_NM
    up_nm    = up_m * M_TO_NM
    range_nm = np.sqrt(east_nm**2 + north_nm**2 + up_nm**2)

    return pd.DataFrame({
        "t": t,
        "rel_east_nm": east_nm,
        "rel_north_nm": north_nm,
        "rel_up_nm": up_nm,
        "range_nm": range_nm,
    })


rel = relative_from_csv("ac_1sec.csv", time_grid="as_is")
rel.to_csv("rel1.csv")
print(rel.head())
