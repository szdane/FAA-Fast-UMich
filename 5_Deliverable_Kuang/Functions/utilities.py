import numpy as np
import pandas as pd
from pathlib import Path


def load_star_fixes(csv_path):
    """Load STAR fix definitions from CSV.

    Returns:
      fixes     : dict {name: (lat, lon, (alt_max_ft, alt_min_ft))}
      lat_vals  : tuple of latitudes
      lon_vals  : tuple of longitudes
      alt_max   : tuple of max altitudes (ft)
      alt_min   : tuple of min altitudes (ft)
    """
    df = pd.read_csv(csv_path)
    fixes = {
        row['name']: (row['lat'], row['lon'], (row['alt_max_ft'], row['alt_min_ft']))
        for _, row in df.iterrows()
    }
    lat_vals, lon_vals, alt_max, alt_min = zip(*[
        (v[0], v[1], v[2][0], v[2][1]) for v in fixes.values()
    ])
    return fixes, lat_vals, lon_vals, alt_max, alt_min


def load_flights(csv_path, flights_to_optimize, dt):
    """Load and preprocess flight data from CSV.

    Returns a DataFrame with relative entry/landing times and
    the integer entry timestep for each flight.
    """
    df = pd.read_csv(csv_path)
    df = df[df['acId'].isin(flights_to_optimize)].reset_index(drop=True)

    df['entry_rectime'] = pd.to_datetime(df['entry_rectime'])
    df['exit_rectime']  = pd.to_datetime(df['exit_rectime'])
    min_time = df['entry_rectime'].min()

    df['rel_entry_time']   = (df['entry_rectime'] - min_time).dt.total_seconds()
    df['rel_landing_time'] = (df['exit_rectime']  - min_time).dt.total_seconds()

    columns = [
        'acId',
        'entry_lat', 'entry_lon', 'entry_alt', 'rel_entry_time',
        'exit_lat',  'exit_lon',  'exit_alt',  'rel_landing_time',
    ]
    flights = df[columns].copy()
    flights['flight_entry_timestep'] = (flights['rel_entry_time'] / dt).astype(int)
    return flights


def compute_time_grid(flights, n_steps_budget):
    """Compute the time grid size for the MILP horizon.

    The grid is sized so that the latest-entering flight still has exactly
    n_steps_budget active steps available after it enters. All earlier flights
    will also be capped at n_steps_budget via the step-budget constraint.

    Returns (N_steps, max_entry_k):
      - N_steps     : total number of time steps in the grid
      - max_entry_k : entry timestep of the latest-entering flight
    """
    # Latest entry step across all flights
    max_entry_k = int(flights['flight_entry_timestep'].max())
    # Grid ends one step after the last flight's budget is exhausted
    N_steps = max_entry_k + n_steps_budget + 1
    return N_steps, max_entry_k


def load_flights_utc(csv_path, flights_to_optimize, dt, grid_epoch=None):
    """Load and preprocess flight data aligned to a standardized UTC time grid.

    Unlike load_flights (which computes entry timesteps relative to the earliest
    flight in the batch), this function anchors all times to a shared UTC epoch —
    by default, midnight UTC of the earliest entry day.  This makes step index k
    a universal UTC-clock address: k=0 is always midnight, k=1 is 00:05, etc.,
    regardless of which flights are selected.  Any future flight on the same day
    maps into the same infinite grid without reindexing.

    Args:
        csv_path            : Path to entry_exit_points.csv
        flights_to_optimize : list of acId strings to include
        dt                  : time step size in seconds (e.g. 300 for 5 min)
        grid_epoch          : pd.Timestamp anchor for k=0; defaults to midnight
                              of the earliest entry day in the selected flights

    Returns:
        flights    : DataFrame with UTC-anchored flight_entry_timestep
        grid_epoch : pd.Timestamp used as k=0 (tz-naive, conceptually UTC)
    """
    df = pd.read_csv(csv_path)
    df = df[df['acId'].isin(flights_to_optimize)].reset_index(drop=True)

    df['entry_rectime'] = pd.to_datetime(df['entry_rectime'])
    df['exit_rectime']  = pd.to_datetime(df['exit_rectime'])

    if grid_epoch is None:
        # Midnight UTC of the earliest entry day.
        # All flights (even those on later dates) get positive entry_epoch_s
        # measured from this single anchor, so multi-day batches are handled
        # correctly. Note: if flights span many days, N_steps grows large and
        # the MILP will become expensive — keep batches within ~1 day.
        earliest = df['entry_rectime'].min()
        grid_epoch = earliest.normalize()

    n_days = (df['entry_rectime'].max().normalize() - grid_epoch).days
    if n_days > 1:
        import warnings
        warnings.warn(
            f"Flights span {n_days} days from epoch {grid_epoch.date()}. "
            "This produces a large k_entry and a costly MILP. "
            "Consider passing a grid_epoch closer to the actual flight window.",
            stacklevel=2,
        )

    df['entry_epoch_s'] = (df['entry_rectime'] - grid_epoch).dt.total_seconds()
    df['exit_epoch_s']  = (df['exit_rectime']  - grid_epoch).dt.total_seconds()

    columns = [
        'acId',
        'entry_lat', 'entry_lon', 'entry_alt', 'entry_epoch_s',
        'exit_lat',  'exit_lon',  'exit_alt',  'exit_epoch_s',
    ]
    flights = df[columns].copy()
    flights['flight_entry_timestep'] = (flights['entry_epoch_s'] / dt).astype(int)
    return flights, grid_epoch


def print_chosen_star_fixes(flights, fix_sel, fix_names, star_fixes, f_alt, N_steps):
    """Print the MILP-chosen STAR fix (name, lat, lon, final altitude) for each flight."""
    print('Chosen STAR fixes for each flight:')
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        for k, name in enumerate(fix_names):
            if fix_sel[i][k].X > 0.5:
                fix_lat, fix_lon, _ = star_fixes[name]
                final_alt = f_alt[i][N_steps - 1].X
                end = '\n\n' if i == len(flights) - 1 else '\n'
                print(f'  {flight_id}: {name} (lat={fix_lat:.4f}, lon={fix_lon:.4f}, alt={final_alt:.0f} ft)', end=end)
                break


def print_exit_times(flights, k_arrive, grid_epoch_utc, timestep_dt):
    """Print the optimized exit step index and corresponding UTC time for each flight."""
    print('Optimized exit_k by flight:')
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        k_exit    = int(round(k_arrive[i].X))
        exit_utc  = grid_epoch_utc + pd.Timedelta(seconds=k_exit * timestep_dt)
        end = '\n\n' if i == len(flights) - 1 else '\n'
        print(f'  {flight_id}: exit_k={k_exit}, exit_utc={exit_utc.strftime("%H:%M:%S")} UTC', end=end)


def save_trajectory_csv(f_lat, f_lon, f_alt, N_flights, N_steps, grid_epoch_utc, timestep_dt, output_dir, flights=None, k_arrive=None):
    """Build the wide-format trajectory DataFrame and save it to df_wide.csv.

    If flights and k_arrive are provided, adds a per-flight status column:
      0 = before entry  (k < k_entry)
      1 = in flight     (k_entry <= k < k_arrive)
      2 = arrived       (k >= k_arrive)

    Returns the DataFrame.
    """
    rows = []
    for k in range(N_steps):
        t_utc = grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)
        row = {
            "t":      k * timestep_dt,
            "t_step": k,
            "t_utc":  t_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        for i in range(N_flights):
            row[f"f{i+1}_lat"]    = f_lat[i][k].X
            row[f"f{i+1}_lon"]    = f_lon[i][k].X
            row[f"f{i+1}_alt_ft"] = f_alt[i][k].X
            if flights is not None and k_arrive is not None:
                k_entry_i  = int(flights.iloc[i]['flight_entry_timestep'])
                k_arrive_i = int(round(k_arrive[i].X))
                row[f"f{i+1}_status"] = 0 if k < k_entry_i else (1 if k < k_arrive_i else 2)
        rows.append(row)
    df_wide = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "df_wide.csv"
    df_wide.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}\n")
    return df_wide


def print_waypoint_table(flights, f_lat, f_lon, f_alt, fix_reached, sep_bypass, N_steps, grid_epoch_utc, timestep_dt):
    """Print the per-flight waypoint table with step, UTC time, position, and status flags."""
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        k_entry   = int(flights.iloc[i]['flight_entry_timestep'])
        lines = [
            f"  Waypoints for {flight_id}:",
            f"  {'Step':>4}  {'UTC Time':>8}  {'Lat':>10}  {'Lon':>11}  {'Alt (ft)':>10}  {'fix_reached':>11}  {'sep_bypass':>10}",
            f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*11}  {'-'*10}  {'-'*11}  {'-'*10}",
        ]
        for k in range(N_steps):
            t_utc_str = (grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)).strftime("%H:%M")
            lat_val   = f_lat[i][k].X
            lon_val   = f_lon[i][k].X
            alt_val   = f_alt[i][k].X
            end_val   = int(round(fix_reached[i, k].X)) if k >= k_entry else 0
            land_val  = int(round(sep_bypass[i, k].X))
            marker    = " <-- entry" if k == k_entry else (" <-- ARRIVED" if end_val == 1 and (k == 0 or int(round(fix_reached[i, k-1].X)) == 0) else "")
            lines.append(f"  {k:>4}  {t_utc_str:>8}  {lat_val:>10.4f}  {lon_val:>11.4f}  {alt_val:>10.1f}  {end_val:>11}  {land_val:>10}{marker}")
        print('\n'.join(lines) + '\n')


def print_separation_check(flights, f_lat, f_lon, f_alt, sep_bypass, N_flights, N_steps, sep_hor_nm, sep_vert_ft):
    """Print the pairwise separation diagnostic table for all active flight pairs."""
    flight_ids = [flights.iloc[i]['acId'] for i in range(N_flights)]
    lines = [
        "  --- Pairwise separation check (active flights only) ---",
        f"  {'Step':>4}  {'Pair':<40}  {'dLat':>8}  {'dLon':>8}  {'dAlt':>8}  {'Sep OK?'}",
        f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*16}",
    ]
    for k in range(N_steps):
        for i in range(N_flights - 1):
            for j in range(i + 1, N_flights):
                if int(round(sep_bypass[i, k].X)) == 1 or int(round(sep_bypass[j, k].X)) == 1:
                    continue
                dlat = abs(f_lat[i][k].X - f_lat[j][k].X)
                dlon = abs(f_lon[i][k].X - f_lon[j][k].X)
                dalt = abs(f_alt[i][k].X - f_alt[j][k].X)
                sep_ok = (dlat >= sep_hor_nm or dlon >= sep_hor_nm or dalt >= sep_vert_ft)
                pair   = f"{flight_ids[i]} vs {flight_ids[j]}"
                flag   = "OK" if sep_ok else "*** VIOLATION ***"
                lines.append(f"  {k:>4}  {pair:<40}  {dlat:>8.4f}  {dlon:>8.4f}  {dalt:>8.1f}  {flag}")
    print('\n'.join(lines) + '\n')


def save_decision_variables_csv(
    flights, fix_sel, fix_names, k_arrive, N_steps,
    f_lat, f_lon, f_alt, fix_reached, fix_enters, sep_bypass,
    d_lat, d_lon, d_alt, u_lat, u_lon, u_alt,
    speed_2d, dd_lat, dd_lon, u_dd_lat, u_dd_lon, accel_cost,
    w_smooth, w_accel, w_z, grid_epoch_utc, timestep_dt, output_dir,
):
    """Extract per-step decision variables and cost terms for all flights and save to decision_variables.csv."""
    n_fixes = len(fix_names)
    chosen_fix_per_flight = []
    for i in range(len(flights)):
        cf = '?'
        for kf in range(n_fixes):
            if fix_sel[i][kf].X > 0.5:
                cf = fix_names[kf]
                break
        chosen_fix_per_flight.append(cf)

    var_rows = []
    for i in range(len(flights)):
        flight_id  = flights.iloc[i]['acId']
        entry_k_i  = int(flights.iloc[i]['flight_entry_timestep'])
        exit_k_val = int(round(k_arrive[i].X))
        chosen_fix = chosen_fix_per_flight[i]

        for k in range(N_steps):
            active_step = k >= entry_k_i + 1   # displacement variables defined
            accel_step  = k >= entry_k_i + 2   # heading-rate variables defined

            smooth = w_smooth * (u_lat[i][k-1].X + u_lon[i][k-1].X + w_z * u_alt[i][k-1].X) if active_step else 0.0
            accel  = w_accel  * accel_cost[i, k].X                                            if accel_step  else 0.0

            var_rows.append({
                # --- Identity ---
                'flight':      flight_id,
                'step':        k,
                't_utc':       (grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                't_epoch_s':   k * timestep_dt,
                'entry_k':     entry_k_i,
                'exit_k':      exit_k_val,
                'chosen_fix':  chosen_fix,
                # --- State flags ---
                'fix_reached': int(round(fix_reached[i, k].X)) if k >= entry_k_i else 0,
                'fix_enters':  int(round(fix_enters[i, k].X))  if k >= entry_k_i else 0,
                'sep_bypass':  int(round(sep_bypass[i, k].X)),
                # --- Position ---
                'lat':         f_lat[i][k].X,
                'lon':         f_lon[i][k].X,
                'alt_ft':      f_alt[i][k].X,
                # --- Displacements (step k vs k-1) ---
                'dx':          d_lat[i, k].X  if active_step else 0.0,
                'dy':          d_lon[i, k].X  if active_step else 0.0,
                'dz':          d_alt[i, k].X  if active_step else 0.0,
                # --- Absolute displacements (L1 terms) ---
                'ux':          u_lat[i][k-1].X if active_step else 0.0,
                'uy':          u_lon[i][k-1].X if active_step else 0.0,
                'uz':          u_alt[i][k-1].X if active_step else 0.0,
                # --- Speed (2-D, degrees/step) ---
                'speed_2d':    speed_2d[i, k].X  if active_step else 0.0,
                # --- Heading-rate (direction change, step k vs k-1) ---
                'ddx':         dd_lat[i, k].X    if accel_step else 0.0,
                'ddy':         dd_lon[i, k].X    if accel_step else 0.0,
                'udx':         u_dd_lat[i, k].X  if accel_step else 0.0,
                'udy':         u_dd_lon[i, k].X  if accel_step else 0.0,
                'accel_xy':    accel_cost[i, k].X if accel_step else 0.0,
                # --- Cost terms ---
                'cost_smooth': smooth,
                'cost_accel':  accel,
                'cost_total':  smooth + accel,
            })

    df_vars   = pd.DataFrame(var_rows)
    vars_path = output_dir / 'decision_variables.csv'
    df_vars.to_csv(vars_path, index=False)
    print(f'Decision variables saved to {vars_path}\n')


# ══════════════════════════════════════════════════════════════════════════════
# XYZ (SI-unit) variants — project lat/lon/alt to local Cartesian metres
# Convention: x = east (m), y = north (m), z = altitude (m MSL)
# Projection origin: DTW airport (lat0, lon0) passed to every xyz function
# Original lat/lon functions above are UNCHANGED so main_v8.py etc. still work
# ══════════════════════════════════════════════════════════════════════════════

def _latlon_to_xy(lat, lon, lat0, lon0):
    """Convert geographic (lat, lon) to local Cartesian (x east, y north) in metres."""
    import openap
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    bearings  = openap.aero.bearing(lat0, lon0, lat, lon) / 180.0 * np.pi
    distances = openap.aero.distance(lat0, lon0, lat, lon)
    return distances * np.sin(bearings), distances * np.cos(bearings)


def _xy_to_latlon(x, y, lat0, lon0):
    """Convert local Cartesian (x east, y north) in metres back to geographic (lat, lon)."""
    import openap
    x, y = np.asarray(x, float), np.asarray(y, float)
    dist    = np.sqrt(x**2 + y**2)
    bearing = np.degrees(np.arctan2(x, y))   # arctan2(east, north) → compass bearing
    return openap.aero.latlon(lat0, lon0, dist, bearing)


def load_star_fixes_xyz(csv_path, lat0, lon0):
    """Load STAR fixes projected to local Cartesian (x, y in metres; z in metres).

    Returns:
      fixes_xyz  : dict {name: (x_m, y_m, (z_max_m, z_min_m))}
      x_vals     : tuple of x (east, m)
      y_vals     : tuple of y (north, m)
      z_max_vals : tuple of max altitudes (m)
      z_min_vals : tuple of min altitudes (m)
    """
    df = pd.read_csv(csv_path)
    fixes_xyz = {}
    for _, row in df.iterrows():
        x, y = _latlon_to_xy(row['lat'], row['lon'], lat0, lon0)
        fixes_xyz[row['name']] = (
            float(x), float(y),
            (row['alt_max_ft'] * 0.3048, row['alt_min_ft'] * 0.3048)
        )
    x_vals, y_vals, z_max_vals, z_min_vals = zip(*[
        (v[0], v[1], v[2][0], v[2][1]) for v in fixes_xyz.values()
    ])
    return fixes_xyz, x_vals, y_vals, z_max_vals, z_min_vals


def load_flights_utc_xyz(csv_path, flights_to_optimize, dt, lat0, lon0, grid_epoch=None):
    """Like load_flights_utc but converts positions to metres (entry_x/y/z, exit_x/y/z).

    Altitude is converted from feet to metres. x/y are projected from DTW (lat0, lon0).
    """
    df = pd.read_csv(csv_path)
    df = df[df['acId'].isin(flights_to_optimize)].reset_index(drop=True)
    df['entry_rectime'] = pd.to_datetime(df['entry_rectime'])
    df['exit_rectime']  = pd.to_datetime(df['exit_rectime'])

    if grid_epoch is None:
        grid_epoch = df['entry_rectime'].min().normalize()

    n_days = (df['entry_rectime'].max().normalize() - grid_epoch).days
    if n_days > 1:
        import warnings
        warnings.warn(
            f"Flights span {n_days} days from epoch {grid_epoch.date()}. "
            "This produces a large k_entry and a costly MILP. "
            "Consider passing a grid_epoch closer to the actual flight window.",
            stacklevel=2,
        )

    df['entry_epoch_s'] = (df['entry_rectime'] - grid_epoch).dt.total_seconds()
    df['exit_epoch_s']  = (df['exit_rectime']  - grid_epoch).dt.total_seconds()

    entry_x, entry_y = _latlon_to_xy(df['entry_lat'].values, df['entry_lon'].values, lat0, lon0)
    exit_x,  exit_y  = _latlon_to_xy(df['exit_lat'].values,  df['exit_lon'].values,  lat0, lon0)
    df['entry_x'] = entry_x;  df['entry_y'] = entry_y;  df['entry_z'] = df['entry_alt'] * 0.3048
    df['exit_x']  = exit_x;   df['exit_y']  = exit_y;   df['exit_z']  = df['exit_alt']  * 0.3048

    cols = ['acId', 'entry_x', 'entry_y', 'entry_z', 'entry_epoch_s',
                    'exit_x',  'exit_y',  'exit_z',  'exit_epoch_s']
    flights = df[cols].copy()
    flights['flight_entry_timestep'] = (flights['entry_epoch_s'] / dt).astype(int)
    return flights, grid_epoch


def save_trajectory_csv_xyz(f_x, f_y, f_z, N_flights, N_steps, grid_epoch_utc, timestep_dt,
                             output_dir, lat0, lon0, flights=None, k_arrive=None):
    """Save trajectory with x/y/z (metres) as the primary columns.

    Also writes back-converted f{i}_lat, f{i}_lon, f{i}_alt_ft columns for human
    readability and backward compatibility with tools that read df_wide.csv directly.
    analyze_optimized_trajectory_xyz performs its own back-conversion from x/y/z
    independently and does not rely on these extra columns.
    Returns the wide-format DataFrame.
    """
    rows = []
    for k in range(N_steps):
        t_utc = grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)
        row = {"t": k * timestep_dt, "t_step": k,
               "t_utc": t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")}
        for i in range(N_flights):
            xv, yv, zv = f_x[i][k].X, f_y[i][k].X, f_z[i][k].X
            row[f"f{i+1}_x"]      = xv
            row[f"f{i+1}_y"]      = yv
            row[f"f{i+1}_z"]      = zv
            lat_v, lon_v = _xy_to_latlon(xv, yv, lat0, lon0)
            row[f"f{i+1}_lat"]    = float(lat_v)
            row[f"f{i+1}_lon"]    = float(lon_v)
            row[f"f{i+1}_alt_ft"] = zv / 0.3048
            if flights is not None and k_arrive is not None:
                k_entry_i  = int(flights.iloc[i]['flight_entry_timestep'])
                k_arrive_i = int(round(k_arrive[i].X))
                row[f"f{i+1}_status"] = 0 if k < k_entry_i else (1 if k < k_arrive_i else 2)
        rows.append(row)
    df_wide = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "df_wide.csv"
    df_wide.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}\n")
    return df_wide


def print_chosen_star_fixes_xyz(flights, fix_sel, fix_names, star_fixes_xyz, f_z, N_steps):
    """Print chosen STAR fix name and position in metres."""
    print('Chosen STAR fixes for each flight:')
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        for k, name in enumerate(fix_names):
            if fix_sel[i][k].X > 0.5:
                fx, fy, _ = star_fixes_xyz[name]
                final_z   = f_z[i][N_steps - 1].X
                end = '\n\n' if i == len(flights) - 1 else '\n'
                print(f'  {flight_id}: {name} (x={fx:.0f} m, y={fy:.0f} m, z={final_z:.0f} m)', end=end)
                break


def print_waypoint_table_xyz(flights, f_x, f_y, f_z, fix_reached, sep_bypass,
                              N_steps, grid_epoch_utc, timestep_dt):
    """Print per-flight waypoint table with positions in metres."""
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        k_entry   = int(flights.iloc[i]['flight_entry_timestep'])
        lines = [
            f"  Waypoints for {flight_id}:",
            f"  {'Step':>4}  {'UTC':>8}  {'x (m)':>12}  {'y (m)':>12}  {'z (m)':>10}  {'fix_reached':>11}  {'sep_bypass':>10}",
            f"  {'-'*4}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*11}  {'-'*10}",
        ]
        for k in range(N_steps):
            t_str    = (grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)).strftime("%H:%M")
            xv, yv, zv = f_x[i][k].X, f_y[i][k].X, f_z[i][k].X
            end_val  = int(round(fix_reached[i, k].X)) if k >= k_entry else 0
            land_val = int(round(sep_bypass[i, k].X))
            marker   = (" <-- entry" if k == k_entry else
                        (" <-- ARRIVED" if end_val == 1 and
                         (k == 0 or int(round(fix_reached[i, k-1].X)) == 0) else ""))
            lines.append(
                f"  {k:>4}  {t_str:>8}  {xv:>12.1f}  {yv:>12.1f}  {zv:>10.1f}"
                f"  {end_val:>11}  {land_val:>10}{marker}"
            )
        print('\n'.join(lines) + '\n')


def print_separation_check_xyz(flights, f_x, f_y, f_z, sep_bypass,
                                N_flights, N_steps, sep_hor_m, sep_vert_m):
    """Print pairwise separation diagnostic in metres."""
    flight_ids = [flights.iloc[i]['acId'] for i in range(N_flights)]
    lines = [
        "  --- Pairwise separation check (active flights only) ---",
        f"  {'Step':>4}  {'Pair':<40}  {'dx (m)':>10}  {'dy (m)':>10}  {'dz (m)':>10}  Sep OK?",
        f"  {'-'*4}  {'-'*40}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*16}",
    ]
    for k in range(N_steps):
        for i in range(N_flights - 1):
            for j in range(i + 1, N_flights):
                if int(round(sep_bypass[i, k].X)) or int(round(sep_bypass[j, k].X)):
                    continue
                dx = abs(f_x[i][k].X - f_x[j][k].X)
                dy = abs(f_y[i][k].X - f_y[j][k].X)
                dz = abs(f_z[i][k].X - f_z[j][k].X)
                ok   = dx >= sep_hor_m or dy >= sep_hor_m or dz >= sep_vert_m
                pair = f"{flight_ids[i]} vs {flight_ids[j]}"
                lines.append(
                    f"  {k:>4}  {pair:<40}  {dx:>10.1f}  {dy:>10.1f}  {dz:>10.1f}"
                    f"  {'OK' if ok else '*** VIOLATION ***'}"
                )
    print('\n'.join(lines) + '\n')


def save_decision_variables_csv_xyz(
    flights, fix_sel, fix_names, k_arrive, N_steps,
    f_x, f_y, f_z, fix_reached, fix_enters, sep_bypass,
    d_x, d_y, d_z, u_x, u_y, u_z,
    speed_2d, dd_x, dd_y, u_dd_x, u_dd_y, accel_cost,
    w_smooth, w_accel, w_z_weight, grid_epoch_utc, timestep_dt, output_dir,
):
    """Save per-step decision variables in SI units (metres) to decision_variables.csv."""
    n_fixes = len(fix_names)
    chosen = ['?' for _ in range(len(flights))]
    for i in range(len(flights)):
        for kf in range(n_fixes):
            if fix_sel[i][kf].X > 0.5:
                chosen[i] = fix_names[kf]; break

    rows = []
    for i in range(len(flights)):
        flight_id  = flights.iloc[i]['acId']
        entry_k_i  = int(flights.iloc[i]['flight_entry_timestep'])
        exit_k_val = int(round(k_arrive[i].X))
        for k in range(N_steps):
            active = k >= entry_k_i + 1
            acc    = k >= entry_k_i + 2
            smooth = w_smooth * (u_x[i][k-1].X + u_y[i][k-1].X + w_z_weight * u_z[i][k-1].X) if active else 0.0
            accel  = w_accel  * accel_cost[i, k].X if acc else 0.0
            rows.append({
                'flight':    flight_id,
                'step':      k,
                't_utc':     (grid_epoch_utc + pd.Timedelta(seconds=k * timestep_dt)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                't_epoch_s': k * timestep_dt,
                'entry_k': entry_k_i, 'exit_k': exit_k_val, 'chosen_fix': chosen[i],
                'fix_reached': int(round(fix_reached[i, k].X)) if k >= entry_k_i else 0,
                'fix_enters':  int(round(fix_enters[i, k].X))  if k >= entry_k_i else 0,
                'sep_bypass':  int(round(sep_bypass[i, k].X)),
                'x_m': f_x[i][k].X, 'y_m': f_y[i][k].X, 'z_m': f_z[i][k].X,
                'dx': d_x[i, k].X  if active else 0.0,
                'dy': d_y[i, k].X  if active else 0.0,
                'dz': d_z[i, k].X  if active else 0.0,
                'ux': u_x[i][k-1].X if active else 0.0,
                'uy': u_y[i][k-1].X if active else 0.0,
                'uz': u_z[i][k-1].X if active else 0.0,
                'speed_2d_m_per_step': speed_2d[i, k].X if active else 0.0,
                'ddx': dd_x[i, k].X   if acc else 0.0,
                'ddy': dd_y[i, k].X   if acc else 0.0,
                'udx': u_dd_x[i, k].X if acc else 0.0,
                'udy': u_dd_y[i, k].X if acc else 0.0,
                'accel_xy':    accel_cost[i, k].X if acc else 0.0,
                'cost_smooth': smooth, 'cost_accel': accel, 'cost_total': smooth + accel,
            })
    df_vars   = pd.DataFrame(rows)
    vars_path = output_dir / 'decision_variables.csv'
    df_vars.to_csv(vars_path, index=False)
    print(f'Decision variables saved to {vars_path}\n')
