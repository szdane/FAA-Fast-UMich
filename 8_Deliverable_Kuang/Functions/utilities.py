import numpy as np
import pandas as pd
from pathlib import Path


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


def print_exit_times(flights, k_arrive, grid_epoch_utc, timestep_dt):
    """Print the optimized exit step index and corresponding UTC time for each flight."""
    print('Optimized exit_k by flight:')
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        k_exit    = int(round(k_arrive[i].X))
        exit_utc  = grid_epoch_utc + pd.Timedelta(seconds=k_exit * timestep_dt)
        end = '\n\n' if i == len(flights) - 1 else '\n'
        print(f'  {flight_id}: exit_k={k_exit}, exit_utc={exit_utc.strftime("%H:%M:%S")} UTC', end=end)


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
    """Load flight entry/exit points and convert to local Cartesian (x, y, z in metres).
    Input CSV must contain columns:
      acId, entry_lat, entry_lon, entry_alt, entry_rectime,
        exit_lat, exit_lon, exit_alt, exit_rectime
    Output:
        flights : DataFrame with columns
            acId, entry_x, entry_y, entry_z, entry_epoch_s,
                exit_x,  exit_y,  exit_z,  exit_epoch_s,
                flight_entry_timestep
        grid_epoch : datetime of the grid epoch (k=0) in UTC
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


def print_flight_results(flights, fix_sel, fix_names, star_fixes_xyz, f_z, N_steps,
                          k_arrive, grid_epoch_utc, timestep_dt):
    """Print one combined table: chosen STAR fix, position, exit step, and exit UTC per flight."""
    SEP = "=" * 52
    col_header = f"  {'Flight':<26} {'STAR fix':<10} {'x (m)':>10} {'y (m)':>10} {'z (m)':>8}  {'exit_k':>6}  {'exit_utc':>12}"
    print(f"\n{SEP}")
    print(f"  FLIGHT RESULTS")
    print(SEP)
    print(col_header)
    for i in range(len(flights)):
        flight_id = flights.iloc[i]['acId']
        k_exit    = int(round(k_arrive[i].X))
        exit_utc  = grid_epoch_utc + pd.Timedelta(seconds=k_exit * timestep_dt)
        final_z   = f_z[i][N_steps - 1].X
        for k, name in enumerate(fix_names):
            if fix_sel[i][k].X > 0.5:
                fx, fy, _ = star_fixes_xyz[name]
                print(f"  {flight_id:<26} {name:<10} {fx:>10.0f} {fy:>10.0f} {final_z:>8.0f}  {k_exit:>6}  {exit_utc.strftime('%H:%M:%S UTC'):>12}")
                break
    print(f"{SEP}\n")


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


def select_cluster(eep_csv, query_time, cluster_window_min):
    """Select flights entering within cluster_window_min of query_time (HH:MM UTC).
    Returns (flights_to_optimize, cluster, t_start, t_end, span_min)."""
    clusters = _cluster_flights_by_time(eep_csv, cluster_window_min)  # groups flights into clusters where all entry times fall within the sliding window; list of [(acId, entry_time_utc)] lists
    cluster  = _find_cluster_for_query(clusters, query_time, cluster_window_min)  # returns the cluster whose window contains query_time, or None if nothing matches
    if cluster is None or len(cluster) == 0:  # guard against no match
        avail = ", ".join(
            f"{c[0][1].strftime('%H:%M')} ({len(c)} flights)"
            for c in clusters
        )
        raise ValueError(
            f"No cluster found for '{query_time}' UTC.\n"
            f"Available cluster start times (UTC): {avail}"
        )
    t_start             = cluster[0][1]   # earliest entry time in the cluster
    t_end               = cluster[-1][1]  # latest entry time in the cluster
    span_min            = (t_end - t_start).total_seconds() / 60  # span in minutes
    flights_to_optimize = [ac for ac, _ in cluster]
    return flights_to_optimize, cluster, t_start, t_end, span_min


def _cluster_flights_by_time(csv_path, window_min):
    """Group flights into time clusters using a sliding window on entry_rectime."""
    df = pd.read_csv(csv_path)
    df["entry_rectime"] = pd.to_datetime(df["entry_rectime"], format="mixed")
    entries = df.set_index("acId")["entry_rectime"].sort_values()
    window  = pd.Timedelta(minutes=window_min)
    clusters, cur, cur_start = [], [], None
    for acId, t in entries.items():
        if cur_start is None or (t - cur_start) > window:
            if cur:
                clusters.append(cur)
            cur, cur_start = [(acId, t)], t
        else:
            cur.append((acId, t))
    if cur:
        clusters.append(cur)
    return clusters

def _find_cluster_for_query(clusters, query_hhmm, window_min):
    """Find the cluster whose first-entry time-of-day is nearest to query_hhmm (HH:MM UTC)."""
    h, m  = map(int, query_hhmm.strip().split(":"))
    q_sec = h * 3600 + m * 60
    best_cluster, best_dist = None, float("inf")
    for cluster in clusters:
        t0     = cluster[0][1]
        t0_sec = t0.hour * 3600 + t0.minute * 60 + t0.second
        dist   = abs(t0_sec - q_sec)
        if dist < best_dist:
            best_dist    = dist
            best_cluster = cluster
    return best_cluster