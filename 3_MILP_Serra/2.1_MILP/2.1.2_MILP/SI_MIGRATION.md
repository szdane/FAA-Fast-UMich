# Debug model: units and SI migration

Both `main_v15.py` here and `8_Deliverable_Kuang/main_v15.py` use local east/north
coordinates in metres from DTW (42.2125, -83.3534), altitude in metres MSL, and
time in seconds. Their geographic projection uses OpenAP distance and bearing.

| Quantity | Existing debug script | main_v15 / new SI script |
| --- | --- | --- |
| Horizontal positions | x = latitude degrees, y = longitude degrees | x = east metres, y = north metres |
| Altitude | feet | metres |
| Time step | 300 seconds | main_v15: 400 seconds; SI debug: 300 seconds |
| Horizontal separation | 500 / 6076.12 = 0.082289… **NM**, applied to degrees | 500 ft = **152.4 m** |
| Vertical separation | 100 ft | **30.48 m** |
| Vertical speed limit | 1000/60 ft/s | **5.08 m/s** |
| Horizontal speed limits | 0.14/60 deg latitude/s; 0.072/60 deg longitude/s | main_v15: 300 m/s per axis; SI debug: approximately 259.455 m/s north, 98.829 m/s east |
| Per-step displacement | degree differences / feet | metres (not m/s; divide by dt for speed) |

The SI debug version approximates the original component speed limits at DTW's
reference latitude. This preserves their intent; projection curvature means it
is not an exact transformation of the original latitude/longitude constraints.
It does not adopt main_v15's 300 m/s limit, timestep, or flight selection.

## What was wrong

1. A nautical-mile separation cannot be compared with latitude/longitude degree
   differences. Also, a degree of longitude and a degree of latitude have
   different physical lengths. The old 0.082289 degree threshold corresponds
   to roughly 9.15 km north/south or 6.78 km east/west near DTW, instead of 152.4 m.
2. The distance objective added degree changes to vertical changes in NM.
   The new objective measures all three axes in metres and divides by 1852,
   retaining the original CT/CF per-step cost. The horizontal/vertical tradeoff
   changes deliberately; old objective values and optimal paths are not directly
   comparable.
3. One BIG_M was used in linear coordinate inequalities and squared-distance
   inequalities. Those need different units (length versus length squared),
   independently derived valid bounds, and adequate scaling.
4. With M=100000 and default integrality tolerance 1e-5, the product M*tolerance
   is 1. That is a useful warning scale relative to a 0.082289 threshold,
   not evidence that a particular returned solution has that violation.
5. Explicit slopes can become very large near horizontal/vertical segments.
   Changing degrees to metres alone does not fix that.

The existing inputs indicate **500 ft horizontal and 100 ft vertical**, which
the migration retains. These are the code's configured values, not a claim
about operational separation requirements.

## Files and behavior

- `milp_multiple_Debug_SI.py`: independent, runnable migration; existing edited
  files are untouched. Converts entry coordinates and STAR fixes before model
  construction. Weather is omitted, matching the original's commented-out
  weather constraints. The existing inactive fuel calculations are omitted.
- `capsule_geometry_si.py`: bounded unit direction vectors replace scalar slopes;
  linear indicators gate distances/projections in metres. Quadratic equalities
  define nonlinear quantities before indicator constraints are applied. There
  is no explicit separation BIG_M; Gurobi may reformulate indicators internally.
- `test_capsule_geometry_si.py`: direction, strip, endpoint, stationary,
  disabled-branch, octagon diagonal, and projection round-trip checks.

The octagonal waypoint clearance is retained. Vertical separation is split
into two linear indicator branches instead of a squared-distance inequality.
Position bounds follow from entry position, elapsed time, and component speed
limits. Entry `is_end` flags are fixed to zero: the original unconstrained flags
could bypass separation at entry. Arrival still bypasses separation as in the
original; this is not a change to the final-waypoint separation policy.

This is a **nonconvex MIQCP**, despite the MILP filename. Indicators can improve
numerical behavior but do not guarantee faster solving. A unit-consistent model
can still be slow or infeasible. No original numerical failure was reproduced.

The two directional capsule tests remain alternatives (OR), following the
original formulation. They do **not** establish continuous-time separation of
both moving aircraft between waypoints; neither would simply requiring both
capsule tests. That requires a separate relative-motion formulation/check.

## Run

From this directory, using Python with `gurobipy`, `numpy`, `pandas`, and `openap`:

```sh
python3 -m unittest test_capsule_geometry_si.py
python3 milp_multiple_Debug_SI.py --build-only
python3 milp_multiple_Debug_SI.py --time-limit 60
```

Input/output paths are anchored to the script directory. Output is
`../../2.3_Outputs_and_Results/test_si.csv`, leaving `test.csv` intact. It includes
the original lat/lon/alt_ft column convention plus x_m/y_m/z_m columns. Time is
seconds from the earliest selected entry; the epoch is printed.

The default first two flights and 48000-second horizon match the debug script.
For a shorter experiment, explicitly use `--horizon-seconds 1800`; the model
still extends through the latest selected recorded exit. Reducing the horizon
changes the feasible problem. A time limit may return a feasible incumbent
without proving optimality; inspect the solver status, gap, and quality report.

## Verification performed

- Five test methods passed, including 60 moving-direction/endpoint probes,
  nine stationary shape probes, two disabled branches, diagonal octagon checks,
  and geographic projection round trips.
- Default two-flight model construction passed: 12536 variables, 1920 quadratic
  constraints, 9298 general constraints. Reported linear matrix coefficients
  range from 1 to about 8e4; quadratic coefficients are 1 in magnitude.
- Small models solved: a stationary single flight at its STAR fix is feasible;
  two aircraft entering at the same position and altitude are infeasible.
- A 15-second solve of the default two-flight model returned a feasible incumbent
  with maximum reported constraint violation about **1.69e-9**, zero reported
  bound/integrality violation, and a **48.8% optimality gap**. CSV export to a
  temporary file passed. This demonstrates a feasible solve, not optimality
  or a performance improvement over the original. The 48000-second horizon
  gives broad reachable bounds.

Gurobi's guidance explains the interaction between large M and integrality
tolerance, and the tradeoffs of indicators:
https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html
