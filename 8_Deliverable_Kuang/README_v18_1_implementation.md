# Understanding v18.1: weather avoidance and optional airborne holding

This guide explains the current [main_v18.1.py](main_v18.1.py) in simple terms.
Code blocks below are excerpts from the implementation, not standalone programs.
Source links and section names show where to continue reading.

## 1. The idea in one minute

**Find routes from aircraft entry points to STAR fixes while avoiding weather and
other aircraft, then compare whether allowing airborne holding changes the result.**

A STAR fix is a named arrival waypoint. Reaching it is the end of this model's
route, not a simulated runway landing.

The program solves two cases:

| Case | What the aircraft may do |
|---|---|
| Holding OFF | Fly transit segments to a selected STAR; no holding circuits. |
| Holding ON | Fly transit segments and optionally spend intervals in holding circuits. Zero holds is still allowed. |

Both cases use the same input aircraft, weather, objective weights, and model
constraints. The holding budget changes from zero to allowing holding.

```text
Read command-line options and flight/STAR data
    -> Set each aircraft's clock and create the weather regions
    -> Build position, arrival, STAR-choice, holding, and fuel variables
    -> Add costs and constraints
    -> Find an initial route guess
    -> Solve holding OFF and save/check its result
    -> Use the OFF solution as a starting guess for holding ON
    -> Solve holding ON and save/check its result
    -> Write the comparison and plots
```

Gurobi chooses the variables together. The code does not first pick a final route
and then separately decide where to hold.

## 2. Which files matter?

| File | Responsibility | Read first for |
|---|---|---|
| [main_v18.1.py](main_v18.1.py) | Scenario setup, main model, and OFF/ON solve loop | Overall flow |
| [Functions/free_holding.py](Functions/free_holding.py) | Holding intervals, weather avoidance, separation, starting guesses, and holding reporting | How holding and weather actually work |
| [Functions/fuel_pwl_v1.py](Functions/fuel_pwl_v1.py) | Speed approximation and OpenAP fuel interpolation | How fuel enters a linear optimization model |
| [Functions/holding_output.py](Functions/holding_output.py) | Extract results, validate them, and produce tables/plots/animation | What saved results mean |
| [Functions/utilities.py](Functions/utilities.py) | Load/project flight and STAR data and select a traffic cluster | Input preparation |
| [test_v181.py](test_v181.py) | Regression tests of the model and reporting | Concrete examples of expected behavior |

The main script is procedural: its setup and solve code run at module level.
It is intended to be executed as a script, not casually imported. The tests
deliberately execute only its model-building portion for their fixtures.

## 3. Inputs, settings, and clocks

### Inputs and defaults

Source: [main script, setup and argument parsing](main_v18.1.py#L41).

| Setting | Current behavior |
|---|---|
| Flight data | [Input/entry_exit_points.csv](Input/entry_exit_points.csv) |
| STAR data | [Input/star_fixes.csv](Input/star_fixes.csv) |
| Traffic selection | Select a cluster near 08:11 UTC using a 15-minute clustering window, then restrict it to a fixed list of flight IDs. |
| `--flight-count` | 1, 3, or 6; default 3. Takes that many IDs from the beginning of the fixed list. |
| `--weather-regions` | 1 or 3. By default: 1 for one aircraft, otherwise 3. Three regions require at least three aircraft. |
| Interval length | 400 seconds |
| Horizon | Six intervals, seven waypoints per aircraft; up to 2400 seconds after its entry. Earlier arrival is allowed. |
| Aircraft fuel model | Every flight is treated as a B737 at a fixed 80% of maximum takeoff mass. |
| `--time-limit` | 1200 seconds per full OFF/ON solve by default; initialization and output generation take additional time. |
| Target MIP gap | 1%; this is a stopping target, not a promise that the solver reaches it before the time limit. |
| `--holding-cost-factor` | Default 1.0. Scales holding time/distance/fuel costs in the objective only. |

The six IDs are DAL1120, DAL1655, EDV5018, EDV5043, EDV5455, and NKS1449,
with their origin/destination suffixes in the code.

### Local waypoint index, shared real time

Each aircraft has its own waypoint zero at its actual entry time. Its entry
offset is retained for weather and aircraft-to-aircraft comparisons.

Source: [main script, sections 1.4-1.5](main_v18.1.py#L145).

```python
flights, GRID_EPOCH_UTC = load_flights_utc_xyz(
    csv_path, flights_to_optimize, TIMESTEP_DT, DTW_LAT0, DTW_LON0, grid_epoch=grid_start)
# Indices are local to each flight; entry_epoch_s retains its exact shared-clock offset.
flights["flight_entry_timestep"] = 0
```

The relationship is:

```text
shared time = aircraft entry offset + local waypoint index * 400 seconds
```

For example, if an aircraft enters 166 seconds after the first aircraft, its
first interval is 166-566 seconds, not 0-400 seconds. That matters when weather
is active from 400-800 seconds.

Positions are stored in meters: x east and y north relative to DTW, and z altitude.

## 4. What does Gurobi choose?

Source: [main script, section 2.2](main_v18.1.py#L240).

Here, `i` identifies an aircraft and `k` identifies one of its local waypoints.
For an interval variable at `k`, the interval runs from waypoint `k-1` to `k`.

| Variable | Plain meaning |
|---|---|
| `f_x`, `f_y`, `f_z` | Aircraft position at each waypoint |
| `fix_sel` | Which STAR the aircraft uses; exactly one must be selected |
| `fix_reached[i, k]` | 1 if the aircraft has arrived at its STAR by waypoint k |
| `fix_enters[i, k]` | 1 only at the transition from not arrived to arrived |
| `k_arrive[i]` | Local waypoint index of arrival |
| `hold[i, k]` | 1 if interval k is a holding circuit |
| `route_inactive[i, k]` | 1 if ordinary transit fuel should be disabled: holding now, or already arrived before this interval |
| `d_x`, `d_y`, `d_z` | Position changes over an interval |
| `speed_2d[i, k]` | Despite its name, approximate horizontal **distance per interval**, in meters |
| `fuel_step_kg` | Fuel for normal transit intervals |
| `hold_step_kg` | Fuel for holding intervals |

### STAR selection is a real decision, not fixed to the nearest STAR

Source: [main script, STAR constraints](main_v18.1.py#L384).

```python
for j in range(N_flights):
    m.addConstr(quicksum(fix_sel[j][i] for i in range(n_fixes)) == 1, f"one_fix{j+1}")
    m.addConstr(f_x[j][N_steps-1] == LinExpr(x_vals,     fix_sel[j].values()), f"x_choice{j+1}")
    m.addConstr(f_y[j][N_steps-1] == LinExpr(y_vals,     fix_sel[j].values()), f"y_choice{j+1}")
```

The next constraints put the final altitude within the selected STAR's allowed
range. The nearest STAR is used only to construct weather and initial guesses.
The solver can choose another STAR, and OFF and ON may choose different ones.

Once `fix_reached` becomes 1, it stays 1 and position is fixed at the destination.
Every aircraft must arrive by the final waypoint.

## 5. How airborne holding works

### A hold returns to the same position; it is not hovering

Source: [add_hold_interval](Functions/free_holding.py#L20).

```python
def add_hold_interval(model, selected, start, finish, fuel_table, speed_mps, duration, name):
    for dimension, (previous, current) in enumerate(zip(start, finish)):
        model.addConstr((selected == 1) >> (current == previous), f'{name}_closed_{dimension}')
```

When the hold binary is 1, the interval ends at its starting x, y, and altitude.
The plane is understood to fly a closed racetrack during those 400 seconds.

| Property | Implementation |
|---|---|
| Duration | One complete 400-second interval per selected hold |
| Speed | Fixed at 210 knots |
| Position and altitude | The anchor waypoint and its altitude are chosen by the optimizer; altitude stays constant during the circuit. |
| Circuit shape/orientation | Fixed construction based on speed, duration, and the shared weather basis; shape and orientation are not independently optimized. |
| Location | May occur at entry or later waypoints, subject to feasibility |
| Multiple holds | Consecutive or separated hold intervals are possible. The flight still must reach a STAR in the horizon. |
| Distance | About 43.2 km per hold, even though start-to-end displacement is zero |
| Fuel | Positive holding fuel is calculated separately from transit fuel. |

The helper evaluates OpenAP fuel flow at the holding speed and zero vertical
speed over 401 altitude sample points, then uses one-dimensional piecewise-linear
interpolation for the optimizer.

Source: [holding fuel activation](Functions/free_holding.py#L20).

```python
    model.addConstr((selected == 1) >> (fuel == duration * flow), f'{name}_active_fuel')
    model.addConstr((selected == 0) >> (fuel == 0), f'{name}_inactive_fuel')
```

### Avoid double-counting fuel

Source: [main script, holding variables](main_v18.1.py#L270).

```python
        m.addConstr(hold[i, k] + fix_reached[i, k] <= 1)
        m.addConstr(route_inactive[i, k] == fix_reached[i, k-1] + hold[i, k])
```

| Interval state | Transit fuel | Holding fuel |
|---|---|---|
| Ordinary flight, including the interval that reaches the STAR | Charged | Zero |
| Holding | Zero | Charged |
| Already arrived before the interval | Zero | Zero |

For output, [interior_hold_track](Functions/free_holding.py#L148) and
[entry_hold_track](Functions/fuel_emission_analysis_main_holding.py#L77) reconstruct
the racetrack samples. The optimizer protects the circuit's bounding area rather
than optimizing each point around the turn.

## 6. How weather avoidance works

### The weather is a constructed scenario

Source: [main script, shared weather scenario](main_v18.1.py#L158).

These are fixed rectangular obstacles with activation windows, not a live weather
feed or a wind model.

| Region | Geometry | Default active window |
|---|---|---|
| W1 | Along the first aircraft's entry-to-nearest-STAR direction, from 25% to 85% of that reference length, with 70 km on each side | 400-800 s |
| W2 | 50 km by 40 km, centered at the second aircraft's entry/nearest-STAR midpoint | 800-1200 s |
| W3 | 50 km by 40 km, centered at the third aircraft's entry/nearest-STAR midpoint | 1000-1500 s |

All boxes use the same rotated coordinate basis. Weather does not move when the
solver selects a different STAR.

`--w3-window 300 700` changes only W3's timing.
`--w3-match-w1-size` makes W3 as wide and tall as W1 without moving its center.
Both options require three weather regions.

### Any positive overlap activates protection for the whole interval

Source: [active_weather_cells](Functions/free_holding.py#L52).

```python
    return [
        (index, bounds)
        for index, (bounds, (start, end)) in enumerate(zip(scenario["weather_boxes_uv_m"], windows))
        if scenario["weather_enabled"] and start_s < end and end_s > start
    ]
```

For W1 active at 400-800 seconds:

| Aircraft interval | Weather avoidance required? |
|---|---|
| 0-400 s | No; touching the start boundary is not positive overlap. |
| 166-566 s | Yes, for the entire interval. |
| 400-800 s | Yes. |
| 800-1200 s | No; touching the end boundary is not positive overlap. |

Source: [main script, weather constraint calls](main_v18.1.py#L485).

```python
        for region, bounds in active_weather_cells(
                flight.entry_epoch_s + (k - 1) * TIMESTEP_DT,
                flight.entry_epoch_s + k * TIMESTEP_DT, scenario):
            add_weather_interval(
                m, [(f_x[i][j], f_y[i][j]) for j in (k - 1, k)], hold[i, k],
                [bounds], weather_origin, weather_basis, HOLD_SPEED_MPS,
                TIMESTEP_DT, scenario["clearance_m"], f"weather_{i}_{k}_w{region + 1}")
```

[add_weather_interval](Functions/free_holding.py#L134) makes both endpoints stay on
one common side of each active rectangle, with 1000 meters of clearance.
That also keeps the straight segment between them outside. If holding is selected,
the inequalities include the racetrack's bounding area, not just its anchor.

This is conservative: an interval cannot use the obstacle area for its
weather-free fraction if another part of that interval overlaps active weather.
The main weather loop also includes post-arrival intervals up to the horizon;
it does not switch weather constraints off using the arrival flag.

## 7. How aircraft separation and flight limits work

### Separation is checked between waypoints too

Source: [add_separation](Functions/free_holding.py#L88), called from the
[main model](main_v18.1.py#L475).

1. Find pairs of flight intervals that overlap on the shared clock.
2. Compute positions at the start and end of their overlap.
3. Require one common separating direction throughout that overlap:
   positive/negative x, y, or z.
4. Expand the protected area when either aircraft is holding.
5. Stop requiring separation for an interval after an aircraft has already arrived.
   The interval that reaches the STAR remains included.

Because transit positions are linear in time, a common separating direction at
both ends protects the intervening straight motion too. Holding uses a
conservative enclosing box.

The configured thresholds are **500 ft horizontally** and **100 ft vertically**.
These are this experiment's settings, not a claim of operational FAA approval.
The test is an axis-based separation rule, not an exact circular-distance rule.

### Other movement constraints

Source: [main script, section 2.5](main_v18.1.py#L375).

| Rule | Plain meaning |
|---|---|
| Start at recorded entry | The optimizer cannot move the initial waypoint. |
| Reach a selected STAR | Final x/y and altitude must match the STAR constraints. |
| Speed limits | Transit has a minimum speed based on 120 knots and an approximate speed capped at 300 m/s. |
| Vertical-rate limit | At most 1000 ft/min in either direction. |
| Flight-path-angle limit | Vertical movement is limited relative to horizontal movement, using a 5-degree bound. |
| Speed-change limit | Restricts changes between transit intervals; bypassed around holding/arrival as coded. |
| Heading-change bound and penalty | Restricts/penalizes changes in displacement direction; relaxed around holding transitions. |

There is **no monotone-descent constraint** in this version: climb is allowed
within its bounds. These are simplified movement rules, not full aircraft dynamics.

## 8. What does the objective minimize?

Source: [main script, section 2.4](main_v18.1.py#L309).

In simple terms:

```text
objective =
    transit time cost + transit distance cost + transit fuel cost
    + holding factor * (holding time cost + holding distance cost + holding fuel cost)
    + heading-change penalty
```

The actual time/holding part is:

```python
        active = 1 - fix_reached[i, k-1]  # Count the interval that reaches the fix.
        obj += w_time * TIMESTEP_DT * (active - (1 - HOLD_COST_FACTOR) * hold[i, k])
        obj += HOLD_COST_FACTOR * (
            w_dist * HOLD_SPEED_MPS * TIMESTEP_DT * hold[i, k] + w_fuel * hold_step_kg[i, k])
```

The previous waypoint's arrival flag is important: the final flying interval
still consumes time and fuel.

| Cost | Weight / status |
|---|---|
| Time | 1.0 |
| Horizontal transit distance and holding circuit distance | 0.1 |
| Fuel | 25.0 |
| Heading-change penalty | 0.002; zeroed for transitions involving holding |
| L1 displacement smoothness | Disabled: its objective line is commented out, despite `w_smooth` being defined. |
| Descent reward and final-altitude penalty | Both weights are zero. |

The objective is a weighted score, **not simply kilograms of fuel**. A lower
objective does not necessarily mean lower physical fuel consumption.

Changing the holding factor does not alter its duration, speed, fuel reporting,
weather clearance, or separation requirements. A factor of zero makes those
holding objective terms free, not the physical holding maneuver.

## 9. How fuel stays compatible with a MILP

Source: [FuelFlowTable](Functions/fuel_pwl_v1.py#L43) and
[add_segment](Functions/fuel_pwl_v1.py#L103).

MILP means mixed-integer linear programming: continuous quantities such as
position are combined with yes/no choices such as holding. Instead of embedding
the original nonlinear OpenAP fuel calculation directly in every constraint:

1. Sample OpenAP at combinations of speed, altitude, and vertical speed.
2. Divide each small 3D grid cell into tetrahedra (pieces with four corners).
3. Let a flying interval select one piece.
4. Blend that piece's four sampled fuel values with nonnegative weights.
5. Multiply the resulting fuel flow in kg/s by the interval's 400 seconds.

Only corners of the selected local piece may contribute; the model cannot blend
arbitrary distant fuel-table points to invent a cheaper operating state.

Transit uses interval speed, average altitude, and vertical speed. Holding uses
the separate altitude-only fuel approximation explained earlier.

[add_horizontal_speed](Functions/fuel_pwl_v1.py#L33) also avoids a nonlinear square
root by taking the maximum of 32 linear directional projections. The result
slightly overestimates true horizontal speed; `NORM_COSINE` corrects the minimum
speed and flight-path-angle constraints conservatively. These 32 directions are
measuring directions, not the only allowed flight headings.

The main model sets `NonConvex = 0`. The intent is a linear mixed-integer model
with Gurobi general constraints and piecewise-linear approximations, not a
nonlinear trajectory solve.

## 10. Why solve OFF first, and how ON differs

Source: [main script, section 2.6](main_v18.1.py#L497).

```python
    for holding_enabled in (False, True):
        case_name = "holding_on" if holding_enabled else "holding_off"
        case_dir = output_dir / case_name
        case_dir.mkdir()
        for budget in hold_budgets:
            budget.RHS = N_steps - 1 if holding_enabled else 0
        initialization_seconds = 0.0
        if not holding_enabled:
            initialization_seconds = seed_geometric_start(m, min(30.0, TIME_LIMIT), case_dir / "initialization.log")
```

Each budget is an upper bound on the number of hold intervals for one aircraft.
Zero prohibits holding. Raising it allows holding without requiring any.
Other constraints, including arrival by the horizon, still apply.

The starting-guess sequence is:

| Step | What happens |
|---|---|
| Straight-route hints | Populate `.Start` values toward the nearest STAR; these are guesses, not fixed routes. |
| Geometric initializer | [seed_geometric_start](Functions/free_holding.py#L110) copies the model and removes named transit-fuel constraints from the copy to find a route guess. The full model retains its fuel constraints. |
| OFF solve | Optimize the full fuel-aware model with holding forbidden. |
| ON solve | Raise the holding budgets and use the OFF solution as the next guess. |

The OFF solution is passed on using:

```python
        if not holding_enabled:
            for variable in m.getVars():
                variable.Start = variable.X
```

The same model object is reused. A starting guess helps the search but does not
freeze its positions or STAR choices.

If the initializer cannot find a feasible starting solution, it raises an error.
If a full solve has no feasible solution, output handling records its status and
raises an error rather than presenting a successful comparison.

## 11. What gets saved, and how to read it

Source: [write_case_output](Functions/holding_output.py#L105),
[report_free_holding](Functions/free_holding.py#L158), and
[write_comparison_output](Functions/holding_output.py#L384).

By default, a new timestamped run directory is created under
[Output/v18_holding_comparison](Output/v18_holding_comparison).
An explicit `--output-dir` must also name a new directory: `exist_ok=False`
prevents overwriting an existing one.

The names in this table are relative to that newly generated run:

| Output | Purpose |
|---|---|
| `scenario.json` | Shared scenario inputs. Nearest STARs here are references, not necessarily the final chosen destinations. |
| `holding_off/` and `holding_on/` | Separate results for the two cases |
| Each case's `scenario.json` | Includes the actual selected STARs when a solution exists |
| Each case's `solver.log` and `run_summary.json` | Solver status, time, gap, bounds, objective, and aggregate physical results |
| Each case's `aircraft_summary.csv`, `trajectories.csv` | Per-flight totals and waypoint schedules |
| Each case's `separation_checks.csv` | Checks of the solved interval separation |
| Per-aircraft `holding_intervals.csv`, `hold_tracks.csv` | Selected holding times and reconstructed racetrack samples |
| Per-aircraft `weather_checks.csv` | Clearance checks against weather active during each interval |
| Per-aircraft fuel/emissions tables | OpenAP evaluation of the same solved schedule, separated into transit and holding |
| `comparison.csv`, `comparison.json`, `aircraft_comparison.csv` | Fleet and aircraft OFF/ON comparisons |
| `comparison.png`, `comparison.gif` | Comparison plot and animation |

The reporting path **does not run a second NLP to replace the route**. It
re-evaluates the solved schedule with OpenAP and reconstructs holding circuits.
Thus `pwl_fuel_kg` is the solver approximation, while `total_fuel_kg` is the
reported OpenAP evaluation. Neither is measured real-world fuel burn.

Weather and separation checks can raise errors. Merely seeing a run folder does
not mean both solves and all output checks finished successfully.

### Interpretation rules

- **ON does not mean a hold occurred.** Check `hold_seconds` or `hold_intervals`.
- **TIME_LIMIT does not mean no solution.** A feasible best-found solution can
  still be exported. Check `solution_count`, `mip_gap`, and `objective_bound`.
- **A better ON solution with zero holds is not a demonstrated holding benefit.**
  The two searches can find different feasible routes before their limits.
- **Lower weighted objective is not necessarily lower fuel.** Holding discounts
  affect optimization scores, not physical reports.
- **This is a simplified experiment.** Weather consists of rectangular exclusion
  zones, mass is fixed, and holding turns are not bank-aware aerodynamic solutions.
  Do not interpret its constraints or reported fuel as operational certification.

## 12. What to read or change first

| To understand or change... | Start here |
|---|---|
| Flight count, solve time, holding factor, W3 options | [Argument parsing](main_v18.1.py#L41) |
| Which flights are considered | [Cluster and subset selection](main_v18.1.py#L125) |
| Weather geometry and shared-clock setup | [Scenario construction](main_v18.1.py#L145) |
| Whether a hold is allowed and how fuel is counted | [Holding variables](main_v18.1.py#L270), then [add_hold_interval](Functions/free_holding.py#L20) |
| How routes are ranked | [Objective](main_v18.1.py#L309) |
| STAR choice, arrival, and movement limits | [Main constraints](main_v18.1.py#L375) |
| Weather and separation protection | [free_holding.py](Functions/free_holding.py) |
| OFF/ON comparison mechanics | [Solve loop](main_v18.1.py#L497) |
| Physical results versus solver approximations | [Output helpers](Functions/holding_output.py) |

For a first reading, follow sections **4, 5, 6, 8, and 10** of this guide.
Those explain the decisions, holding, weather, objective, and paired solve.

### Example commands

Run from this deliverable directory with the project's dependencies and a usable
Gurobi license. These commands launch optimizations and create new output folders.

```powershell
python .\main_v18.1.py
python .\main_v18.1.py --flight-count 1 --weather-regions 1 --time-limit 60
python .\main_v18.1.py --flight-count 3 --weather-regions 3 --holding-cost-factor 0.5 --time-limit 1200
python .\main_v18.1.py --flight-count 3 --weather-regions 3 --w3-window 300 700 --w3-match-w1-size --holding-cost-factor 0.5 --time-limit 1200
```

### Tests that illustrate the implementation

| Question | Test to read |
|---|---|
| Are STAR choices still free and smoothness disabled? | [Preserved model settings](test_v181.py#L112) |
| Are exact entry offsets and weather windows used? | [Offsets and multiple regions](test_v181.py#L131) |
| Do W3 options leave the other inputs unchanged? | [W3 changes](test_v181.py#L151) |
| Does initialization leave the full model intact? | [Geometric start](test_v181.py#L223) |
| Does OFF reject a forced hold? | [Holding OFF behavior](test_v181.py#L239) |
| Does holding consume fuel and produce the expected files? | [Holding fuel and output](test_v181.py#L256) |
| Does weather reject a crossing transit segment? | [Weather crossing](test_v181.py#L277) |
| Are existing output directories protected? | [Output protection](test_v181.py#L293) |

In short, v18.1 optimizes transit routes and optional fixed-duration holding
circuits together, then compares holding OFF and ON under the same weather.
