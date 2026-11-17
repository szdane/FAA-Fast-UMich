# V18: paired airborne holding experiments

[main_v18.py](main_v18.py) runs **holding OFF, then holding ON** against the
same input scenario. Earlier versions and shared helpers are unchanged.

## Run

Use the existing Python environment containing Gurobi (with a license), OpenAP,
NumPy, pandas, matplotlib, Shapely and the dependencies of the shared helpers.
From the repository root:

```powershell
# One aircraft, one storm
python 8_Deliverable_Kuang\main_v18.py

# Three aircraft, one shared storm
python 8_Deliverable_Kuang\main_v18.py --flight-count 3

# Three aircraft, three shared storms
python 8_Deliverable_Kuang\main_v18.py --flight-count 3 --weather-regions 3
```

Each command runs both cases. `--time-limit 120` increases the full MILP solve
budget to 120 seconds **per case** (default 60). Each case also has a geometric
initialization solve, capped at the smaller of 30 seconds and the requested
time limit. Construction, reporting and plotting take additional time.

Outputs default to a new timestamped directory under
`Output/v18_holding_comparison`. `--output-dir PATH` selects a **new** directory;
an existing directory is never overwritten. No solve occurs when importing
the module.

## Identical conditions

- Single flight: DAL1120. Three flights: DAL1120, DAL1655 and EDV5018.
- Time zero is DAL1120's actual entry, `2023-12-22 08:10:01 UTC`, not a
  rounded grid boundary. The other aircraft retain their recorded offsets of
  485 and 166 seconds, respectively.
- All storm regions exist at fixed locations during shared seconds **400-800**
  after the first entry. They do not restart when another aircraft enters or
  when a hold begins.
- Region W1 covers 25%-85% of the first flight's direct entry-to-nearest-STAR
  distance, with a 70 km half-width. In the three-region case, W2 and W3 are
  50 km by 40 km rectangles centered halfway along the other two approaches.
  All rectangles use the first route's orientation. Exact projected coordinates,
  basis and entry/exit data are saved in `scenario.json`.
- Aircraft, entry states/times, nearest STAR destinations, altitude bounds,
  six 400-second intervals per aircraft, fuel model, objective weights, weather,
  clearance and separation rules are identical within a pair.
- The **only model difference** is each aircraft's holding budget: zero for OFF,
  or up to the horizon's six intervals for ON. Transit and arrival feasibility
  normally reduce the number of possible holds.

The joint multi-aircraft model includes separation constraints; these are not
three unrelated independent solves.

## What optional airborne holding means

At every airborne interval the optimizer can select transit or a complete,
level holding circuit. Entry, interior, late, consecutive and separated holds
are supported. The horizontal anchor and altitude are decision variables, not
prescribed holding fixes. Holding before entry or at/after arrival is forbidden.
Holds are never forced in the comparison.

A hold returns to the same macro waypoint after 400 seconds, but **the aircraft
does not hover**. Its reconstructed racetrack has two straights and two
semicircles, flies at 210 kt, and incurs nonzero distance, fuel, emissions and
airborne time. Circuit orientation is fixed to the common weather basis.
The holding-fuel approximation uses the shared 100-ft altitude PWL curve.
Transit uses the existing local tetrahedral OpenAP PWL model; reporting also
reevaluates OpenAP on exactly the selected schedule, without NLP retiming.

The common weighted objective includes time, distance, fuel and heading/
smoothness terms; it is not a fuel-only objective. Minimum transit speed is
still enforced with holding OFF. An aircraft can slow down or take a detour
without selecting an explicit holding circuit.

## Weather and traffic checks

All storm columns are impassable at every altitude, with 1 km clearance.
Transit endpoints must share an exterior side of each expanded rectangle;
holding uses the whole racetrack envelope, not just its anchor.

Each aircraft has its own 400-second grid starting at its actual entry time.
An interval overlapping any part of the weather window is conservatively
checked against weather for the **whole interval**. For example, DAL1655's
485-885 second interval is weather-constrained even though weather clears at
800. OFF and ON use the same conservative rule.

For every pair of temporally overlapping airborne intervals, one common
separating side must hold at both overlap endpoints. Straight motion is
interpolated at these endpoints; holding expands the position by its complete
rotated racetrack bounding box. This protects the continuous straight segments
and the abstract holding envelopes, not just the original grid nodes.
Independent post-solve checks reject weather or separation violations.

The inherited demonstration separation values are only 500 ft horizontal or
100 ft vertical, **not operational ATC minima**. Holding envelopes and weather
half-space tests can reject otherwise feasible paths. This is a synthetic
optimization experiment, not flight guidance: no terrain, wind, bank-angle,
turn-entry continuity, aircraft-specific performance certification or ATC
authorization is modeled. All flights use a B737 at fixed 0.8 MTOW.

## Plots and comparison files

`comparison.png` shows OFF and ON side by side with identical axis scales:

- Shaded, numbered weather regions W1-W3, labeled active only at 400-800 seconds.
- Aircraft-specific route colors and timestamps on the shared clock.
- Solid transit legs during weather; dashed legs outside its time window.
- Thick reconstructed holding circuits with `HOLD start-end` annotations.
- An explicit selected-hold count, including zero when no circuit is chosen.

Crossing a shaded footprint **after the storm clears** is allowed.

Other outputs:

| File | Content |
| --- | --- |
| `scenario.json` | One shared, reproducible scenario for the pair |
| `comparison.csv`, `comparison.json` | Fleet totals, objective/bound/gap, solve status, ON-minus-OFF differences |
| `aircraft_comparison.csv` | Both cases' per-aircraft fuel, distance, time and hold counts |
| `{case}/aircraft_summary.csv` | Per-aircraft metrics for one case |
| `{case}/trajectories.csv` | Long-form schedules with aircraft IDs and exact timestamps |
| `{case}/separation_checks.csv` | Minimum common-side clearance margin over each interval overlap |
| `{case}/solver.log`, `initialization.log` | Full and geometric solver logs |
| `{case}/{aircraft}/df_wide.csv` | Macro waypoints and interval-end holding flags |
| `{case}/{aircraft}/holding_intervals.csv`, `hold_tracks.csv` | Selected holds and reconstructed circuits |
| `{case}/{aircraft}/free_holding_2d.png` | V17-style individual weather/holding visualization |
| `{case}/{aircraft}/fuel_emissions_*.csv`, `weather_checks.csv` | Interval accounting and independent weather checks |

`airborne_seconds` is the sum of all aircraft's flown time, including holding.
`last_arrival_seconds` is the final arrival on the shared scenario clock.
Distance includes the full holding circuit length, despite zero net waypoint
displacement.

## Solver fairness and interpretation

The geometric initializer solves the same kinematic/weather/separation problem
without the transit fuel PWL objective, then supplies a feasible route hint and
fuel-table interpolation start to the full MILP. It does not constrain the
final route or force its holding decisions.

ON also receives the solved OFF incumbent as a second feasible start. Thus ON
can retain the OFF route while searching the larger feasible set. These are
equal full-solve time limits, **not identical search histories**. Gurobi's
1% target gap is not guaranteed within the time limit; all actual gaps and
bounds are reported. No feasible incumbent produces an explicit error rather
than a fabricated comparison.

**Holding ON can choose zero holds.** If both runs choose zero holds, differences
between their finite-time solutions are solver-incumbent differences, not
demonstrated benefits of holding. The comparison explicitly reports this.
Even with holds selected, do not interpret a finite-gap comparison as proven
optimal savings.

## Verified examples

These initial examples use the original holding-cost multiplier of 1.0.

All six full solves completed with feasible incumbents, using 30 seconds per
case, and were checked from their saved outputs:

| Aircraft / regions | OFF fuel (kg) | ON fuel (kg) | ON holding (s) | OFF / ON gap |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 | 667.30 | 518.40 | 0 | 11.50% / 6.84% |
| 3 / 1 | 1712.25 | 1669.20 | 0 | 16.83% / 18.61% |
| 3 / 3 | 1730.15 | 1581.56 | 0 | 18.37% / 19.38% |

All reached the time limit. These particular runs **did not select holding**;
their fuel differences are not evidence of holding savings. Longer solves may
change the incumbents or holding choices, but are not guaranteed to do so.
Results are under `Output/v18_holding_comparison/verified_suite`:

- [Single-aircraft plot](Output/v18_holding_comparison/verified_suite/single/comparison.png)
- [Three-aircraft / one-region plot](Output/v18_holding_comparison/verified_suite/three_one/comparison.png)
- [Three-aircraft / three-region plot](Output/v18_holding_comparison/verified_suite/three_three/comparison.png)

## Holding-penalty sensitivity

There was no separate holding surcharge in the original objective: holding
was charged its weighted airborne time, flown circuit distance, and PWL fuel.
`--holding-cost-factor FACTOR` now multiplies **only those holding terms**.
The default 1.0 preserves the original model; 0.5 halves their objective cost;
0 makes those objective contributions free. Factors must be finite and
nonnegative.

Transit costs, weather, speeds, holding geometry, altitude constraints and
separation do not change. Actual fuel, emissions, time and distance are
**never discounted** in the physical reports. The original-weight objective,
the holding subtotal before discount, and the applied discount are also saved
in each case's `run_summary.json`.

```powershell
# One paired comparison with discounted holding.
python 8_Deliverable_Kuang\main_v18.py --holding-cost-factor 0.5 --time-limit 60

# Same one-aircraft/one-region scenario at multiple penalty levels.
python 8_Deliverable_Kuang\run_v18_holding_penalty_sweep.py --time-limit 60

# Custom values for refining an observed transition.
python 8_Deliverable_Kuang\run_v18_holding_penalty_sweep.py --factors 0.9 0.8 0.7 --time-limit 60
```

The sweep solves holding OFF once, then optional holding at each factor with
the same OFF incumbent as a warm start. It does not carry an earlier factor's
incumbent into the next solve, and it never requires a hold. Each optional run
also gets its usual geometric initializer. The only scenario field that
changes is `holding_cost_factor`.

Outputs are under `Output/v18_holding_penalty_sweep/<timestamp>` unless a new
`--output-dir` is supplied. `penalty_sweep.csv`/`.json` record the holding
decisions, interval times, physical fuel, objectives and bounds;
`penalty_sweep.png` plots the sensitivity. Each factor has the usual individual
weather/holding plot and interval reports.

An observed activation threshold is not automatically an exact optimization
threshold: finite-time solves may miss better routes. However, if a
holding-selected incumbent's discounted objective is below the **OFF lower
bound**, every zero-hold route is worse at that factor. The sweep records this
stronger test as `holding_beats_all_no_hold_solutions`. A false value means
this certificate was not obtained, not that holding is necessarily worse.

### Measured activation range

The saved `coarse` and `refined` sweeps used 60-second full MILP limits per
case. Both OFF controls obtained the same objective (36,700.07), lower bound
(35,310.50), and actual fuel (514.54 kg). The optional-holding results were:

| Holding factor | Discount | Selected holding (s) | Actual fuel (kg) | Achieved gap |
| --- | ---: | ---: | ---: | ---: |
| 1.00 | 0% | 0 | 502.42 | 3.60% |
| 0.90 | 10% | 0 | 501.28 | 3.36% |
| 0.85 | 15% | 0 | 502.15 | 3.70% |
| 0.80 | 20% | 400 | 488.05 | 3.98% |
| 0.75 | 25% | 400 | 489.92 | 1.31% |
| 0.50 | 50% | 400 | 490.68 | 1.74% |
| 0.25 | 75% | 1200 | 723.22 | 7.26% |
| 0.00 | 100% | 1200 | 739.09 | 1.00% |

**Observed activation lies between factors 0.80 and 0.85**: 0.80 selected
holding; 0.85 did not. This is a finite-time selection result, not an exact
globally optimal transition. The residual gaps are larger than the near-
threshold objective differences, so the exact transition remains unresolved.

At factors 0.80, 0.75 and 0.50, the selected circuit is at entry during
**0-400 seconds**. The planner anticipates the upcoming storm, flies outside
its footprint during 400-800, and crosses the footprint after 800.
The activation time and location remain choices, not imposed constraints.

At 0.50 the held objective is 33,544.22, below the OFF lower bound of
35,310.50. Thus a zero-hold solution cannot be globally optimal under that
discounted objective. The same certificate holds at 0.25 and 0.00.

Very cheap holding produces additional circuits and **more actual fuel**:
0.25 selected intervals 0-400, 400-800 and 1600-2000; 0.00 selected 0-400,
400-800 and 1200-1600. Their 723-739 kg totals exceed the OFF incumbent's
514.54 kg. Discounting the objective does not physically make holding cheaper.
Do not treat these artificial objective discounts as calibrated fuel savings.
The production default remains 1.0.

- [Combined sensitivity plot](Output/v18_holding_penalty_sweep/holding_penalty_results.png)
- [All eight results](Output/v18_holding_penalty_sweep/holding_penalty_results.csv)
- [Selected circuit at factor 0.75](Output/v18_holding_penalty_sweep/coarse/factor_01_0.75/DAL1120_KMSNtoKDTW/free_holding_2d.png)
- [Coarse sweep data](Output/v18_holding_penalty_sweep/coarse/penalty_sweep.json)
- [Refined sweep data](Output/v18_holding_penalty_sweep/refined/penalty_sweep.json)

## Tests

```powershell
python -m unittest discover -s 8_Deliverable_Kuang -p test_v18.py -v

# Rerun all six full solves and check their generated output.
$env:RUN_V18_INTEGRATION = '1'
python -m unittest discover -s 8_Deliverable_Kuang -p test_v18.py -v
Remove-Item Env:RUN_V18_INTEGRATION

# Validate the already saved examples without solving them again.
$env:V18_RESULTS_ROOT = (Resolve-Path '8_Deliverable_Kuang\Output\v18_holding_comparison\verified_suite').Path
python -m unittest discover -s 8_Deliverable_Kuang -p test_v18.py -v
Remove-Item Env:V18_RESULTS_ROOT

# Check physical fuel, objective discounts, circuits and weather in a saved sweep.
$env:V18_PENALTY_RESULTS_ROOT = (Resolve-Path '8_Deliverable_Kuang\Output\v18_holding_penalty_sweep\coarse').Path
python -m unittest discover -s 8_Deliverable_Kuang -p test_v18.py -v
Remove-Item Env:V18_PENALTY_RESULTS_ROOT
```

Checks cover identical paired-model coefficients except holding budgets, exact
entry/weather timing, early/late/consecutive/separated holding feasibility,
closed nonzero-length circuits and their plotting/reporting, continuous crossing
conflicts, fuel accounting, and all three saved paired scenarios. Hold flags are
fixed only in the geometry regression fixtures, never in normal runs.

The final combined run of v18 and existing free-holding tests passed 9 tests;
3 optional repeat-solve tests were skipped. The six paired solves had already
run separately and their saved results passed the integration checks. Pylance
reported no diagnostics in the new main script or tests.

After adding penalty sensitivity, the combined run passed 11 tests with
3 optional repeat-solve tests skipped; the refined sweep also passed its
saved-output check separately. Additional tests verify that only holding
objective coefficients change, reject invalid multipliers, independently
recompute undiscounted OpenAP fuel, and validate selected circuits and
weather clearance at every tested factor.
