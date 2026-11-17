# V17 Entry-Holding Experiment

`main_v17_holding_test.py` extends the supplied MILP with optional **entry-only airborne holding**. The supplied script is preserved as `main_v17_holding_test_original.py`. The original Functions modules are unchanged; analysis uses copies named `fuel_emission_analysis_main_holding.py` and `fuel_emission_analysis_trajectory_holding.py`.

## Run

From the repository root, using the project environment with Gurobi, OpenAP, and CasADi installed:

```powershell
python 8_Deliverable_Kuang/main_v17_holding_test.py
python 8_Deliverable_Kuang/main_v17_holding_test.py --flight-limit 1 --force-hold-steps 1 --time-limit 15 --run-nlp
python 8_Deliverable_Kuang/main_v17_holding_test.py --force-hold-steps 1 --time-limit 20
python 8_Deliverable_Kuang/main_v17_holding_test.py --max-hold-steps 0 --time-limit 20
```

- Default: six flights, at most one circuit each, 60-second Gurobi limit, 1% target gap, no NLP or plots. Holding is optional and need not be selected without a scheduling benefit.
- `--force-hold-steps N` requires at least N circuits on the first selected flight. Use `--max-hold-steps N` to permit more than one.
- `--run-nlp` runs the copied non-hold trajectory analysis. `--nlp-time-limit 30` caps IPOPT CPU seconds **per aircraft**; neither limit includes model construction, I/O, or plotting.
- `--plots` enables the legacy MILP animation, which displays hold anchors, not the reconstructed circuits.
- Outputs default to a fresh timestamped directory under `Output/v17_holding_test`. `--output-dir PATH` overrides this; reusing it can overwrite that run's files. The shared Output directory is never deleted.

## Formulation

Each binary `hold[i,k]` represents the interval **ending** at node k. Holds may occupy only a contiguous prefix after entry. The maximum circuit count is bounded, and the horizon gains that many intervals.

For a holding interval:

- Its macro waypoint remains at the entry position and altitude, representing a complete circuit returning to that point, not a stationary airplane.
- Arrival remains false; separation is not bypassed.
- The ordinary route PWL fuel model is inactive. Holding fuel instead equals a precomputed constant times the binary.
- Flight time still costs 400 seconds. Distance cost includes the flown circuit length, not zero net displacement.
- The displacement-based heading and speed-change proxies are relaxed at hold release because zero macro displacement is not zero airspeed.

Fuel per circuit is `400 * OpenAP.FuelFlow.enroute(mass, tas=210, alt=entry_altitude_ft, vs=0)`. All aircraft use B737, constant `0.8 * MTOW`, no wind, no bank-angle correction, and no acceleration correction. TAS is not an operational holding IAS limit. No new fuel-table dimension or nonlinear constraint is introduced.

The geometric reconstruction has two 100-second straight legs and two 100-second semicircular turns at 210 kt TAS. It is exported separately and never sent through the NLP. It is a kinematic template, not a dynamically validated turn model.

## Fuel And NLP Outputs

- `df_wide.csv`: original schedule/status columns plus `fN_hold`, marking interval ends. Hold rows remain airborne (`status=1`).
- `holding_summary.csv`: circuits, holding seconds, fixed holding conditions, hold fuel, and route PWL fuel.
- `fuel_emissions_summary.csv`: direct OpenAP route, hold, and total fuel/emissions on the MILP schedule. Fuel is kg; emissions columns ending `_g` are grams.
- `fuel_emissions_segments.csv`: route intervals and aggregate hold operating points. Cumulative fuel/emissions reset for each phase; use the summary for combined totals.
- `<acId>_hold_track.csv`: reconstructed holding coordinates in projected metres and absolute grid seconds; empty when no hold is selected.
- `run_summary.json` and `simulation_summary.md`: solver status, runtime, gap, objective, assumptions, and solver log. Time-limited incumbents are saved.

With `--run-nlp`, the copied wrapper removes hold duplicates but keeps the last hold point as the route's release anchor. It fixes the NLP's **total non-hold duration** to the remaining MILP duration, fixes endpoint altitudes, and offsets route UTC timestamps by the hold duration. Integration starts with a zero-duration row, so no boundary interval is counted twice.

The `nlp` subdirectory contains `<acId>_route.csv`, `nlp_status.csv`, and `combined_fuel_emissions_summary.csv`. The combined report adds hold fuel/emissions to the NLP route only when the solver succeeds. A failed NLP retains the separately labeled `same_schedule_openap` estimate, not a claimed NLP trajectory. The same-schedule report remains separate from the NLP report.

## Limits

- This prototype schedules entry holds only; no interior holding-fix selection, holding stacks, or exact entry/exit heading and speed continuity.
- Holding separation uses a conservative enclosing radius of half the circuit path length around the entry anchor at hold-start grid nodes. Transit-transit separation remains the existing node-based formulation. Neither ensures continuous-time separation, weather clearance, or obstacle clearance; the thresholds remain toy 500-ft horizontal/100-ft vertical values, not operational minima.
- The NLP preserves the release time and total route duration, but not every intermediate waypoint time or cross-flight separation. Its existing 10 km waypoint tolerance and performance restrictions remain. A MILP route may still fail the NLP.
- Holding speed, duration, and altitude are fixed assumptions. No aircraft-specific stall/performance certification or bank-fuel calibration was added. The NLP uses its legacy changing-mass dynamics; fuel/emissions reporting retains the existing constant-mass assumption.
- Increasing `--max-hold-steps` also enlarges the route horizon and existing PWL model. Comparing against `--max-hold-steps 0` is a smoke control, not an equal-horizon fuel-savings experiment.

## Verification

```powershell
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_holding.py -v
$env:RUN_HOLDING_INTEGRATION = "1"
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_holding.py -v
Remove-Item Env:RUN_HOLDING_INTEGRATION
```

Focused tests cover zero/one/two holds, retained release anchors, no pre-entry/post-arrival or duplicate-boundary fuel, all pollutant totals, and racetrack closure/length. The opt-in six-flight test checks a forced hold under a 20-second limit, speed/vertical bounds, grid-node separation including the hold envelope, and zero quadratic constraints.

Observed runs in the project environment:

| Case | Gurobi Runtime | Gap | Result |
| --- | ---: | ---: | --- |
| One flight, holding disabled | 1.89 s | 0.49% | Zero hold fuel |
| One flight, one forced circuit | 5.61 s | 0.98% | 400 s holding; 169.86 kg hold fuel |
| Six flights, first flight forced to hold | 20.04 s | 15.37% | Feasible time-limited incumbent; not near-optimality proof |

The forced single-flight NLP reported `Solve_Succeeded`: 1,200 s non-hold route, 244.66 kg route fuel, and 414.52 kg combined route-plus-hold fuel. This verifies the split calculation, not an operational fuel-savings claim. The local PROJ database warning was non-fatal.