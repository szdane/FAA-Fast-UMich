# Optional Holding At Free Locations And Grid Times

`main_v17_free_holding_test.py` is a separate copy of the fixed interior-holding experiment. It uses the new `Functions/free_holding.py`; earlier Python examples and helpers are unchanged.

## What Is Free

- Holding is optional at every airborne interval, including the first interval after entry. It is forbidden before entry and at/after arrival.
- The route optimizer chooses the horizontal holding anchor and its altitude. There is no designated holding fix or prescribed holding altitude.
- Holds can be consecutive or separated by transit intervals. `--max-hold-steps` is an upper bound, not a required count; the default is two and zero disables holding.
- The example remains a single flight to RKCTY, with six 400-second intervals. The finite horizon and travel requirements limit how much holding can fit.

"Anywhere" means a reachable location that satisfies the constraints and leaves clearance for the entire circuit, not unrestricted airspace. "Anytime" means an eligible interval on the time grid, not any continuous timestamp or arbitrary-duration turn.

## Formulation

For each interval ending at node k, binary `hold[i,k]` selects a complete circuit. When selected, indicator constraints require `position[k] == position[k-1]` in x, y and z. These are equalities between decision variables, not equalities to a fixed location. The aircraft remains airborne.

The circuit has nonzero speed, distance and fuel despite zero net displacement. Ordinary transit PWL fuel is disabled during holding; time still counts, and distance is `hold_speed * timestep`. Holding flow is evaluated at fixed 210 kt and zero vertical speed, with a one-dimensional altitude PWL curve sampled from OpenAP every 100 ft between 0 and 40,000 ft. Indicator constraints charge `duration * flow` when holding and zero otherwise, without a binary-times-continuous product. Reports also evaluate exact OpenAP at the selected altitude, separately from the MILP PWL estimate.

Displacement-based heading and speed-change proxies are bypassed into and out of a hold. A logical OR of current/previous hold flags handles consecutive holds. This does not enforce physical turn entry/exit continuity.

The no-bank geometric template retains two straight legs and two semicircles, one complete circuit per 400-second interval. Its orientation remains fixed along the entry-to-STAR bearing; orientation, airspeed and circuit duration are not optimized.

## Weather Clearance

The example retains four synthetic rectangles active only during grid seconds 400-800. Their central pocket affects feasible locations but is not a holding-fix constraint. The prior 35% location and 21,000-ft altitude appear only in the solver's initial route hint, never as holding-location constraints.

For transit, both endpoints must lie on one common exterior side of each clearance-expanded cell. For holding, the same disjunction includes the whole racetrack's bounding rectangle relative to its variable anchor. In route-aligned coordinates, its offsets are `[-radius, leg+radius]` and `[0, 2*radius]`, where `leg = speed * duration / 4` and `radius = leg / pi`. This is a conservative MILP condition: some geometrically clear circuits may be rejected, but checking only the anchor cannot accidentally admit a circuit through a weather cell.

Every exported straight interval and reconstructed circuit is independently checked against active weather. The template's bounding rectangle contains its continuous arcs. Weather activation remains coarse and abrupt; this is synthetic, not observed weather.

## Run From Repository Root

```powershell
python 8_Deliverable_Kuang/main_v17_free_holding_test.py --time-limit 30
python 8_Deliverable_Kuang/main_v17_free_holding_test.py --time-limit 30 --clear-weather
python 8_Deliverable_Kuang/main_v17_free_holding_test.py --time-limit 30 --max-hold-steps 0
python 8_Deliverable_Kuang/main_v17_free_holding_test.py --time-limit 30 --max-hold-steps 3
```

Use the existing Gurobi/OpenAP environment. Each run defaults to a new timestamped directory under `Output/v17_free_holding_test`; `--output-dir PATH` selects a directory explicitly. The solve limit excludes construction, reporting and plotting. Increasing the hold limit does not enlarge the six-interval horizon.

## Saved Results

| Case | Holding | Gurobi time | Gap | Direct OpenAP total fuel |
| --- | ---: | ---: | ---: | ---: |
| Temporary weather, free holding | 400 s | 30.06 s | 1.13% | 486.52 kg |
| Clear weather, free holding | 0 s | 26.16 s | 0.44% | 281.50 kg |
| Temporary weather, holding disabled | 0 s | 30.02 s | 17.22% | 698.07 kg |

The weather run chose projected anchor approximately `(-165263.10, 97847.17)` metres relative to DTW and altitude 23,775 ft, different from the former fixed anchor and 21,000-ft altitude. Its hold lasts seconds 400-800, costs 158.28 kg fuel, and its non-hold route costs 328.24 kg. The reconstructed circuit maintains 1 km clearance from active weather. These are feasible-solution comparisons, not exact-optimum or operational fuel-savings claims. Both weather runs reached their time limits; the disabled-holding control has a substantial remaining gap.

Saved examples are `Output/v17_free_holding_test/{weather,clear,no_hold}`. Outputs include:

- `df_wide.csv`: full schedule and interval-end hold flags.
- `holding_intervals.csv`: each selected hold's start/end time, coordinates, altitude, exact fuel and PWL fuel. Different locations and altitudes are reported separately.
- `holding_summary.csv`: per-flight counts and exact/PWL fuel totals.
- `hold_tracks.csv`: reconstructed circuits grouped by `end_step`; do not connect separate circuits into a single polyline.
- `free_holding_2d.png`: selected racetracks and time-coloured transit legs, with no artificial lines joining separate holds.
- `fuel_emissions_segments.csv`, `fuel_emissions_summary.csv`: each airborne interval counted once, including entry holds and arrival intervals.
- `weather_checks.csv`, `free_holding_scenario.json`, `run_summary.json`: clearance, scenario and solver metadata.

## Limits And Tests

All aircraft are still represented as a B737 at fixed 0.8 MTOW, without bank-angle or wind corrections. Free holding can increase solve difficulty. This demonstration is single-flight; it does not validate multi-aircraft continuous-time separation, obstacle/terrain clearance, stall limits, ATC permission or operational turn dynamics. `--run-nlp` remains rejected because the earlier entry-only handoff does not support free/interior holds and their weather timing.

```powershell
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_free_holding.py -v
$env:RUN_FREE_HOLDING_INTEGRATION = '1'
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_free_holding.py -v
Remove-Item Env:RUN_FREE_HOLDING_INTEGRATION
```

The tests check free anchors/altitudes, zero inactive fuel, closed level holds, altitude interpolation, rotated full-circuit weather exclusion, optional weather/clear controls, and full-model feasibility for entry, late, consecutive and separated holds. The last test fixes hold flags only inside its feasibility fixtures; normal scenario runs never force a hold. Three choice solves are capped at 30 seconds each and four fixture solves at 15 seconds each.

All six tests passed with integration enabled (114.855 seconds). The 100-ft holding-fuel curve had a maximum relative error of approximately 0.095% at 4,001 evenly spaced validation altitudes; this is a sampled check, not a formal continuous error bound. The original 1,000-ft spacing failed the 0.5% test threshold and was replaced before final verification.