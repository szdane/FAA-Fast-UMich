# Optional Interior Holding With Temporary Weather

`main_v17_interior_holding_test.py` is an isolated copy of the entry-holding experiment. Existing Python files remain unchanged. The default aircraft is DAL1120, modeled from entry to the RKCTY STAR, not to landing.

## Scenario

- A designated holding fix is 35% along the direct horizontal route, approximately 66.5 km from entry, at 21,000 ft.
- Four deliberately synthetic weather rectangles surround a clear pocket. They are active only during grid seconds 400-800, at all altitudes. The flight approaches before activation and can leave after clearance.
- The model may choose one 400-second hold at 210 knots, or avoid holding. No constraint forces a hold. A hold and a detour are supplied as initial guesses in their respective runs, not prescribed solutions.
- Hold entry and release share the fixed anchor; separate fuel and distance costs represent the circuit despite its zero net displacement. Both transitions bypass displacement-derived turn/speed proxies.
- A 1 km weather margin is enforced for whole straight intervals using common exterior halfspaces. The reconstructed racetrack is independently checked against the active cells.
- OpenAP uses a B737 at fixed 0.8 MTOW, no wind and no bank-angle fuel correction. Pre-hold route, hold, post-hold route and arrival intervals are counted once each.

The rectangular pocket is intentionally designed to demonstrate the choice, not obtained from observed weather or an operational holding procedure. Abrupt activation/clearance and coarse 400-second steps are simplifying assumptions. The hold altitude and location are designated inputs, not freely optimized.

## Run From Repository Root

```powershell
python 8_Deliverable_Kuang/main_v17_interior_holding_test.py --time-limit 30
python 8_Deliverable_Kuang/main_v17_interior_holding_test.py --time-limit 30 --clear-weather
python 8_Deliverable_Kuang/main_v17_interior_holding_test.py --time-limit 30 --max-hold-steps 0
```

Requires the same Gurobi/OpenAP environment as the existing holding experiment. Default output is a fresh timestamped directory under `Output/v17_interior_holding_test`. Use `--output-dir` to choose a directory explicitly.

## Verified Saved Results

| Case | Holding | Gurobi time | Gap | Direct OpenAP total fuel |
| --- | ---: | ---: | ---: | ---: |
| Clear weather | 0 s | 14.6 s | 0.68% | 279.70 kg |
| Temporary weather, optional hold | 400 s | 22.9 s | 0.99% | 492.75 kg |
| Temporary weather, holding disabled | 0 s | 30.0 s | 12.44% | 647.46 kg |

Saved examples: `Output/v17_interior_holding_test/{clear,weather,no_hold}`. The weather hold burns 150.42 kg; its non-hold route burns 342.33 kg. The disabled-holding case is a feasible time-limited incumbent, not an exact optimum. Comparisons use the same STAR, horizon and objective weights; fuel figures are direct OpenAP evaluations, not the MILP's PWL objective values.

`interior_holding_2d.png` shows the actual reconstructed racetrack at the interior anchor. Blue is the approach before the cells form; teal is the route after they clear. Crossing a displayed footprint outside its active interval is permitted. `df_wide.csv` retains macro hold flags, `interior_hold_track.csv` contains the closed circuit, `fuel_emissions_segments.csv` contains interval accounting, and `weather_checks.csv` records time-dependent clearance.

This experiment verifies the MILP and direct same-schedule fuel/emissions only. `--run-nlp` is rejected: the existing entry-only NLP handoff cannot preserve an interior hold or its weather timing. The plot is not an NLP trajectory or a bank-aware/continuous-time flight-safety certificate.

## Tests

```powershell
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_interior_holding.py -v
$env:RUN_INTERIOR_INTEGRATION = '1'
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_interior_holding.py -v
```

Three fast checks cover a closed level interior circuit, pre/post-hold fuel accounting, and collision rejection. The opt-in integration check runs all three scenarios, with each MILP capped at 30 seconds, and independently checks optional holding, active-weather clearance, linearity and fuel totals.