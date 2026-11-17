# Saved V17 Trajectory Analysis

`run_v17_trajectory_test.py` runs trajectory generation and fuel/emissions analysis on an existing MILP result folder. It does not rerun Gurobi. The original holding/weather scripts and helpers remain unchanged; this runner uses new `fuel_emission_analysis_main_weather_nlp.py` and `fuel_emission_analysis_trajectory_weather_nlp.py` copies.

## Run

From the repository root with the project Python environment:

```powershell
python 8_Deliverable_Kuang/run_v17_trajectory_test.py
python 8_Deliverable_Kuang/run_v17_trajectory_test.py --input-dir 8_Deliverable_Kuang/Output/v17_holding_test/forced_nlp
```

The default input is `Output/v17_weather_test/bad_weather` under the deliverable folder. An input folder needs `df_wide.csv` and `holding_summary.csv`; `weather_scenario.json` enables weather constraints. Default output is a new timestamped subfolder of the input. `--output-dir PATH` selects a specific folder; `--nlp-time-limit 30` limits IPOPT CPU seconds per flight. Existing B737, fixed-mass fuel reporting and no-bank holding assumptions are retained.

## Output And Plot

- `trajectories_2d.png` overlays the MILP waypoints and generated NLP route. Selected holds have an orange racetrack symbol and duration at their entry anchor. The symbol is schematic, not drawn to physical scale. No symbol is drawn when no holding was selected.
- `<acId>_route.csv` contains the generated non-hold trajectory, timestamps, speed, altitude, fuel, and emissions. `<acId>_hold_track.csv` separately contains the geometric hold reconstruction.
- `combined_fuel_emissions_summary.csv` separates route, hold, and combined totals. Fuel is kg; emissions ending `_g` are grams. `fuel_emissions_summary.csv` remains the separate same-schedule MILP estimate.
- `nlp_status.csv` records solver status and weather clearance. Its inherited `nlp_range_km` is endpoint distance, not flown path length; measure the route CSV polyline for flown distance.

Holds are excluded from the NLP but included once in combined fuel/emissions. The route starts at the hold release time and retains the MILP's total non-hold duration. A failed NLP is explicitly labeled as a same-schedule fallback, and the runner exits unsuccessfully rather than claiming trajectory-generation success.

For weather inputs, the NLP excludes a conservative circle containing the clearance-expanded weather-box corners in its local projection, plus 250 m padding. This is enforced at mesh and collocation points. The exported polyline is independently checked against the actual box and requested clearance before being accepted. This is not a continuous-time aircraft safety proof or an exact rectangle-optimal NLP.

## Observed Runs

Both cases are DAL1120_KMSNtoKDTW to RKCTY. Both returned `Solve_Succeeded` within the configured 30-second per-flight NLP CPU cap.

| Case | Non-Hold Time | Hold Time | NLP Route Length | Route Fuel | Hold Fuel | Combined Fuel | CO2 | NOx |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bad weather | 1,600 s | 0 s | 253.04 km | 391.38 kg | 0 kg | 391.38 kg | 1,236.75 kg | 2.219 kg |
| Earlier forced-hold example, no weather | 1,200 s | 400 s | 190.11 km | 244.66 kg | 169.86 kg | 414.52 kg | 1,309.88 kg | 2.696 kg |

Saved results are under `Output/v17_weather_test/bad_weather/nlp_verified` and `Output/v17_holding_test/forced_nlp/trajectory_verified`.

The weather polyline has 8.77 km minimum clearance from the box. It also contains broad bends: the 1,600-second schedule and inherited Mach 0.5 minimum, together with loose waypoint tolerances, allow substantial extra path length. Observed TAS was 298-318 kt. The MILP route was 197.82 km and its direct OpenAP estimate was 309.66 kg, so those results must not be confused with the NLP's 253.04 km / 391.38 kg result. No causal ablation or operational validation was performed.

Verified from both saved trajectories: successful solver status, strictly increasing timestamps, zero first-row integration duration, total route time, exact addition of hold and route fuel, and weather clearance where applicable. Both 2D plots were rendered and inspected. These two different scenarios are not a hold-versus-reroute fuel-savings comparison.