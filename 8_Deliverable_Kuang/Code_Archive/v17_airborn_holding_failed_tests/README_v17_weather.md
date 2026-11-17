# Mandatory Reroute, Optional Holding

`main_v17_weather_test.py` is a separate copy of the holding experiment. Existing scripts and Functions modules are untouched.

## Scenario

- Flight: DAL1120_KMSNtoKDTW, entry to the RKCTY STAR fix (not landing).
- Synthetic bad weather: a 40 x 40 km rectangular no-fly region centered on the direct route, persistent for the entire horizon at all altitudes, with a 1 km clearance margin.
- The destination STAR and horizon are the same with and without weather. Waiting cannot clear this obstacle and changing STAR cannot avoid the test.
- Up to one 400-second entry circuit remains available by default, but holding is not forced. `--force-hold-steps` must be zero in this scenario.
- For each MILP segment, both endpoints must lie on the same exterior side of the expanded box. This conservative linear disjunction excludes crossings between waypoints as well as waypoints inside the box. Shapely independently verifies the solved polyline and any reconstructed hold.

## Run

```powershell
python 8_Deliverable_Kuang/main_v17_weather_test.py --time-limit 20
python 8_Deliverable_Kuang/main_v17_weather_test.py --clear-weather --time-limit 20
python -m unittest discover -s 8_Deliverable_Kuang -p test_v17_weather.py -v
```

Use the project Python environment with its Gurobi license. Results go to timestamped folders under `Output/v17_weather_test`; `--output-dir PATH` selects a specific folder. No shared outputs are deleted. Each run writes `weather_route.png`, `weather_scenario.json`, the waypoint CSV, solver statistics, and same-schedule OpenAP fuel/emissions. The test executes both cases with a 20-second solver cap each.

## Observed Results

| Case | Route Length | Holding | Gurobi Runtime | Gap |
| --- | ---: | ---: | ---: | ---: |
| Clear weather | 192.13 km | 0 s | 2.75 s | 0.79% |
| Bad weather | 197.82 km | 0 s | 20.02 s | 1.32% |

The direct entry-to-RKCTY line is 190.11 km. The clear-weather optimized route crosses the hypothetical weather footprint; the bad-weather optimized route avoids it. The optimizer chose no holding in both cases. These are feasible incumbent comparisons, not exact global-optimality claims; the weather run stopped at its time limit. A weather-aware no-hold detour MIP start helps it find an incumbent quickly but does not fix the optimized path.

## Scope

This is a single-flight, static synthetic-weather MILP test. It retains the existing toy aircraft/separation assumptions and no-bank OpenAP fuel estimate. It does not use observed weather or certify an operational trajectory. The rectangle condition is sufficient but conservative: some nonintersecting diagonal segments are excluded too.

`--run-nlp` is explicitly rejected here because the legacy smoother does not enforce this weather box and might shortcut the detour. Fuel/emissions are evaluated directly on the MILP schedule, not claimed as NLP results. Making the NLP honor the weather region is a separate follow-up.