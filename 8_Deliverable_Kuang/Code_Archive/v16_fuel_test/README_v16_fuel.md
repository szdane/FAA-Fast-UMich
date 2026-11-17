# V16 OpenAP Fuel Objective

[main_v16_fuel_test.py](main_v16_fuel_test.py) adds a fuel penalty to the waypoint optimizer. [main_v15.py](main_v15.py) is unchanged. The reference is OpenAP `FuelFlow.enroute`, as used by [Functions/fuel_emission_analysis_computation.py](Functions/fuel_emission_analysis_computation.py#L239), not the custom debug fuel model.

## Run

Run these commands from the repository root using the working project Python environment and a valid Gurobi license. The verified environment used Python 3.11, Gurobi 12.0.3, OpenAP 2.2, NumPy 2.4.6, and SciPy 1.17.1. pandas and Matplotlib are also required.

```powershell
python 8_Deliverable_Kuang/main_v16_fuel_test.py
```

The default selects the existing six-flight subset, uses `--fuel-weight 25`, a 120-second solver limit, and a 5% relative MIP gap target. It displays a PWL-versus-OpenAP fuel plot. Close that plot to let the script finish writing its run summary. Use `--no-plots` for unattended runs.

```powershell
python 8_Deliverable_Kuang/main_v16_fuel_test.py --flight-limit 1 --no-plots
python 8_Deliverable_Kuang/main_v16_fuel_test.py --fuel-weight 0 --no-plots
```

`--output-dir` overrides the default `8_Deliverable_Kuang/Output/v16_fuel_test/weight_<weight>` directory. V16 never deletes the shared Output directory. Reusing an output directory replaces its result files; use distinct directories for comparisons.

## Formulation

The objective retains the time, distance, displacement, and heading-change penalties and adds

$$
J = J_{\mathrm{existing}} + w_{\mathrm{fuel}}\sum_{i,k} \Delta t\,\widehat{F}(\widehat{v}_{i,k},\bar{h}_{i,k},\dot{h}_{i,k}).
$$

`w_fuel` is objective units per kilogram, not a calibrated fuel price. A zero weight is the common-formulation control; it still uses the same state bounds and interpolation constraints.

[Functions/fuel_pwl.py](Functions/fuel_pwl.py) samples OpenAP at grid vertices and divides each three-dimensional cell into six tetrahedra. Each active interval selects one local tetrahedron and interpolates its four vertices. It cannot mix distant vertices to obtain a cheaper global convex-hull value. Inactive intervals have zero interpolation weights and zero fuel.

| Quantity | Grid / Convention |
| --- | --- |
| Aircraft | B737 for every flight, including flights whose real aircraft differ |
| Mass | Constant `0.8 * MTOW`, 56,000 kg with the tested OpenAP database |
| Horizontal speed | 120 through 560 knots in 40-knot increments, plus 300 m/s (583.15 knots) |
| Altitude | 0 ft, then 8,000 through 40,000 ft in 4,000-ft increments |
| Vertical speed | -1,000, -500, 0, 500, 1,000 ft/min |
| Fuel flow | OpenAP kg/s multiplied by interval duration in seconds |
| Interval state | Horizontal displacement / duration, mean endpoint altitude, altitude change / duration |

Solver variables use metres and seconds; the OpenAP call converts speed to knots, altitude to feet, and vertical speed to feet/minute. Mass does not decrease. There is no wind or acceleration input, and OpenAP's default limits remain enabled. No extrapolation outside the table is allowed.

Fuel and flight time include the interval reaching the STAR fix. They exclude intervals before entry and strictly after arrival. The report evaluates OpenAP directly on the same solved waypoints and durations, without calling the downstream NLP.

### Solver-Friendly Distance

The initial squared-distance equality created a nonconvex MIQCP that did not run reliably on six flights. V16 instead uses an exact maximum of 32 linear projections as an upper approximation to horizontal speed:

$$
\|v\|_2 \leq \widehat{v} \leq \|v\|_2/\cos(\pi/32).
$$

Its geometric overestimate is at most 0.4839%. Minimum-speed and flight-path-angle bounds are conservatively adjusted for this factor. The model caps horizontal speed magnitude at 300 m/s, rather than only capping each horizontal component. Together with the local fuel table, this makes the optimization a MILP. Straight-route and local-interpolation starting values help Gurobi find an incumbent but do not fix the solution.

The PWL objective uses the approximate speed; the OpenAP report uses the true Euclidean segment speed. Thus the reported fuel error includes both approximations. An upper speed approximation does **not** guarantee an upper fuel estimate, because fuel flow is not monotonic throughout the table.

V16 also corrects early-arrival position freezing and connects the previously unconstrained separation-bypass flags to the previous arrival state. Separation therefore remains enabled at the arrival waypoint, including the final grid node.

## Verified Comparison

These runs used the same six flights, corrected constraints, 400-second intervals, 120-second solver limit, and requested 0.5% MIP gap:

```powershell
python 8_Deliverable_Kuang/main_v16_fuel_test.py --fuel-weight 0 --mip-gap 0.005 --time-limit 120 --no-plots --output-dir 8_Deliverable_Kuang/Output/v16_fuel_test/comparison_weight_0
python 8_Deliverable_Kuang/main_v16_fuel_test.py --fuel-weight 25 --mip-gap 0.005 --time-limit 120 --no-plots --output-dir 8_Deliverable_Kuang/Output/v16_fuel_test/comparison_weight_25
```

| Result | Fuel Weight 0 | Fuel Weight 25 |
| --- | ---: | ---: |
| Direct same-schedule OpenAP fuel, kg | 2,225.85 | 1,686.82 |
| PWL fuel, kg | 2,256.11 | 1,701.01 |
| Total PWL error | +1.36% | +0.84% |
| Largest absolute segment fuel error | 3.07% | 3.93% |
| Summed flight duration, seconds | 9,200 | 8,800 |
| True horizontal distance, km | 1,217.58 | 1,226.62 |
| Achieved MIP gap | 0.67% | 3.16% |
| Solver status | TIME_LIMIT, feasible | TIME_LIMIT, feasible |

The fuel-weighted incumbent used **24.2% less OpenAP fuel** in this comparison. Neither run proved the requested 0.5% gap, so this is not a globally optimal savings claim. Zero-weight solutions have no fuel tie-break, and changing runtime, weights, or aircraft assumptions can change the comparison. Shorter modeled duration is not evidence of reduced gate-arrival delay.

On 500 deterministic held-out points across the full fuel table, the median absolute interpolation error was 0.49%, the 95th percentile 4.67%, and the maximum 16.32%. These are empirical errors, not certified error bounds.

Saved artifacts:

- [Comparison table](Output/v16_fuel_test/fuel_weight_comparison.csv) and [comparison plot](Output/v16_fuel_test/fuel_weight_comparison.png).
- [Fuel-weighted waypoints](Output/v16_fuel_test/comparison_weight_25/df_wide.csv), [interval fuel](Output/v16_fuel_test/comparison_weight_25/fuel_segments.csv), and [per-flight summary](Output/v16_fuel_test/comparison_weight_25/fuel_summary.csv).
- [Run summary](Output/v16_fuel_test/comparison_weight_25/run_summary.json), including achieved gap and weighted objective components.
- [Held-out samples](Output/v16_fuel_test/comparison_weight_25/pwl_validation.csv), [solver log](Output/v16_fuel_test/comparison_weight_25/gurobi.log), and [same-schedule approximation plot](Output/v16_fuel_test/comparison_weight_25/fuel_comparison.png).

Each run also writes a Gurobi MIP-start artifact. With no feasible incumbent the script fails explicitly instead of accessing nonexistent solution values; an infeasible model also writes an IIS file.

## Tests

[test_fuel_pwl.py](test_fuel_pwl.py) covers grid vertices, held-out errors, no extrapolation, active/inactive fuel, warm-start interpolation, the linear speed norm, arrival-interval accounting, true-speed reference evaluation, and optional NLP output forwarding.

```powershell
python -m unittest discover -s 8_Deliverable_Kuang -p test_fuel_pwl.py -v
```

The full six-flight regression is opt-in because it runs Gurobi for up to 30 seconds. It checks fuel totals, speed bounds, and the configured separation thresholds through arrival:

```powershell
$env:RUN_V16_INTEGRATION = "1"
python -m unittest discover -s 8_Deliverable_Kuang -p test_fuel_pwl.py -v
Remove-Item Env:RUN_V16_INTEGRATION
```

All **nine tests passed** with integration enabled. The saved comparison runs also passed checks for zero inactive fuel, the norm error bound, vertical speed, flight-path angle, and arrival-inclusive waypoint separation. Both plots were rendered and inspected.

## Limitations And Optional NLP

- The 400-second grid is coarse. No timestep-convergence study or continuous aircraft dynamics validation was performed. Existing floor-to-grid entry timing and clock labels are retained; absolute time-zone alignment was not revalidated here.
- Separation uses the existing toy thresholds: 500 ft horizontally along either projected axis, or 100 ft vertically. These are **not operational FAA separation minima**, and checks only apply at grid nodes, not continuously between them.
- A table state need not satisfy a complete aircraft performance envelope. B737-for-all, fixed mass, no wind, and average interval states limit the physical interpretation of fuel savings.
- `--run-nlp` runs the legacy analysis separately and writes into `retimed_nlp` below the run directory. That NLP changes waypoint timing; its totals must not be presented as fuel on the original v16 schedule. Output forwarding was regression-tested, but the full optional NLP optimization was not rerun. The legacy import emits a non-fatal local PROJ database warning in the tested environment.
- V16 minimizes fuel consumption, not emissions directly. A time-limited feasible solution is saved with its achieved gap; `OPTIMAL` means within the configured Gurobi tolerance, not necessarily a zero-gap proof.