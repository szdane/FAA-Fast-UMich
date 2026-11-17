# v16.2 objective ablation experiment

## Measured results

The [completed experiment report](Experiments/v162_objective_ablation/REPORT_20261104.md)
contains eight matched runs (two seeds), an additional controlled warm-start
diagnostic, gap histories, and failure analysis. Removing only heading reached
1% within 900 seconds in both seeds. Removing only distance timed out in both;
removing both succeeded once and timed out once. All runs retained feasible
incumbents. These are instance-specific observations, not general guarantees.

## Question and controlled factors

Does the MILP in [main_v16.2_test.py](main_v16.2_test.py) still reach its 1%
optimality-gap target when the distance penalty, heading-rate penalty, or both
are removed?

| Case | Horizontal distance weight | Heading-rate weight |
|---|---:|---:|
| `baseline` | 0.1 | 0.002 |
| `no_distance` | 0 | 0.002 |
| `no_heading` | 0.1 | 0 |
| `no_both` | 0 | 0 |

These are the current reference weights. The runner reads the source weights
rather than changing them. Descent is excluded because its weight is already
zero. The separate L1 displacement smoothness term remains commented out.
Time and fuel costs are unchanged.

Only objective coefficients change. In particular, **do not delete the heading
block**: its second-difference definitions also support the hard heading-change
constraints. ABS/MAX constraints, the speed/fuel relation, STAR selection,
arrival logic, separation, and physical limits remain unchanged.

## Run

From this directory, using the project's Python environment and Gurobi license:

```powershell
python .\run_v162_objective_ablation.py
python -m unittest test_v162_objective_ablation -v
```

The default is four **sequential** fresh-model solves with seed 0, the original
thread count (at most four), a 1,200-second solver limit, and a 1% MIP gap.
No plotting, downstream NLP, or normal output-directory deletion is executed.
The maximum screening solver budget is approximately 80 minutes; model building
and export are timed separately.

A result passes the practical runtime test if it reaches the gap target within
900 seconds and satisfies Gurobi's feasibility/integrality tolerances.
900 seconds is an experiment decision, not an operational deadline supplied by
the user. A result between 900 and 1,200 seconds is reported separately.
`TIME_LIMIT` with an incumbent is **not** infeasibility or proven divergence.

Repeat selected cases with matched additional seeds before generalizing:

```powershell
python .\run_v162_objective_ablation.py --seeds 1 2
python .\run_v162_objective_ablation.py --cases no_distance --time-limit 1800 --acceptable-seconds 900
```

For a slow case, test a shared feasible incumbent as a diagnostic, not as part
of the original-start benchmark:

```powershell
python .\run_v162_objective_ablation.py --cases no_distance --start-from "<baseline directory>\solution.sol"
```

Do not run other optimization jobs concurrently when comparing wall time.

## Evidence and reproducibility

[run_v162_objective_ablation.py](run_v162_objective_ablation.py) executes only
the original script's explicit model-building prefix, following the procedural
fixture pattern in [test_v181.py](test_v181.py). It fails if that boundary is
missing or duplicated. The original script is not edited or copied.

Outputs go under `Experiments/v162_objective_ablation/<timestamp>`, **outside**
the ordinary `Output` folder, which normal v16.2 execution can delete.
Each case has:

- Original and objective-zeroed model fingerprints, model sizes, original
  MIP-start hash, weights, and solver settings.
- Full Gurobi log, an exported compressed MPS model, and a parameter file.
- Progress sampled at five-second intervals when MIP callbacks are available,
  plus every solution callback. Callback intervals can be longer; use the
  result JSON for the authoritative final bound and gap.
- Solver runtime, deterministic work, nodes, first incumbent, final relative
  and absolute gaps, and solution-quality violations.
- A full solution and waypoint CSV whenever an incumbent exists.
- Fuel, time, distance, heading variation, and the original objective evaluated
  at the returned solution, so differing objectives are not compared naively.

The experiment manifest records Python/package versions and source/input
SHA-256 hashes. Every case verifies that the source/input files, constraints,
original objective, and original starts match. Summaries are saved after each
completed case. Infrastructure errors propagate visibly; incomplete batches
must not be interpreted as successful runs.

The original script's normal output branch requires `OPTIMAL`; this runner
also saves feasible time-limited incumbents so they are not mistaken for
infeasible models.

MPS alone does not preserve the initial MIP start; use this runner and the
recorded source/environment to reproduce the benchmark. Wall-time cutoffs are
load-dependent even when the model and random seed match.

## Interpreting slow or failed runs

Removing an objective term cannot make the unchanged feasible set infeasible.
A timeout can instead reflect difficulty finding a better incumbent, proving
a sufficiently strong lower bound, or both. The 1% relative gap uses the
incumbent magnitude as its denominator; removing a positive cost can also
reduce the absolute gap allowed at the same percentage.

Objective changes can alter branch-and-bound decisions and the strength of
objective-based pruning. Here the local fuel-table formulation and exact
MAX/ABS encodings remain in every case. Their combinatorial work does not
disappear just because a penalty is removed. These are mechanisms to investigate
using the logs, not explanations to assume before running the experiment.

The heading cost is an L1 change in horizontal displacement, not a literal
angular-rate penalty; it also penalizes changes in speed. Feasibility refers to
this waypoint MILP, not a continuous-time safety certification.

### Research sources

- [Gurobi: interpreting MIP logs](https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/mip.html)
- [Gurobi: what the MIP gap certifies](https://support.gurobi.com/hc/en-us/articles/8265539575953-What-is-the-MIPGap)
- [Gurobi: determinism, fingerprints, seeds, and time-dependent limits](https://support.gurobi.com/hc/en-us/articles/360031636051-Is-Gurobi-deterministic)
