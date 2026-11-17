# v16.2 objective ablation results

Objective-only changes; all physical constraints remain in place. OPTIMAL means within the configured 1% MIP gap, not exact optimality.

| Case | Seed | Outcome | Solve (s) | First feasible (s) | Gap (%) | Nodes |
|---|---:|---|---:|---:|---:|---:|
| baseline | 0 | PASS | 779.692 | 0.033 | 0.9533 | 37503 |
| no_distance | 0 | FEASIBLE_GAP_OPEN | 1200.154 | 0.073 | 2.5907 | 85989 |
| no_heading | 0 | PASS | 665.869 | 0.066 | 0.9563 | 30999 |
| no_both | 0 | SOLVED_OVER_TARGET | 956.965 | 0.117 | 0.9022 | 31792 |

A time limit with an incumbent is not infeasibility. Raw objective values across different objectives are not comparable; use physical metrics or original_objective_at_solution. Inspect each case's progress.csv and gurobi.log to distinguish incumbent search from bound improvement.
