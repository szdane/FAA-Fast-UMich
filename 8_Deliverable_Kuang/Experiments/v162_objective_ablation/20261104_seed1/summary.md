# v16.2 objective ablation results

Objective-only changes; all physical constraints remain in place. OPTIMAL means within the configured 1% MIP gap, not exact optimality.

| Case | Seed | Outcome | Solve (s) | First feasible (s) | Gap (%) | Nodes |
|---|---:|---|---:|---:|---:|---:|
| baseline | 1 | PASS | 381.401 | 0.077 | 0.7702 | 20756 |
| no_distance | 1 | FEASIBLE_GAP_OPEN | 1200.192 | 0.089 | 1.7430 | 76032 |
| no_heading | 1 | PASS | 453.625 | 0.158 | 0.9217 | 29785 |
| no_both | 1 | FEASIBLE_GAP_OPEN | 1200.105 | 0.079 | 4.4040 | 46399 |

A time limit with an incumbent is not infeasibility. Raw objective values across different objectives are not comparable; use physical metrics or original_objective_at_solution. Inspect each case's progress.csv and gurobi.log to distinguish incumbent search from bound improvement.
