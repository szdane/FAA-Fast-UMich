# v16.2 objective ablation results

Objective-only changes; all physical constraints remain in place. OPTIMAL means within the configured 1% MIP gap, not exact optimality.

| Case | Seed | Outcome | Solve (s) | First feasible (s) | Gap (%) | Nodes |
|---|---:|---|---:|---:|---:|---:|
| no_distance | 1 | FEASIBLE_GAP_OPEN | 1200.166 | 0.000 | 1.1544 | 104287 |

A time limit with an incumbent is not infeasibility. Raw objective values across different objectives are not comparable; use physical metrics or original_objective_at_solution. Inspect each case's progress.csv and gurobi.log to distinguish incumbent search from bound improvement.
