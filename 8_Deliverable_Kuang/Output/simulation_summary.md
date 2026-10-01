# Simulation Summary

## Conditions

- **Script:** main_v15.py
- **Report time (UTC):** 2026-09-28T04:17:01.050761+00:00
- **Input files:** d:\Folder\3. Files\U-M\Year 2\Semester 1\U-M Lessons\FAA FAST Project\FAA-Fast-UMich\8_Deliverable_Kuang\Input\entry_exit_points.csv; d:\Folder\3. Files\U-M\Year 2\Semester 1\U-M Lessons\FAA FAST Project\FAA-Fast-UMich\8_Deliverable_Kuang\Input\star_fixes.csv
- **Cluster query (UTC) / window (min):** 08:11 / 15
- **Selected entry window (UTC):** 2023-12-22 08:10:01 to 2023-12-22 08:21:21
- **Flight subset:** ['DAL1120_KMSNtoKDTW', 'DAL1655_KPHLtoKDTW', 'EDV5018_CYULtoKDTW', 'EDV5043_KOMAtoKDTW', 'EDV5455_KGRBtoKDTW', 'NKS1449_KFLLtoKDTW']
- **Flight count:** 6
- **Grid epoch / end (UTC):** 2023-12-22 00:00:00 / 2023-12-22 08:40:00
- **Time step (s) / grid points / extra horizon steps:** 400.0 / 79 / 3
- **Projection origin (latitude, longitude):** 42.2125, -83.3534
- **Horizontal / vertical separation (m):** 152.4 / 30.48
- **Separation policy:** Waypoint checks only; v15 separation bypass is not linked to arrival. Not a safety guarantee.
- **Maximum per-axis horizontal / vertical displacement (m/step):** 120000.0 / 2032.0000000000005
- **Minimum horizontal displacement (m/step):** 24693.333333333332
- **Heading-change / speed-change limits (m/step):** 30000.0 / 47653.333333333336
- **Flight-path-angle tangent limit:** 0.08748866352592401
- **Big-M:** 10000000.0
- **Objective weights:** time=1.0, distance=0.1, smoothness=5e-07, vertical=0.1, acceleration=0.002, descent=0, final_altitude=0
- **Distance / fuel model:** Exact quadratic distance; no fuel in the routing objective.
- **Arrival interval:** Excluded from the v15 time penalty.
- **Solver time limit (s) / target gap:** inf / 0.01%
- **Solver threads / MIPFocus:** 0 / 0
- **Solver status / runtime (s):** OPTIMAL / 11.510
- **Objective / best bound / achieved gap:** 129996.268948 / 129994.951381 / 0.00%
- **Downstream analysis:** B737 for all flights; independent fuel-optimizing NLP retimes routes. Its results are not the routing schedule or a separation guarantee.

## Selected Flights

```text
   entry_utc               acId        entry_x        entry_y      entry_z  entry_epoch_s        exit_x        exit_y      exit_z  exit_epoch_s  flight_entry_timestep
08:10:01 UTC DAL1120_KMSNtoKDTW -227097.574537  121040.690954  8229.600000        29401.0 -49377.675828  53123.786280 3956.196963       30313.0                     73
08:18:06 UTC DAL1655_KPHLtoKDTW  216749.788083 -168012.095855  9144.000000        29886.0  46136.239399 -51471.125055 5403.392593       30933.0                     74
08:12:47 UTC EDV5018_CYULtoKDTW  233181.762956  100456.400666  8685.908120        29567.0  47416.490976  31514.770631 2195.157940       30707.0                     73
08:13:48 UTC EDV5043_KOMAtoKDTW -227983.236650  118056.557847 10058.400000        29628.0 -49340.671955  53155.569335 4037.277062       30506.0                     74
08:21:21 UTC EDV5455_KGRBtoKDTW -213458.502080  159218.851857  7620.000000        30081.0 -49365.430235  53134.304575 3423.375145       30962.0                     75
08:14:51 UTC NKS1449_KFLLtoKDTW  127525.298731 -288151.439107 10598.490775        29691.0  45954.045680 -51600.899875 6017.430526       30829.0                     74
```

## Terminal Results

### Flight Results

```text

====================================================
  FLIGHT RESULTS
====================================================
  Flight                     STAR fix        x (m)      y (m)    z (m)  exit_k      exit_utc
  DAL1120_KMSNtoKDTW         RKCTY          -49606      52928     3962      76  08:26:40 UTC
  DAL1655_KPHLtoKDTW         BONZZ           46143     -51467     6401      77  08:33:20 UTC
  EDV5018_CYULtoKDTW         FERRL           61083      22951     3048      76  08:26:40 UTC
  EDV5043_KOMAtoKDTW         RKCTY          -49606      52928     3962      77  08:33:20 UTC
  EDV5455_KGRBtoKDTW         LAYKS          -16010      71261     3048      78  08:40:00 UTC
  NKS1449_KFLLtoKDTW         CRAKN           34295     -59907     7925      78  08:40:00 UTC
====================================================


```

### Downstream NLP Analysis

```text
Program Start...  

====================================================
  Clip Historic Trajectory to Pre-TRACON Area
====================================================
  acId                        entry_lat   entry_lon  entry_alt  exit_lat    exit_lon  exit_alt
  DAL1120_KMSNtoKDTW            43.2672    -86.1585    27000.0   42.6887    -83.9575   12979.6
  DAL1655_KPHLtoKDTW            40.6724    -80.7832    30000.0   41.7483    -82.7973   17727.7
  EDV5018_CYULtoKDTW            43.0804    -80.4818    28497.1   42.4945    -82.7751    7202.0
  EDV5043_KOMAtoKDTW            43.2401    -86.1682    33000.0   42.6890    -83.9571   13245.7
  EDV5455_KGRBtoKDTW            43.6143    -86.0051    25000.0   42.6888    -83.9574   11231.5
  NKS1449_KFLLtoKDTW            39.6113    -81.8652    34772.0   41.7471    -82.7995   19742.2
====================================================

Historic Trajectory Fuel Estimation Success.

======================================================================
  NLP TIME SCALING — Historic → MILP
======================================================================
  acId                        hist_range(km)  hist_t(s)  hist_spd(m/s)  milp_range(km)  milp_t(s)  milp_spd(m/s)  nlp_range(km)  nlp_ts_guess       NLP_status
  DAL1120_KMSNtoKDTW               190      912        209       190      911        209      190         951  Solve_Succeeded
  DAL1655_KPHLtoKDTW               207     1047        197       207     1047        197      207        1033  Solve_Succeeded
  EDV5018_CYULtoKDTW               198     1140        174       189     1086        174      189         944  Solve_Succeeded
  EDV5043_KOMAtoKDTW               190      878        216       190      877        216      190         949  Solve_Succeeded
  EDV5455_KGRBtoKDTW               195      881        222       216      975        222      216        1081  Solve_Succeeded
  NKS1449_KFLLtoKDTW               250     1138        220       247     1121        220      247        1233  Solve_Succeeded
======================================================================

Optimized Trajectory Fuel Estimation Success.

======================================================================
  ALTITUDE AT ARRIVAL COMPARISON
======================================================================
              acId chosen_fix  fix_lat  fix_lon  fix_alt_max_ft  fix_alt_min_ft  milp_fix_alt_ft  hist_arrival_alt_ft  opt_arrival_alt_ft
DAL1120_KMSNtoKDTW      RKCTY  42.6869 -83.9603           13000           11000          13000.0              12980.0             12800.0
DAL1655_KPHLtoKDTW      BONZZ  41.7483 -82.7972           21000           15000          21000.0              17728.0             21200.0
EDV5018_CYULtoKDTW      FERRL  42.4165 -82.6093           10000            8000          10000.0               7202.0              9800.0
EDV5043_KOMAtoKDTW      RKCTY  42.6869 -83.9603           13000           11000          13000.0              13246.0             12800.0
EDV5455_KGRBtoKDTW      LAYKS  42.8532 -83.5498           10000           10000          10000.0              11232.0              9800.0
NKS1449_KFLLtoKDTW      CRAKN  41.6730 -82.9405           26000           12000          26000.0              19742.0             25800.0
======================================================================
  Saved → d:\Folder\3. Files\U-M\Year 2\Semester 1\U-M Lessons\FAA FAST Project\FAA-Fast-UMich\8_Deliverable_Kuang\Output\altitude_at_arrival.csv


======================================================================
  SUMMARY — Historic vs Optimized
======================================================================
  Flight                                   Hist time   Opt time     Δtime   Hist fuel   Opt fuel     Δfuel
  ──────────────────────────────────────         (s)        (s)       (s)        (kg)       (kg)      (kg)
  DAL1120_KMSNtoKDTW                             912        971       +59       320.9      300.7     -20.2
  DAL1655_KPHLtoKDTW                            1047        942      -105       315.3      325.7     +10.4
  EDV5018_CYULtoKDTW                            1140        982      -158       266.3      232.0     -34.4
  EDV5043_KOMAtoKDTW                             878        917       +39       272.7      205.9     -66.8
  EDV5455_KGRBtoKDTW                             881       1072      +191       368.9      414.2     +45.4
  NKS1449_KFLLtoKDTW                            1138       1018      -120       381.7      388.9      +7.2
  ──────────────────────────────────────  ──────────  ─────────  ────────  ──────────  ─────────  ────────
  TOTAL                                         5996       5902       -94      1925.7     1867.3     -58.4
  AVERAGE                                        999        984       -16       321.0      311.2      -9.7
======================================================================


Plots saved to d:\Folder\3. Files\U-M\Year 2\Semester 1\U-M Lessons\FAA FAST Project\FAA-Fast-UMich\8_Deliverable_Kuang\Output\plots  (83.16 s total)

```

## Appendix: MILP Solver Log

```text
Gurobi Optimizer version 12.0.3 build v12.0.3rc0 (win64 - Windows 11.0 (22000.2))

CPU model: 12th Gen Intel(R) Core(TM) i7-1260P, instruction set [SSE2|AVX|AVX2]
Thread count: 12 physical cores, 16 logical processors, using up to 16 threads

Non-default parameters:
LogToConsole  0

Optimize a model with 3330 rows, 9012 columns and 6382 nonzeros
Model fingerprint: 0xeb6194e9
Model has 25 quadratic constraints
Model has 208 simple general constraints
  113 ABS, 95 INDICATOR
Variable types: 7110 continuous, 1902 integer (1896 binary)
Coefficient statistics:
  Matrix range     [9e-02, 1e+07]
  QMatrix range    [1e+00, 1e+00]
  Objective range  [5e-08, 4e+02]
  Bounds range     [1e+00, 8e+01]
  RHS range        [1e+00, 1e+07]
  GenCon coe range [1e+00, 1e+00]
Presolve removed 2892 rows and 8643 columns
Presolve time: 0.27s
Presolved: 658 rows, 436 columns, 1800 nonzeros
Presolved model has 66 bilinear constraint(s)

Solving non-convex MIQCP

Variable types: 350 continuous, 86 integer (86 binary)

Root relaxation: objective 4.662050e+04, 397 iterations, 0.00 seconds (0.00 work units)

    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0 46620.5001    0   69          - 46620.5001      -     -    0s
     0     0 48850.4351    0   77          - 48850.4351      -     -    0s
     0     0 51173.8629    0   78          - 51173.8629      -     -    0s
     0     0 51233.5000    0   75          - 51233.5000      -     -    0s
     0     0 52673.8141    0   79          - 52673.8141      -     -    0s
     0     0 53936.9855    0   81          - 53936.9855      -     -    0s
     0     0 53936.9855    0   76          - 53936.9855      -     -    0s
     0     0 53936.9855    0   76          - 53936.9855      -     -    0s
     0     0 57179.6215    0   81          - 57179.6215      -     -    0s
     0     0 57657.7795    0   76          - 57657.7795      -     -    0s
     0     0 59134.9591    0   84          - 59134.9591      -     -    0s
     0     0 59134.9591    0   83          - 59134.9591      -     -    0s
     0     0 59144.1987    0   81          - 59144.1987      -     -    0s
     0     0 59157.2652    0   81          - 59157.2652      -     -    0s
     0     0 59173.7130    0   81          - 59173.7130      -     -    0s
     0     0 59173.7152    0   81          - 59173.7152      -     -    0s
     0     0 59176.9352    0   81          - 59176.9352      -     -    0s
     0     0 59358.3587    0   80          - 59358.3587      -     -    0s
     0     2 59358.3587    0   80          - 59358.3587      -     -    0s
  9027  7073 110851.251   73   20          - 65301.8644      -   5.5    9s
H 9027  6719                    140360.06870 65301.8644  53.5%   5.5    9s
H 9053  6400                    130013.85993 117926.373  9.30%   5.5   10s
*11290  5995              92    130001.04622 127407.487  2.00%   6.6   10s
H14322  4887                    129996.29948 128556.083  1.11%   8.0   10s
*22719   967              92    129996.26895 129975.113  0.02%   9.7   11s

Cutting planes:
  MIR: 2
  RLT: 4
  Relax-and-lift: 1

Explored 22954 nodes (221869 simplex iterations) in 11.51 seconds (3.43 work units)
Thread count was 16 (of 16 available processors)

Solution count 5: 129996 129996 130001 ... 140360

Optimal solution found (tolerance 1.00e-04)
Best objective 1.299962689480e+05, best bound 1.299949513807e+05, gap 0.0010%

User-callback calls 47226, time in user-callback 0.08 sec

```
