"""
Functions/acm_milp.py
=====================
Active Corner Method (ACM) — segment-square separation constraints for MILP.

Replaces the axis-aligned box separation in main_v1x.py with a direction-aware
segment-square intersection check.  For each aircraft pair (i,j) and each time
segment [k-1 → k], enforces that the relative trajectory segment does NOT pass
through the square exclusion zone of half-width w centred at the origin in the
relative frame (flight i fixed at origin, flight j moving).

Mathematical structure (PDF sketch Steps 1–8):

  Step 1  Sort the two relative endpoints by x-coordinate
            → 1 binary z_sort; 4 continuous aux vars xL, xR, yL, yR
  Step 2  Select active corner from sign(yR − yL)
            → 1 binary z_ac
  Step 3  Shifted-clipped-trajectory condition  g1·g2 ≤ 0
            G1 = w·(xR−xL) − c_x·(yR−yL) + CROSS
            G2 = −w·(xR−xL) + c_x·(yR−yL) + CROSS
            where c_x = w·(1−2·z_ac)  (= −w when z_ac=1, +w when z_ac=0)
            CROSS = yR·xL − yL·xR  ← bilinear, handled via McCormick
  Step 4  Corner-pair-line condition  L1·L2 ≤ 0
            L1 = w·(xL − yL) + 2w·p_zyL   [fully linear]
            L2 = w·(xR − yR) + 2w·p_zyR   [fully linear]
  Step 5  Domain bound  x_left ≤ w  AND  x_right ≥ −w
  Step 7  Notch conditions (endpoint squares in y-direction)
  Step 8  Vertical separation

Safety escape structure — at least ONE of the 12 clauses must hold:

  idx  name       meaning
  ---  ---------  ----------------------------------------------------------
   0   dom_E      x_left  ≥  w   (segment entirely east of box)
   1   dom_W      x_right ≤ −w   (segment entirely west of box)
   2   g_pos      G1 ≥ 0  AND  G2 ≥ 0  (origin above trajectory band)
   3   g_neg      G1 ≤ 0  AND  G2 ≤ 0  (origin below trajectory band)
   4   L_pos      L1 ≥ 0  AND  L2 ≥ 0  (origin outside corner lines, + side)
   5   L_neg      L1 ≤ 0  AND  L2 ≤ 0  (origin outside corner lines, − side)
   6   nLy_p      y_left  ≥  w   (left  endpoint above box — notch N)
   7   nLy_n      y_left  ≤ −w   (left  endpoint below box — notch S)
   8   nRy_p      y_right ≥  w   (right endpoint above box — notch N)
   9   nRy_n      y_right ≤ −w   (right endpoint below box — notch S)
  10   z_ij       f_z[i][k] − f_z[j][k] ≥ sep_vert_m
  11   z_ji       f_z[j][k] − f_z[i][k] ≥ sep_vert_m

SOUNDNESS NOTE:
  Clauses 2–3 use McCormick relaxation for CROSS = yR·xL − yL·xR.  The
  relaxation is an OUTER approximation: the feasible set is larger than the
  exact bilinear set.  This means bv[2] or bv[3] may be set to 1 even when
  the true G-condition is not satisfied → potential false safety.  To get an
  exact check, set m.setParam("NonConvex", 2) and replace pA/pB with true
  quadratic constraints.  Clauses 0–1, 4–11 are exact (linear or exact
  binary-×-continuous linearisation).
"""

from __future__ import annotations
from gurobipy import GRB, Model, quicksum


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def add_acm_separation_constraints(
    m: Model,
    f_x: list,
    f_y: list,
    f_z: list,
    flights,            # pandas DataFrame with columns flight_entry_timestep, entry_x, entry_y
    N_flights: int,
    N_steps: int,
    w: float,           # half-width of the square exclusion zone (= SEP_HOR_M)
    sep_vert_m: float,  # vertical separation minimum (= SEP_VERT_M)
    sep_bypass,         # Gurobi binary var dict [N_flights × N_steps]; 1 = flight landed
    big_m: float,       # big-M for linear constraints (e.g. 1e7)
    pos_bound: float,   # fallback symmetric bound on |relative position| (m)
    vmax_xy: float = None,  # if provided, compute per-pair bounds from entry positions
                            # + vmax_xy * steps; much tighter than pos_bound
) -> None:
    """
    Add ACM segment-square separation constraints to Gurobi model *m*.

    Call INSTEAD OF the current axis-aligned box loop in main_v1x.py.
    The separation bypass (landed flights) is propagated identically to the
    original constraints.

    Parameters
    ----------
    m           : Gurobi Model (already created, variables already added).
    f_x, f_y    : list of N_flights Gurobi var dicts, indexed by step k.
    f_z         : list of N_flights Gurobi var dicts, indexed by step k.
    flights     : DataFrame row i gives flight i metadata.
    N_flights   : number of flights.
    N_steps     : number of time steps (k = 0 … N_steps-1).
    w           : half-width of exclusion square in metres.
    sep_vert_m  : minimum vertical separation in metres.
    sep_bypass  : Gurobi binary vars [i, k]; 1 when flight i is parked.
    big_m       : big-M constant for linear (non-bilinear) constraints.
    pos_bound   : fallback bound (m) used when vmax_xy is None.
    vmax_xy     : if given, per-pair-step bounds are derived from entry positions
                  + vmax_xy*(steps_i + steps_j + 2), separately for x and y.
    """

    PB_x = PB_y = pos_bound   # global fallback; overridden per-pair below
    n_constraints_added = 0

    for k in range(1, N_steps):
        for i in range(N_flights - 1):
            for j in range(i + 1, N_flights):

                ki = int(flights.iloc[i]['flight_entry_timestep'])
                kj = int(flights.iloc[j]['flight_entry_timestep'])
                # Skip segment if either flight is not yet active at k-1 or k
                if k - 1 < ki or k - 1 < kj:
                    continue

                # Per-pair bounds: entry separation + max reachable displacement
                if vmax_xy is not None:
                    steps_i = max(0, k - 1 - ki)
                    steps_j = max(0, k - 1 - kj)
                    slack   = vmax_xy * (steps_i + steps_j + 2)
                    PB_x = max(abs(float(flights.iloc[j]['entry_x']) - float(flights.iloc[i]['entry_x'])) + slack, w * 2)
                    PB_y = max(abs(float(flights.iloc[j]['entry_y']) - float(flights.iloc[i]['entry_y'])) + slack, w * 2)
                else:
                    PB_x = PB_y = pos_bound

                DB_y = 2.0 * PB_y                                          # bound on |yR − yL|
                BM_G = max(big_m, 2*w*PB_x + 6*w*PB_y + 2*PB_x*PB_y)    # max |G1|, |G2|
                BM_L = max(big_m, w * (PB_x + 3 * PB_y))                  # max |L1|, |L2|

                tag   = f"{i}_{j}_{k}"
                bp_i  = sep_bypass[i, k]
                bp_j  = sep_bypass[j, k]
                # Convenience: bypass relaxation term added to RHS of each clause
                BY    = big_m * bp_i + big_m * bp_j

                # ── Raw relative positions (LinExpr) ──────────────────────────
                xp = f_x[j][k-1] - f_x[i][k-1]   # B/A at step k-1
                yp = f_y[j][k-1] - f_y[i][k-1]
                xc = f_x[j][k]   - f_x[i][k]     # B/A at step k
                yc = f_y[j][k]   - f_y[i][k]

                # ── Step 1: Sorting — z_sort=1 iff xp ≤ xc ───────────────────
                z_sort = m.addVar(vtype=GRB.BINARY, name=f"zs_{tag}")
                # (xc − xp) ≥ 0 when z_sort=1; (xc − xp) ≤ 0 when z_sort=0
                m.addConstr((xc - xp) >= -big_m * (1 - z_sort), f"sort_lo_{tag}")
                m.addConstr((xc - xp) <=  big_m * z_sort,        f"sort_hi_{tag}")

                xL = m.addVar(lb=-PB_x, ub=PB_x, name=f"xL_{tag}")
                xR = m.addVar(lb=-PB_x, ub=PB_x, name=f"xR_{tag}")
                yL = m.addVar(lb=-PB_y, ub=PB_y, name=f"yL_{tag}")
                yR = m.addVar(lb=-PB_y, ub=PB_y, name=f"yR_{tag}")

                # Conditional assignment: if z_sort=1 → (xL,yL)=(xp,yp), (xR,yR)=(xc,yc)
                #                        if z_sort=0 → (xL,yL)=(xc,yc), (xR,yR)=(xp,yp)
                _pin(m, xL, xp, xc, z_sort, big_m, f"xL_{tag}")
                _pin(m, xR, xc, xp, z_sort, big_m, f"xR_{tag}")
                _pin(m, yL, yp, yc, z_sort, big_m, f"yL_{tag}")
                _pin(m, yR, yc, yp, z_sort, big_m, f"yR_{tag}")

                # ── Step 2: Active corner — z_ac=1 iff yR ≥ yL ───────────────
                z_ac = m.addVar(vtype=GRB.BINARY, name=f"zac_{tag}")
                m.addConstr((yR - yL) >= -big_m * (1 - z_ac), f"ac_lo_{tag}")
                m.addConstr((yR - yL) <=  big_m * z_ac,        f"ac_hi_{tag}")

                # ── Binary × continuous exact linearisation ───────────────────
                # p_zy  = z_ac · (yR − yL) ∈ [−DB_y, DB_y]
                # p_zyL = z_ac · yL         ∈ [−PB_y, PB_y]
                # p_zyR = z_ac · yR         ∈ [−PB_y, PB_y]
                p_zy  = _bin_times_lin(m, z_ac, yR - yL, -DB_y, DB_y, f"pzy_{tag}")
                p_zyL = _bin_times_lin(m, z_ac, yL,      -PB_y, PB_y, f"pzyL_{tag}")
                p_zyR = _bin_times_lin(m, z_ac, yR,      -PB_y, PB_y, f"pzyR_{tag}")

                # ── Step 3: CROSS = yR·xL − yL·xR (McCormick) ───────────────
                # See soundness note in module docstring.
                pA = _mccormick(m, yR, xL, -PB_y, PB_y, -PB_x, PB_x, f"pA_{tag}")
                pB = _mccormick(m, yL, xR, -PB_y, PB_y, -PB_x, PB_x, f"pB_{tag}")

                # G1 = w·(xR−xL) − w·(yR−yL) + 2w·p_zy + CROSS   (LinExpr)
                # G2 = −w·(xR−xL) + w·(yR−yL) − 2w·p_zy + CROSS
                G1 =  w*(xR - xL) - w*(yR - yL) + 2*w*p_zy + pA - pB
                G2 = -w*(xR - xL) + w*(yR - yL) - 2*w*p_zy + pA - pB

                # ── Step 4: L1, L2 (corner-pair lines, fully linear) ──────────
                # L1 = c_y·xL − c_x·yL  at (x,y)=(0,0)
                #    = w·xL − (w − 2w·z_ac)·yL = w·(xL − yL) + 2w·p_zyL
                # L2 = w·(xR − yR) + 2w·p_zyR
                L1 = w*(xL - yL) + 2*w*p_zyL
                L2 = w*(xR - yR) + 2*w*p_zyR

                # ── Escape binary variables (12 clauses) ─────────────────────
                bv = m.addVars(range(12), vtype=GRB.BINARY, name=f"bv_{tag}")

                # 0: dom_E — x_left ≥ w
                m.addConstr(xL - w >= -big_m*(1-bv[0]) - BY, f"domE_{tag}")

                # 1: dom_W — x_right ≤ −w
                m.addConstr(-xR - w >= -big_m*(1-bv[1]) - BY, f"domW_{tag}")

                # 2: g_pos — G1 ≥ 0 AND G2 ≥ 0  (McCormick, see soundness note)
                m.addConstr(G1 >= -BM_G*(1-bv[2]) - BY, f"g1pos_{tag}")
                m.addConstr(G2 >= -BM_G*(1-bv[2]) - BY, f"g2pos_{tag}")

                # 3: g_neg — G1 ≤ 0 AND G2 ≤ 0
                m.addConstr(G1 <= BM_G*(1-bv[3]) + BY, f"g1neg_{tag}")
                m.addConstr(G2 <= BM_G*(1-bv[3]) + BY, f"g2neg_{tag}")

                # 4: L_pos — L1 ≥ 0 AND L2 ≥ 0  (exact; BM_L = w·(PB_x + 3·PB_y))
                m.addConstr(L1 >= -BM_L*(1-bv[4]) - BY, f"L1pos_{tag}")
                m.addConstr(L2 >= -BM_L*(1-bv[4]) - BY, f"L2pos_{tag}")

                # 5: L_neg — L1 ≤ 0 AND L2 ≤ 0  (exact)
                m.addConstr(L1 <= BM_L*(1-bv[5]) + BY, f"L1neg_{tag}")
                m.addConstr(L2 <= BM_L*(1-bv[5]) + BY, f"L2neg_{tag}")

                # 6: north — BOTH endpoints above box  (yL ≥ w AND yR ≥ w)
                m.addConstr(yL - w >= -big_m*(1-bv[6]) - BY, f"nLyp_{tag}")
                m.addConstr(yR - w >= -big_m*(1-bv[6]) - BY, f"nRyp_{tag}")

                # 7: south — BOTH endpoints below box  (yL ≤ −w AND yR ≤ −w)
                m.addConstr(-yL - w >= -big_m*(1-bv[7]) - BY, f"nLyn_{tag}")
                m.addConstr(-yR - w >= -big_m*(1-bv[7]) - BY, f"nRyn_{tag}")

                # 8: z_ij — vertical separation i above j
                m.addConstr(
                    f_z[i][k] - f_z[j][k] >= sep_vert_m
                    - big_m*(1-bv[8]) - BY,
                    f"zij_{tag}",
                )

                # 9: z_ji — vertical separation j above i
                m.addConstr(
                    f_z[j][k] - f_z[i][k] >= sep_vert_m
                    - big_m*(1-bv[9]) - BY,
                    f"zji_{tag}",
                )

                # At least one escape must hold (10 clauses)
                m.addConstr(
                    quicksum(bv[n] for n in range(10)) >= 1,
                    f"esc_{tag}",
                )

                n_constraints_added += 1

    print(f"ACM separation constraints added: {n_constraints_added} segments checked.")


# ---------------------------------------------------------------------------
# Helper functions (module-private)
# ---------------------------------------------------------------------------

def _pin(m, var, val_if_z1, val_if_z0, z, big_m, name):
    """
    Conditional assignment: var = val_if_z1 when z=1, val_if_z0 when z=0.
    val_if_z1 and val_if_z0 may be Gurobi LinExpr or float.
    """
    m.addConstr(var >= val_if_z1 - big_m*(1 - z), f"{name}_lo1")
    m.addConstr(var <= val_if_z1 + big_m*(1 - z), f"{name}_hi1")
    m.addConstr(var >= val_if_z0 - big_m*z,        f"{name}_lo0")
    m.addConstr(var <= val_if_z0 + big_m*z,        f"{name}_hi0")


def _bin_times_lin(m, z, expr, lo, hi, name):
    """
    Exact linearisation of p = z * expr for binary z and bounded linear expr.

    Constraints:
        p ≥ lo · z          (p ≥ lo when z=1)
        p ≤ hi · z          (p ≤ hi when z=1; p ≤ 0 when z=0)
        p ≥ expr − hi·(1−z) (p ≥ expr when z=1; trivial when z=0)
        p ≤ expr − lo·(1−z) (p ≤ expr when z=1; trivial when z=0)

    Returns the new Gurobi variable p.
    """
    p = m.addVar(lb=lo, ub=hi, name=name)
    m.addConstr(p >= lo * z,               f"{name}_blo")
    m.addConstr(p <= hi * z,               f"{name}_bhi")
    m.addConstr(p >= expr - hi*(1 - z),    f"{name}_clo")
    m.addConstr(p <= expr - lo*(1 - z),    f"{name}_chi")
    return p


def _mccormick(m, u, v, u_lo, u_hi, v_lo, v_hi, name):
    """
    McCormick outer-relaxation of p ≈ u·v for bounded continuous u, v.

    Four envelope constraints (two lower, two upper):
        p ≥ u_lo·v + u·v_lo − u_lo·v_lo
        p ≥ u_hi·v + u·v_hi − u_hi·v_hi
        p ≤ u_hi·v + u·v_lo − u_hi·v_lo
        p ≤ u_lo·v + u·v_hi − u_lo·v_hi

    Returns the new Gurobi variable p (lb/ub computed from corner products).
    """
    corners = [u_lo*v_lo, u_lo*v_hi, u_hi*v_lo, u_hi*v_hi]
    p = m.addVar(lb=min(corners), ub=max(corners), name=name)
    m.addConstr(p >= u_lo*v + u*v_lo - u_lo*v_lo, f"{name}_mc1")
    m.addConstr(p >= u_hi*v + u*v_hi - u_hi*v_hi, f"{name}_mc2")
    m.addConstr(p <= u_hi*v + u*v_lo - u_hi*v_lo, f"{name}_mc3")
    m.addConstr(p <= u_lo*v + u*v_hi - u_lo*v_hi, f"{name}_mc4")
    return p


# ---------------------------------------------------------------------------
# Pure-Python checker (no Gurobi) — for unit testing
# ---------------------------------------------------------------------------

def acm_check_conflict(
    xAp: float, yAp: float, xAk: float, yAk: float,
    xBp: float, yBp: float, xBk: float, yBk: float,
    w: float,
) -> bool:
    """
    Exact check: does the relative trajectory segment of B w.r.t. A pass through
    the square exclusion zone [-w, w]^2 centred at the origin?

    Returns True if a conflict is detected, False if safe.

    Uses the exact segment-vs-axis-aligned-box intersection test (interval
    intersection on the parameter t ∈ [0,1]) rather than the ACM band
    approximation, so it is correct for all cases including endpoint-in-box
    and diagonal crossings.
    """
    # Relative positions
    xp = xBp - xAp
    yp = yBp - yAp
    xc = xBk - xAk
    yc = yBk - yAk

    dx = xc - xp
    dy = yc - yp

    # Find t-range where x-coordinate of segment is in [-w, w]
    if abs(dx) < 1e-12:
        if abs(xp) > w:
            return False  # x never in [-w,w]
        tx_lo, tx_hi = 0.0, 1.0
    else:
        t1 = (-w - xp) / dx
        t2 = ( w - xp) / dx
        tx_lo, tx_hi = (t1, t2) if dx > 0 else (t2, t1)

    # Find t-range where y-coordinate of segment is in [-w, w]
    if abs(dy) < 1e-12:
        if abs(yp) > w:
            return False  # y never in [-w,w]
        ty_lo, ty_hi = 0.0, 1.0
    else:
        t1 = (-w - yp) / dy
        t2 = ( w - yp) / dy
        ty_lo, ty_hi = (t1, t2) if dy > 0 else (t2, t1)

    # Intersection of both t-ranges and [0, 1]
    t_lo = max(tx_lo, ty_lo, 0.0)
    t_hi = min(tx_hi, ty_hi, 1.0)

    return t_lo <= t_hi
