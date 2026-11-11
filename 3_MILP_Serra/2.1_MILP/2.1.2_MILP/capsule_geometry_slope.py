"""
Active-corner capsule separation geometry - scalar-slope formulation.

An explicit segment slope m = dy/dx and perpendicular
slope m' = -dx/dy, used directly in "y {>=,<=} m*x + b"-style line tests,
rather than the vector dot/cross-product forms in capsule_geometry.py.

Two consequences of using m and m' directly (rather than cross-multiplying
by dx/dy the way capsule_geometry.py does):

  - m is undefined for a vertical segment (dx=0); m' is undefined for a
    horizontal one (dy=0). Each gets an explicit fallback to a vertical-
    line test in that one case.

  - (1) and (6) pass exactly through the far/near waypoint (no offset), so
    their sense only has to encode "which side is the direction of
    travel" - but because m' is used directly (not cross-multiplied),
    picking that sense genuinely depends on the segment's direction
    (specifically sign(dy), tie-broken by sign(dx) when dy==0), unlike
    capsule_geometry.py's dot-product form, which is already direction-
    consistent for any (dx,dy). So this version does need a small
    direction lookup for (1)/(6). (3)/(4) need no such lookup: each is a
    direction-invariant "clear of the strip on this fixed side" test,
    expressed as the line through the near waypoint offset by
    delta_sep*sqrt(1+m^2) - the correct perpendicular-to-a-slope-m-line
    offset - computed via Gurobi's power general constraint.
"""
from gurobipy import GRB
import math


def build_seg_geom_slope(m, diffx1, diffy1, tag, eps=1e-3, big_m=1e5, slope_bound=1e4):
    """Segment geometry for the scalar-slope formulation: segment length,
    direction-sign indicators, and the (conditionally-defined) slope and
    perpendicular slope, returning the geom dict consumed by
    add_capsule_direction_slope.

    eps must be safely larger than Gurobi's default FeasibilityTol (1e-6) -
    see capsule_geometry.build_seg_geom for why.

    slope_bound caps how large m_slope/m_perp (and the terms derived from
    them) are allowed to be. Unlike the vector formulation in
    capsule_geometry.py, an explicit slope is fundamentally unbounded near
    a vertical/horizontal segment (m_slope = dy/dx blows up as dx -> eps),
    and Gurobi's nonconvex solver needs finite variable bounds to build a
    useful relaxation for the bilinear terms that use it - leaving it
    unbounded produces the "variables with very large bounds participating
    in nonlinear terms" warning and a much slower solve. The true worst
    case within the "regular" (non-fallback) branch is bounded by
    (max plausible |dx| or |dy| over one step) / eps, so pass a
    slope_bound comfortably above that for your problem's actual position
    and velocity scale; the default (1e4) is a generic, permissive cap.
    """
    seglen = m.addVar(name=f"seglenS_{tag}")
    m.addConstr(seglen * seglen == diffx1 * diffx1 + diffy1 * diffy1, name=f"seglenS_def_{tag}")

    moves = m.addVar(vtype=GRB.BINARY, name=f"movesS_{tag}")
    m.addGenConstrIndicator(moves, 1, seglen, GRB.GREATER_EQUAL, eps, name=f"movesS_on_{tag}")
    m.addGenConstrIndicator(moves, 0, seglen, GRB.LESS_EQUAL, eps, name=f"movesS_off_{tag}")

    dxpos = m.addVar(vtype=GRB.BINARY, name=f"dxposS_{tag}")
    m.addGenConstrIndicator(dxpos, 1, diffx1, GRB.GREATER_EQUAL, eps, name=f"dxposS_on_{tag}")
    m.addGenConstrIndicator(dxpos, 0, diffx1, GRB.LESS_EQUAL, eps, name=f"dxposS_off_{tag}")

    dxneg = m.addVar(vtype=GRB.BINARY, name=f"dxnegS_{tag}")
    m.addGenConstrIndicator(dxneg, 1, diffx1, GRB.LESS_EQUAL, -eps, name=f"dxnegS_on_{tag}")
    m.addGenConstrIndicator(dxneg, 0, diffx1, GRB.GREATER_EQUAL, -eps, name=f"dxnegS_off_{tag}")

    dypos = m.addVar(vtype=GRB.BINARY, name=f"dyposS_{tag}")
    m.addGenConstrIndicator(dypos, 1, diffy1, GRB.GREATER_EQUAL, eps, name=f"dyposS_on_{tag}")
    m.addGenConstrIndicator(dypos, 0, diffy1, GRB.LESS_EQUAL, eps, name=f"dyposS_off_{tag}")

    dyneg = m.addVar(vtype=GRB.BINARY, name=f"dynegS_{tag}")
    m.addGenConstrIndicator(dyneg, 1, diffy1, GRB.LESS_EQUAL, -eps, name=f"dynegS_on_{tag}")
    m.addGenConstrIndicator(dyneg, 0, diffy1, GRB.GREATER_EQUAL, -eps, name=f"dynegS_off_{tag}")

    # m_slope = dy/dx, pinned down (m_slope*dx == dy) only while dx != 0;
    # left free (and unused) on the vertical-segment fallback branch.
    # Bounded (see slope_bound above) so Gurobi can build a useful
    # relaxation for the bilinear terms it appears in.
    m_slope = m.addVar(lb=-slope_bound, ub=slope_bound, name=f"mslope_{tag}")
    m.addConstr(m_slope * diffx1 - diffy1 <= big_m * (1 - dxpos - dxneg), name=f"mslope_hi_{tag}")
    m.addConstr(m_slope * diffx1 - diffy1 >= -big_m * (1 - dxpos - dxneg), name=f"mslope_lo_{tag}")

    # m_perp = -dx/dy, pinned down only while dy != 0.
    m_perp = m.addVar(lb=-slope_bound, ub=slope_bound, name=f"mperp_{tag}")
    m.addConstr(m_perp * diffy1 + diffx1 <= big_m * (1 - dypos - dyneg), name=f"mperp_hi_{tag}")
    m.addConstr(m_perp * diffy1 + diffx1 >= -big_m * (1 - dypos - dyneg), name=f"mperp_lo_{tag}")

    return dict(dx=diffx1, dy=diffy1, L=seglen, moves=moves,
                dxpos=dxpos, dxneg=dxneg, dypos=dypos, dyneg=dyneg,
                m_slope=m_slope, m_perp=m_perp, slope_bound=slope_bound)


def add_capsule_direction_slope(m, geom, x_far, y_far, x_near, y_near, x_o, y_o,
                                 bin_var, relax_common, delta_sep, tag, big_m,
                                 endcap="circle"):
    """Scalar-slope counterpart to capsule_geometry.add_capsule_direction:
    same 4-way disjunction ((3) or (4) or (1 AND 2) or (5 AND 6)), same
    external contract, different internal algebra.

    endcap="circle" uses the Euclidean disk of radius delta_sep around each
    waypoint for (2)/(5); endcap="square" replaces it with the axis-aligned
    square of half-width delta_sep (|dx| >= delta OR |dy| >= delta), which
    is linear and contains the disk. The strip lines (3)/(4) are unchanged."""
    dxpos, dxneg = geom['dxpos'], geom['dxneg']
    dypos, dyneg = geom['dypos'], geom['dyneg']
    m_slope, m_perp = geom['m_slope'], geom['m_perp']
    moves = geom['moves']

    branch_relax = big_m * (1 - bin_var) + relax_common

    case_vars = m.addVars(4, vtype=GRB.BINARY, name=f"capcaseS_{tag}")
    case_A, case_B, case_C, case_D = case_vars[0], case_vars[1], case_vars[2], case_vars[3]
    m.addConstr(case_A + case_B + case_C + case_D >= 1, name=f"capcaseS_or_{tag}")
    m.addConstr(case_A + case_B <= moves, name=f"capcaseS_moves_{tag}")

    not_vertical = big_m * (1 - dxpos - dxneg)  # relax the regular (3)/(4) form away when dx==0
    is_vertical = big_m * (dxpos + dxneg)       # relax the vertical fallback away when dx!=0

    # --- (3),(4): strip sides, via the segment's own slope ------------------
    # Direction-invariant: case_A is always the "+offset" side, case_B the
    # "-offset" side, regardless of which way the segment points - no
    # direction lookup needed. The perpendicular offset for a line of slope
    # m is delta_sep*sqrt(1+m^2).
    # Explicit finite bounds (derived from m_slope's own bound) so presolve
    # doesn't have to re-derive them through the quadratic/power constraints.
    m_slope_ub = geom['slope_bound']  # m_slope.UB is unreadable before m.update()
    msq_ub = m_slope_ub * m_slope_ub
    msq = m.addVar(lb=0, ub=msq_ub, name=f"mslopesq_{tag}")
    m.addConstr(msq == m_slope * m_slope, name=f"mslopesq_def_{tag}")
    one_plus_msq = m.addVar(lb=1, ub=1 + msq_ub, name=f"one_plus_mslopesq_{tag}")
    m.addConstr(one_plus_msq == 1 + msq, name=f"one_plus_mslopesq_def_{tag}")
    slope_off = m.addVar(lb=0, ub=(1 + msq_ub) ** 0.5, name=f"slope_off_{tag}")
    m.addGenConstrPow(one_plus_msq, slope_off, 0.5, name=f"slope_off_sqrt_{tag}")

    # regular (dx != 0): y_o >= m*(x_o - x_near) + y_near + delta*sqrt(1+m^2)
    m.addConstr(
        y_o - m_slope * (x_o - x_near) - y_near >=
        delta_sep * slope_off - branch_relax - big_m * (1 - case_A) - not_vertical,
        name=f"cap3_reg_{tag}")
    m.addConstr(
        y_o - m_slope * (x_o - x_near) - y_near <=
        -delta_sep * slope_off + branch_relax + big_m * (1 - case_B) + not_vertical,
        name=f"cap4_reg_{tag}")

    # vertical fallback (dx == 0): the segment (and its offset lines) is
    # vertical, so the lines parallel to it, offset by delta_sep, are
    # vertical too: x_o >= x_near + delta_sep  /  x_o <= x_near - delta_sep
    m.addConstr(
        x_o - x_near >= delta_sep - branch_relax - big_m * (1 - case_A) - is_vertical,
        name=f"cap3_vert_{tag}")
    m.addConstr(
        x_o - x_near <= -delta_sep + branch_relax + big_m * (1 - case_B) + is_vertical,
        name=f"cap4_vert_{tag}")

    # --- (1),(2): far end-cap -------------------------------------------
    # (1) passes exactly through the far waypoint, using the perpendicular
    # slope m' directly, so (unlike (3)/(4)) its sense genuinely depends on
    # direction: sign(dy), tie-broken by sign(dx) when dy==0. m' itself is
    # undefined when dy==0 (horizontal segment), which gets its own
    # vertical-line fallback (perpendicular to horizontal is vertical).
    c1_relax = branch_relax + big_m * (1 - case_C)

    m.addConstr(  # dy > 0
        y_o - m_perp * (x_o - x_far) - y_far >= 0 - c1_relax - big_m * (1 - dypos),
        name=f"cap1_dypos_{tag}")
    m.addConstr(  # dy < 0
        y_o - m_perp * (x_o - x_far) - y_far <= 0 + c1_relax + big_m * (1 - dyneg),
        name=f"cap1_dyneg_{tag}")
    m.addConstr(  # dy == 0, dx > 0
        x_o - x_far >= 0 - c1_relax - big_m * dypos - big_m * dyneg - big_m * (1 - dxpos),
        name=f"cap1_horiz_pos_{tag}")
    m.addConstr(  # dy == 0, dx < 0
        x_o - x_far <= 0 + c1_relax + big_m * dypos + big_m * dyneg + big_m * (1 - dxneg),
        name=f"cap1_horiz_neg_{tag}")

    add_waypoint_clearance(m, x_far, y_far, x_o, y_o, delta_sep, c1_relax,
                           f"cap2_{tag}", big_m, endcap)

    # --- (5),(6): near end-cap - mirror of (1),(2), senses reversed --------
    c6_relax = branch_relax + big_m * (1 - case_D)

    m.addConstr(  # dy > 0
        y_o - m_perp * (x_o - x_near) - y_near <= 0 + c6_relax + big_m * (1 - dypos),
        name=f"cap6_dypos_{tag}")
    m.addConstr(  # dy < 0
        y_o - m_perp * (x_o - x_near) - y_near >= 0 - c6_relax - big_m * (1 - dyneg),
        name=f"cap6_dyneg_{tag}")
    m.addConstr(  # dy == 0, dx > 0
        x_o - x_near <= 0 + c6_relax + big_m * dypos + big_m * dyneg + big_m * (1 - dxpos),
        name=f"cap6_horiz_pos_{tag}")
    m.addConstr(  # dy == 0, dx < 0
        x_o - x_near >= 0 - c6_relax - big_m * dypos - big_m * dyneg - big_m * (1 - dxneg),
        name=f"cap6_horiz_neg_{tag}")

    add_waypoint_clearance(m, x_near, y_near, x_o, y_o, delta_sep, c6_relax,
                           f"cap5_{tag}", big_m, endcap)


def add_waypoint_clearance(m, x_w, y_w, x_o, y_o, delta_sep, relax, name, big_m,
                           endcap):
    """Point (x_o,y_o) is at least delta_sep from waypoint (x_w,y_w), unless
    relaxed. "circle": Euclidean distance. "square": outside the axis-aligned
    square of half-width delta_sep, i.e. one of the four half-planes holds.
    "octagon": outside the regular octagon circumscribing the circle, i.e.
    one of the four axis or four diagonal half-planes holds."""
    if endcap == "circle":
        m.addConstr(
            (x_w - x_o) * (x_w - x_o) + (y_w - y_o) * (y_w - y_o) >=
            delta_sep * delta_sep - relax,
            name=name)
    elif endcap == "square":
        sq = m.addVars(4, vtype=GRB.BINARY, name=f"{name}_sq")
        m.addConstr(sq.sum() >= 1, name=f"{name}_sq_or")
        m.addConstr(x_o - x_w >= delta_sep - relax - big_m * (1 - sq[0]), name=f"{name}_sq_xpos")
        m.addConstr(x_w - x_o >= delta_sep - relax - big_m * (1 - sq[1]), name=f"{name}_sq_xneg")
        m.addConstr(y_o - y_w >= delta_sep - relax - big_m * (1 - sq[2]), name=f"{name}_sq_ypos")
        m.addConstr(y_w - y_o >= delta_sep - relax - big_m * (1 - sq[3]), name=f"{name}_sq_yneg")
    elif endcap == "octagon":
        sq = m.addVars(8, vtype=GRB.BINARY, name=f"{name}_sq")
        m.addConstr(sq.sum() >= 1, name=f"{name}_sq_or")
        m.addConstr(x_o - x_w >= delta_sep - relax - big_m * (1 - sq[0]), name=f"{name}_sq_xpos")
        m.addConstr(x_w - x_o >= delta_sep - relax - big_m * (1 - sq[1]), name=f"{name}_sq_xneg")
        m.addConstr(y_o - y_w >= delta_sep - relax - big_m * (1 - sq[2]), name=f"{name}_sq_ypos")
        m.addConstr(y_w - y_o >= delta_sep - relax - big_m * (1 - sq[3]), name=f"{name}_sq_yneg")
        # Diagonal faces of the octagon circumscribing the delta_sep circle:
        # |(x_o-x_w) +/- (y_o-y_w)| >= delta_sep*sqrt(2).
        d2 = delta_sep * math.sqrt(2)
        ddx, ddy = x_o - x_w, y_o - y_w
        m.addConstr( ddx + ddy >= d2 - relax - big_m * (1 - sq[4]), name=f"{name}_oct_pp")
        m.addConstr(-ddx - ddy >= d2 - relax - big_m * (1 - sq[5]), name=f"{name}_oct_nn")
        m.addConstr( ddx - ddy >= d2 - relax - big_m * (1 - sq[6]), name=f"{name}_oct_pn")
        m.addConstr(-ddx + ddy >= d2 - relax - big_m * (1 - sq[7]), name=f"{name}_oct_np")
    else:
        raise ValueError(f"unknown endcap shape: {endcap}")
