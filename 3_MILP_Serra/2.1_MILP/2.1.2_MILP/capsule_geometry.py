"""
Active-corner capsule separation geometry.

Shared by milp_multiple_Debug.py and its verification tests, so both use the
exact same constraint-building code (no risk of the test drifting from what
production actually solves).

For an aircraft's segment from waypoint "near" (k-1) to waypoint "far" (k),
and another aircraft's position (x_o,y_o) at the same time index, the
obstacle is outside the delta_sep-radius capsule swept along the segment
iff it clears the segment via at least one of:
  (3) or (4): perpendicular distance to the segment's line is >= delta_sep
              on one side or the other (each alone is sufficient, since
              distance-to-segment >= distance-to-line always)
  (1) AND (2): past the far endpoint's along-track boundary AND outside
               the far endpoint's circle (the far end-cap region)
  (5) AND (6): before the near endpoint's along-track boundary AND outside
               the near endpoint's circle (the near end-cap region)

(1),(3),(4),(6) are all written as vector dot/cross products of the segment
direction (dx,dy) with the offset to the obstacle. Because dx,dy are the
actual signed segment-direction values, these formulas are already
direction-consistent for any segment orientation - unlike the scalar-slope
form in the original derivation notes (y = m*x + b), which is sign-ambiguous
and needs a per-direction case table to pick the right sense, these vector
forms need no case table at all.
"""
from gurobipy import GRB


def build_seg_geom(m, diffx1, diffy1, tag, eps=1e-3):
    """Derive segment length and a "does this segment actually move" flag
    for (diffx1, diffy1), returning the geom dict consumed by
    add_capsule_direction.

    eps must be safely larger than Gurobi's default FeasibilityTol (1e-6):
    an indicator like "moves=1 => L >= eps" is only reliable if a violation
    of size eps is clearly outside the solver's feasibility tolerance. At
    eps=1e-6 (matching FeasibilityTol exactly) Gurobi can and does treat
    L=0 as satisfying L>=1e-6, silently defeating the zero test.
    """
    seglen = m.addVar(name=f"seglen_{tag}")
    m.addConstr(seglen * seglen == diffx1 * diffx1 + diffy1 * diffy1, name=f"seglen_def_{tag}")

    moves = m.addVar(vtype=GRB.BINARY, name=f"moves_{tag}")
    m.addGenConstrIndicator(moves, 1, seglen, GRB.GREATER_EQUAL, eps, name=f"moves_on_{tag}")
    m.addGenConstrIndicator(moves, 0, seglen, GRB.LESS_EQUAL, eps, name=f"moves_off_{tag}")

    return dict(dx=diffx1, dy=diffy1, L=seglen, moves=moves)


def add_capsule_direction(m, geom, x_far, y_far, x_near, y_near, x_o, y_o,
                           bin_var, relax_common, delta_sep, tag, big_m):
    """Require (x_o,y_o) to stay outside the delta_sep capsule swept from
    (x_near,y_near) to (x_far,y_far), active only when bin_var selects this
    OR-branch (of the outer i-segment / j-segment / vertical disjunction)."""
    dx, dy, L = geom['dx'], geom['dy'], geom['L']
    moves = geom['moves']

    branch_relax = big_m * (1 - bin_var) + relax_common

    case_vars = m.addVars(4, vtype=GRB.BINARY, name=f"capcase_{tag}")
    case_A, case_B, case_C, case_D = case_vars[0], case_vars[1], case_vars[2], case_vars[3]
    m.addConstr(case_A + case_B + case_C + case_D >= 1, name=f"capcase_or_{tag}")
    # The strip-side cases are vacuously satisfied when the segment hasn't
    # moved (perp == 0 == delta_sep*L), which would let the solver "cheat"
    # past real separation checking - exclude them when stationary and let
    # the cap-region cases (which still enforce a real circle check) carry it.
    m.addConstr(case_A + case_B <= moves, name=f"capcase_moves_{tag}")

    # dot((dx,dy), obstacle - far):  >=0 iff obstacle is past the far
    # endpoint's along-track boundary, in the direction of travel.
    expr1 = dx * (x_o - x_far) + dy * (y_o - y_far)
    # dot((dx,dy), obstacle - near): <=0 iff obstacle is before the near
    # endpoint's along-track boundary.
    expr6 = dx * (x_o - x_near) + dy * (y_o - y_near)
    # cross((dx,dy), obstacle - near): the perpendicular offset (scaled by
    # L); >= delta_sep*L or <= -delta_sep*L means clear of the strip on one
    # side or the other.
    perp = dx * (y_o - y_near) - dy * (x_o - x_near)

    m.addConstr(perp >= delta_sep * L - branch_relax - big_m * (1 - case_A),
                name=f"cap3_{tag}")
    m.addConstr(perp <= -delta_sep * L + branch_relax + big_m * (1 - case_B),
                name=f"cap4_{tag}")

    m.addConstr(expr1 >= 0 - branch_relax - big_m * (1 - case_C),
                name=f"cap1_{tag}")
    m.addConstr(
        (x_far - x_o) * (x_far - x_o) + (y_far - y_o) * (y_far - y_o) >=
        delta_sep * delta_sep - branch_relax - big_m * (1 - case_C),
        name=f"cap2_{tag}")

    m.addConstr(expr6 <= 0 + branch_relax + big_m * (1 - case_D),
                name=f"cap6_{tag}")
    m.addConstr(
        (x_near - x_o) * (x_near - x_o) + (y_near - y_o) * (y_near - y_o) >=
        delta_sep * delta_sep - branch_relax - big_m * (1 - case_D),
        name=f"cap5_{tag}")
