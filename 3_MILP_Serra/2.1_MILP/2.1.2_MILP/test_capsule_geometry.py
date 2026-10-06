"""
Verification test for the active-corner capsule separation constraint, used
by milp_multiple_Debug.py. Checks BOTH implementations:

  - capsule_geometry.py            (vector dot/cross-product form, production)
  - capsule_geometry_slope.py      (scalar-slope form, an independent
                                     alternate implementation)

For each of the 8 segment directions (plus a stationary segment), this pins
a segment (x_near,y_near)->(x_far,y_far) and sweeps a grid of probe points
(x_o,y_o) around it. For every probe, for each implementation, it:

  1. Computes a REFERENCE verdict independently, via the textbook point-to-
     line-segment distance formula (no relation to either implementation).
  2. Builds a tiny Gurobi model that calls the implementation's actual
     add_capsule_direction*() function, with the obstacle pinned to that
     probe point, and reads off whether the model is feasible (the
     constraint agrees the probe is safe) or infeasible (it doesn't).
  3. Compares the two. Probes within a small tolerance band of exactly
     delta_sep are skipped (ambiguous boundary, not a meaningful check).

Run: python3 test_capsule_geometry.py
"""
import math
from gurobipy import Model, GRB, Env

from capsule_geometry import build_seg_geom, add_capsule_direction
from capsule_geometry_slope import build_seg_geom_slope, add_capsule_direction_slope

DELTA = 10.0          # delta_sep for the test, in abstract units
SEG_LEN = 3.0 * DELTA  # segment length, in abstract units
BIG_M = 1e5
TOL = 0.05 * DELTA    # skip probes within this band of the exact boundary

DIRECTIONS = {
    "up-right":   (1, 1),
    "up-left":    (-1, 1),
    "down-right": (1, -1),
    "down-left":  (-1, -1),
    "up":         (0, 1),
    "down":       (0, -1),
    "right":      (1, 0),
    "left":       (-1, 0),
}


def point_segment_distance(px, py, ax, ay, bx, by):
    """Reference implementation, independent of both capsule_geometry
    modules."""
    abx, aby = bx - ax, by - ay
    seg_len2 = abx * abx + aby * aby
    if seg_len2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / seg_len2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def make_probes(dx, dy):
    """Grid of (x_o,y_o) probes in the segment's own (along,perp) frame,
    from well before the near point to well past the far point, and from
    well inside the strip to well outside it on both sides."""
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-9:
        along = (0.0, 0.0)
        perp = (1.0, 0.0)
    else:
        along = (dx / seg_len, dy / seg_len)
        perp = (-along[1], along[0])

    probes = []
    s_vals = [k * 0.5 * DELTA for k in range(-3, 9)]   # -1.5*DELTA .. 4.0*DELTA
    t_vals = [k * 0.5 * DELTA for k in range(-4, 5)]   # -2.0*DELTA .. 2.0*DELTA
    for s in s_vals:
        for t in t_vals:
            x_o = s * along[0] + t * perp[0]
            y_o = s * along[1] + t * perp[1]
            probes.append((x_o, y_o))
    return probes


IMPLEMENTATIONS = {
    "vector (capsule_geometry.py)": (build_seg_geom, add_capsule_direction),
    "slope (capsule_geometry_slope.py)": (build_seg_geom_slope, add_capsule_direction_slope),
}


def run_case(env, build_geom_fn, add_direction_fn,
             x_near_v, y_near_v, x_far_v, y_far_v, x_o_v, y_o_v):
    m = Model(env=env)
    m.Params.NonConvex = 2
    m.Params.DualReductions = 0

    x_near = m.addVar(lb=x_near_v, ub=x_near_v, name="x_near")
    y_near = m.addVar(lb=y_near_v, ub=y_near_v, name="y_near")
    x_far = m.addVar(lb=x_far_v, ub=x_far_v, name="x_far")
    y_far = m.addVar(lb=y_far_v, ub=y_far_v, name="y_far")
    x_o = m.addVar(lb=x_o_v, ub=x_o_v, name="x_o")
    y_o = m.addVar(lb=y_o_v, ub=y_o_v, name="y_o")

    diffx1 = m.addVar(lb=-GRB.INFINITY, name="diffx1")
    diffy1 = m.addVar(lb=-GRB.INFINITY, name="diffy1")
    m.addConstr(diffx1 == x_far - x_near)
    m.addConstr(diffy1 == y_far - y_near)

    geom = build_geom_fn(m, diffx1, diffy1, tag="t")

    bin_var = m.addVar(lb=1, ub=1, vtype=GRB.BINARY, name="bin_var")  # always this branch
    add_direction_fn(
        m, geom, x_far, y_far, x_near, y_near, x_o, y_o,
        bin_var, relax_common=0.0, delta_sep=DELTA, tag="t", big_m=BIG_M)

    m.setObjective(0.0)
    m.Params.OutputFlag = 0
    m.optimize()
    return m.status == GRB.OPTIMAL


def main():
    env = Env(params={"OutputFlag": 0})

    cases = dict(DIRECTIONS)
    cases["stationary"] = (0, 0)

    overall_pass = True

    for impl_name, (build_geom_fn, add_direction_fn) in IMPLEMENTATIONS.items():
        total = 0
        skipped = 0
        mismatches = []

        for name, (ux, uy) in cases.items():
            if name == "stationary":
                dx, dy = 0.0, 0.0
            else:
                norm = math.hypot(ux, uy)
                dx, dy = ux / norm * SEG_LEN, uy / norm * SEG_LEN
            x_near_v, y_near_v = 0.0, 0.0
            x_far_v, y_far_v = dx, dy

            for x_o_v, y_o_v in make_probes(dx, dy):
                dist = point_segment_distance(x_o_v, y_o_v, x_near_v, y_near_v, x_far_v, y_far_v)
                if abs(dist - DELTA) < TOL:
                    skipped += 1
                    continue
                expected_safe = dist > DELTA

                got_safe = run_case(env, build_geom_fn, add_direction_fn,
                                     x_near_v, y_near_v, x_far_v, y_far_v, x_o_v, y_o_v)
                total += 1
                if got_safe != expected_safe:
                    mismatches.append((name, x_o_v, y_o_v, dist, expected_safe, got_safe))

        print(f"=== {impl_name} ===")
        print(f"Ran {total} probes ({skipped} skipped near the delta_sep boundary).")
        if mismatches:
            overall_pass = False
            print(f"{len(mismatches)} MISMATCH(ES):")
            print(f"{'direction':<12}{'x_o':>8}{'y_o':>8}{'dist':>8}  {'expected':>8}  {'model':>8}")
            for name, x_o_v, y_o_v, dist, expected_safe, got_safe in mismatches:
                print(f"{name:<12}{x_o_v:>8.2f}{y_o_v:>8.2f}{dist:>8.2f}  "
                      f"{'safe' if expected_safe else 'UNSAFE':>8}  "
                      f"{'safe' if got_safe else 'UNSAFE':>8}")
        else:
            print("All probes agree with the reference point-to-segment-distance test. PASS.")
        print()

    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
