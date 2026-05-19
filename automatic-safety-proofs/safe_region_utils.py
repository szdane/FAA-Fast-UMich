import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import *
from typing import Tuple, List, Dict, Set
import math

from mathematica_utils import *
from plotting_utils import *

init_printing(use_unicode=True)

# global debug if printing
PRINTS = False


def is_infinite_solution_set(soln):
    # Treat "everywhere true" / non-finite sets as no discrete transitions
    return (
        soln in (S.Complexes, S.Reals, S.UniversalSet)
        or isinstance(soln, ConditionSet)
        or (hasattr(soln, "is_FiniteSet") and soln.is_FiniteSet is False)
    )

def compute_polygon_angles(poly: sympy.Polygon) -> list:
    """Return angles corresponding to each vertex of a polygon.
    Also returns vertex pairs corresponding to each side.

    Args:
        poly (sympy.Polygon)

    Returns:
        list: polygon angles, clipped to be positive and under 2pi
        list: pairs of vertices corresponding to each polygon side
    """
    # Go around the polygon and compute the angles (relative to horiz axis)
    # of each of its sides
    verts = poly.vertices
    # Stitch together adjacent vertex pairs (wrap around end of list)
    vertex_pairs = list(zip(verts, verts[1:] + verts[:1]))
    # Angles using atan2: always use atan2 to give correct outputs
    vertex_offsets: List[Tuple] = [
        (nextp.x - p.x, nextp.y - p.y) for (p, nextp) in vertex_pairs
    ]
    angles: List[float] = [
        atan2(offset_y, offset_x) for (offset_x, offset_y) in vertex_offsets
    ]

    # Restrict angles to [0, 2pi]
    positive_angles = [angle % (2 * pi) for angle in angles]
    # assert max(positive_angles) < 2 * pi
    # assert min(positive_angles) >= 0
    return positive_angles, vertex_pairs


def compute_corners_to_angles(poly: sympy.Polygon) -> Dict:
    verts = poly.vertices
    # This section maps each vertex to a range of angles, so we can find
    # the active corners for each of the angles in midpoint_angles
    vertex_pairs_right = list(zip(verts, verts[1:] + verts[:1]))
    vertex_offsets_right: List[Tuple] = [
        (nextp.x - p.x, nextp.y - p.y) for (p, nextp) in vertex_pairs_right
    ]
    angles_right: List[float] = [
        atan2(offset_y, offset_x) for (offset_x, offset_y) in vertex_offsets_right
    ]
    positive_angles_right = [angle % (2 * pi) for angle in angles_right]

    vertex_pairs_left = list(zip(verts[-1:] + verts[:-1], verts))
    vertex_offsets_left: List[Tuple] = [
        (nextp.x - p.x, nextp.y - p.y) for (p, nextp) in vertex_pairs_left
    ]
    angles_left: List[float] = [
        atan2(offset_y, offset_x) for (offset_x, offset_y) in vertex_offsets_left
    ]
    positive_angles_left = [angle % (2 * pi) for angle in angles_left]

    # construct a dict mapping vertices to ranges of angles
    corners_to_angles = dict()
    for i in range(len(verts)):
        if positive_angles_right[i] < positive_angles_left[i]:
            corners_to_angles[verts[i]] = Interval(
                positive_angles_left[i], positive_angles_right[i] + 2 * pi
            )
        else:
            corners_to_angles[verts[i]] = Interval(
                positive_angles_left[i], positive_angles_right[i]
            )
    return corners_to_angles


def ray_method(poly: sympy.Polygon, location: sympy.Point, intruder: sympy.Point):
    shifted_intruder = Point(intruder) - Point(location)
    intersections = poly.intersection(
        Segment2D(
            shifted_intruder,
            Point(shifted_intruder[0] + poly.perimeter, shifted_intruder[1]),
        )
    )
    return len(intersections) % 2 == 0


def encloses_method(poly: sympy.Polygon, location: sympy.Point, intruder: sympy.Point):
    """Check whether a polygon at some location contains an intruder, either on its
    boundary or its interior.

    Args:
        poly (sympy.Polygon): Polygon to check
        location (sympy.Point): Center location of polygon
        intruder (sympy.Point): Intruder point

    Returns:
        bool: True if point is SAFE, False if point is unsafe (inside or on polygon)
    """
    shifted_intruder: Point = Point(intruder) - Point(location)
    if shifted_intruder in poly.vertices or any(
        shifted_intruder in s for s in poly.sides
    ):
        return False
    return not poly.encloses_point(shifted_intruder)


def eval_slope(traj, point, x, y):
    """Compute the slope of a trajectory *expression* and plug in an (x,y) point"""
    df_dy, df_dx = slope_sym(traj, x, y)
    return df_dy.subs(x, point[0]).subs(y, point[1]), df_dx.subs(x, point[0]).subs(
        y, point[1]
    )


def slope_sym(traj, x, y):
    """Returns symbolic slope (dy/dx) for a function of form f(x,y) = 0.
    Negative sign on df/dx required to make results correct for f(x,y) = 0.
    Consider y=sin(x) -> dy/dx is clearly cos(x). In f(x,y) form, we have
    y - sin(x). df/dx would have the wrong sign, so we need the negative sign.

    Args:
        traj (sympy expression): function f(x,y) = 0 of *two* variables,
            representing a trajectory
        x: Sympy symbolic variable
        y: Sympy symbolic variable

    Returns:
        df_dx: derivative of f w.r.t. x. includes negative sign.
        df_dy: derivative of f w.r.t. y.
    """
    return -1 * diff(traj, x), diff(traj, y)


def find_transitions(trajectory, angles, x, y, domain=S.Complexes):
    """Identifies transition points of a trajectory (points where trajectory is
    parallel to any of the angles). Trajectory MUST be an expression (not equation)
    and have two variables. When used in compute_unsafe_cond() with single-var
    trajectories, the function passes -y + f(x) or -x + f(y) to find_transitions().

    Args:
        trajectory (Sympy expression): two-variable expression representing trajectory as f(x,y)
        angles (list[float]): List of angles (corresponding to sides of a)
        x: Sympy symbolic variable
        y: Sympy symbolic variable
        domain (Interval, optional): Sympy interval over which to find transitions. Defaults to S.Complexes.

    Returns:
        dict: Dict mapping angles -> transition points
        set: Set of all transition points
    """
    transitions: set = {}
    # Compute slope symbolically
    df_dy, df_dx = slope_sym(trajectory, x, y)
    for angle in angles:
        # 2 vectors <x1, y1> <x2, y2>
        # parallel iff y1*x2 = x1*y2
        # use vectors <dfdx, dfdy> <cos(theta), sin(theta)>
        soln = solveset(Eq(df_dx * sin(angle), df_dy * cos(angle)), x, domain=domain)
        if is_infinite_solution_set(soln):
            # print("here", soln)
            # slope is parallel for the entire domain at this angle; skip
            continue
        print(f"solveset solution: {soln.doit()}")
        if soln is S.EmptySet:
            soln = solveset(
                Eq(df_dx * sin(angle), df_dy * cos(angle)), y, domain=domain
            )
            if is_infinite_solution_set(soln):
                # print("here", soln)
                continue

            if soln != S.EmptySet and type(soln) != list:
                # In this case, type(soln): S.FiniteSet
                soln = [{y: soln_elem} for soln_elem in list(soln)]
        else:
            # Pack into list of dict so it's clear which variable has been solved for
            print(f"solution when finding transitions for angle {angle}: {soln}")
            if type(soln) is FiniteSet:
                soln = [{x: soln_elem} for soln_elem in list(soln)]
            elif type(soln) is Complement:
                # discard rhs of complmement (shows up in Adler examples for circle path)
                soln = [{x: soln_elem} for soln_elem in list(soln.args[0])]
            else:
                soln = [{x: soln}]

        # TODO: figure out cardinality of this set
        if type(soln) is list:
            for elem in soln:
                if angle in transitions:
                    transitions[angle].append(elem)
                else:
                    transitions[angle] = [elem]
        else:
            if soln is not S.EmptySet:
                if angle in transitions:
                    transitions[angle].append(soln)
                else:
                    transitions[angle] = [soln]

    # soln above may still be symbolic, so we have to evaluate the expression
    # that's what happens below

    transition_points = {}
    set_of_transitions = set()
    traj_eqn = Eq(trajectory, 0)
    for angle, solns in transitions.items():
        for pair in solns:
            print(f"pair used for transition point finding: {pair}")
            # pair should always be a dictionary
            assert type(pair) == dict, "Solution element was not a dictionary!"
            # pair looks like {x: f(y)} or {y: f(x)}
            # remove one variable from equation by substituting pair into traj_eqn
            traj_eqn_single_var = traj_eqn.subs(pair)

            # before going further, figure out the variable for
            # which pair contains a solution
            soln_var = [k for k in pair][0]  # variable is the dict key
            other_var = y if soln_var == x else x

            # traj_eqn used to have two variables but now has only one
            # TODO: not true with symbolic
            single_var_solns = solve(traj_eqn_single_var, other_var)

            for single_var_soln in single_var_solns:
                # substitute in single_var_soln to solve for soln_var
                solved_eqn = Eq(soln_var, pair[soln_var]).subs(
                    other_var, single_var_soln
                )
                # with this, we have a solution for the transition point
                if soln_var == x:
                    if PRINTS:
                        print("x-coord:", solved_eqn.rhs)
                        print("y-coord:", single_var_soln)
                    try:
                        transition_point = Point(solved_eqn.rhs, single_var_soln)
                    except: 
                        print(solved_eqn)
                        print(single_var_solns)
                        print(single_var_soln)
                        print(pair)
                elif soln_var == y:
                    transition_point = Point(single_var_soln, solved_eqn.rhs)
                set_of_transitions.add(transition_point)
                if angle in transition_points:
                    transition_points[angle].append(transition_point)
                else:
                    transition_points[angle] = [transition_point]

    return transition_points, set_of_transitions


def compute_unsafe_cond(
    x,
    y,
    poly: sympy.Polygon,
    trajectory,  # piecewise
    domain,
    add_notches=True,
    print_latex=False,
):
    """Given a trajectory, polygon, and domain, computes a boolean formulation of
    the *unsafe* region. Simply negate to get the formulation of the safe region.

    Args:
        x (): Sympy symbolic variable
        y ([type]): Sympy symbolic variable
        poly (sympy.Polygon): Polygon to use for safe/unsafe region construction. In this
            release of software, polygon must be symmetric.
        trajectory: Single-variable Sympy *expression* for trajectory. May be Piecewise.
        domain: Interval domain over which to compute unsafe region.
        add_notches (bool, optional): Whether or not to add notches for region formulation.
            Defaults to True.
        print_latex (bool, optional): Whether or not to print Latex output for boolean
            formulation of the unsafe region. Defaults to False.

    Raises:
        Exception: Complains if trajectory has two variables.

    Returns:
        Sympy boolean formula for *unsafe region* over domain above.
    """
    angles, vertex_pairs = compute_polygon_angles(poly)
    verts = poly.vertices
    if y not in trajectory.free_symbols:
        func_var = x
        keyfunc = lambda p: p.x
    elif x not in trajectory.free_symbols:
        func_var = y
        keyfunc = lambda p: p.y
    else:
        raise Exception("Trajectory had two variables!")

    # compute width to use in g() function later
    # for functions f(y), this is actually height
    w_point = max(
        [
            -1 * min([v - poly.centroid for v in verts], key=keyfunc),
            max([v - poly.centroid for v in verts], key=keyfunc),
        ],
        key=keyfunc,
    )
    w = getattr(w_point, str(func_var))

    # construct a large set of transitions
    set_of_transitions = set()
    notch_points = set()
    if type(trajectory) == Piecewise:
        # For piecewise trajectories, we need to find transitions for each piece
        for subtraj, subcond in trajectory.as_expr_set_pairs():
            # trim domain by computing intersection, so we don't find transitions
            # outside of the domain for each piece of the trajectory
            subdomain = subcond.intersect(domain)

            if y not in subtraj.free_symbols:  # form y=f(x)
                _, subset_of_transitions = find_transitions(
                    -y + subtraj, angles, x, y, domain=subdomain
                )
                # add piecewise boundary
                left_bound = Point(subdomain.inf, subtraj.subs(func_var, subdomain.inf))
                right_bound = Point(
                    subdomain.sup, subtraj.subs(func_var, subdomain.sup)
                )
            elif x not in subtraj.free_symbols:  # form x=f(y)
                _, subset_of_transitions = find_transitions(
                    -x + subtraj, angles, x, y, domain=subdomain
                )
                # add piecewise boundary
                left_bound = Point(subtraj.subs(func_var, subdomain.inf), subdomain.inf)
                right_bound = Point(
                    subtraj.subs(func_var, subdomain.sup), subdomain.sup
                )
            set_of_transitions.update(subset_of_transitions)
            notch_points.update(subset_of_transitions)

            if left_bound.x.is_finite and left_bound.y.is_finite:
                set_of_transitions.add(left_bound)
            if right_bound.x.is_finite and right_bound.y.is_finite:
                set_of_transitions.add(right_bound)
    else:
        if y not in trajectory.free_symbols:
            _, subset_of_transitions = find_transitions(
                -y + trajectory, angles, x, y, domain=domain
            )
            # Add left and right boundaries to check for notch there too
            left_bound = Point(domain.inf, trajectory.subs(func_var, domain.inf))
            right_bound = Point(domain.sup, trajectory.subs(func_var, domain.sup))
        elif x not in trajectory.free_symbols:
            _, subset_of_transitions = find_transitions(
                -x + trajectory, angles, x, y, domain=domain
            )
            # Add left and right boundaries to check for notch there too
            left_bound = Point(trajectory.subs(func_var, domain.inf), domain.inf)
            right_bound = Point(trajectory.subs(func_var, domain.sup), domain.sup)

        set_of_transitions.update(subset_of_transitions)
        if left_bound.x.is_finite and left_bound.y.is_finite:
            set_of_transitions.add(left_bound)
        if right_bound.x.is_finite and right_bound.y.is_finite:
            set_of_transitions.add(right_bound)
        # set_of_transitions.add(left_bound)
        # set_of_transitions.add(right_bound)

    # In order to identify which corners are active over which intervals,
    # sort transitions and identify active corners at the midpoints of the intervals
    # defined by transition points.
    sorted_transitions: list = sorted(
        set_of_transitions, key=lambda point: getattr(point, str(func_var))
    )
    if PRINTS:
        print(sorted_transitions)
    func_var_transitions = [getattr(p, str(func_var)) for p in sorted_transitions]
    midpoints = np.convolve(func_var_transitions, [1, 1], "valid") / 2
    if func_var == x:
        deriv_traj = diff(trajectory, x)
    elif func_var == y:
        deriv_traj = 1 / diff(trajectory, y)  # invert dx/dy to get slope dy/dx
    dydx_midpoints = [deriv_traj.subs(func_var, val) for val in midpoints]
    # find the trajectory tangent angle at each midpoint
    midpoint_angles = [atan2(d, 1) for d in dydx_midpoints]

    corners_to_angles = compute_corners_to_angles(poly)

    # Map each midpoint angle to the active corner. Because we're only supporting
    # symmetric polygons so far, we only need to find a single active corner here,
    # and we can use its opposing corner (due to central symmetry) later.
    active_corners: dict = {}
    for midpoint_angle in midpoint_angles:
        for k, v in corners_to_angles.items():
            if PRINTS:
                print(midpoint_angle)
                print(v)
            # NOTE: fails for symbolic trajectory parameters
            if midpoint_angle % (2 * pi) in v:
                active_corners[midpoint_angle] = k
    var_intervals = list(
        zip(func_var_transitions[:-1], func_var_transitions[1:])
    )  # same len as midpoints

    # Construct the full boolean formulation of the *unsafe* region
    total_cond = None
    for i, var_interval in enumerate(var_intervals):
        if func_var == x:
            x_left = var_interval[0]
            y_left = trajectory.subs(x, x_left)
            x_right = var_interval[1]
            y_right = trajectory.subs(x, x_right)
        elif func_var == y:
            y_left = var_interval[0]
            x_left = trajectory.subs(y, y_left)
            y_right = var_interval[1]
            x_right = trajectory.subs(y, y_right)

        poly_center_left = Point(x_left, y_left)
        poly_center_right = Point(x_right, y_right)

        # Assume symmetric polygon.
        active_corner_offset = active_corners[midpoint_angles[i]]
        line_left = Line(poly_center_left, poly_center_left + active_corner_offset)
        line_right = Line(poly_center_right, poly_center_right + active_corner_offset)
        print(line_right)
        # using Line.equation() creates duplicate x,y variables and ruins .subs() call later
        # Ensure this check only applies between the active corners at the start and end of the interval
        left_a, left_b, left_c = line_left.coefficients
        right_a, right_b, right_c = line_right.coefficients
        left_eq = left_a * x + left_b * y + left_c
        right_eq = right_a * x + right_b * y + (0 if math.isnan(float(right_c)) else right_c)
        # right_eq = right_a *x + right_b * y + right_c
        print(right_a, x, right_b, y, right_c)
        # print(right_eq, left_eq)
        corner_cond = left_eq * right_eq <= 0

        # Ensure this check only applies between the transition points (plus/minus width)
        if func_var == x:
            func_var_cond = (x >= (x_left - w)) & (x <= (x_right + w))
        elif func_var == y:
            # note that w is actually height when found above
            func_var_cond = (y >= (y_left - w)) & (y <= (y_right + w))

        # construct g functions
        # same trajectory over this interval, held constant outside of it
        if type(trajectory) == Piecewise:
            if func_var == x:
                # Use an open interval in case two cases hold exactly at x_left
                current_piece = trajectory.as_expr_set_pairs(
                    Interval.open(x_left, x_right)
                )
                if len(current_piece) > 1:
                    print(
                        f"Warning! more than one piecewise segment detected over interval ({y_left}, {y_right}). Results may be erroneous due to mis-specified piecewise functions."
                    )
                try:
                    current_fn = current_piece[0][0]
                except:
                    print("current_piece", current_piece)
                g = y - Piecewise(
                    (y_left, x < x_left),
                    (current_fn, x <= x_right),
                    (y_right, x > x_right),
                )
            elif func_var == y:
                # Use an open interval in case two cases hold exactly at y_left
                current_piece = trajectory.as_expr_set_pairs(
                    Interval.open(y_left, y_right)
                )
                if len(current_piece) > 1:
                    print(
                        f"Warning! more than one piecewise segment detected over interval ({y_left}, {y_right}). Results may be erroneous due to mis-specified piecewise functions."
                    )
                current_fn = current_piece[0][0]
                g = x - Piecewise(
                    (x_left, y < y_left),
                    (current_fn, y <= y_right),
                    (x_right, y > y_right),
                )
        else:
            if func_var == x:
                g = y - Piecewise(
                    (y_left, x < x_left),
                    (trajectory, x <= x_right),
                    (y_right, x > x_right),
                )
            elif func_var == y:
                g = x - Piecewise(
                    (x_left, y < y_left),
                    (trajectory, y <= y_right),
                    (x_right, y > y_right),
                )
        # Assume symmetric polygon
        g1 = g.subs(x, x - active_corner_offset.x).subs(y, y - active_corner_offset.y)
        g2 = g.subs(x, x + active_corner_offset.x).subs(y, y + active_corner_offset.y)

        full_cond = corner_cond & func_var_cond & (g1 * g2 <= 0)
        if total_cond is None:
            total_cond = full_cond
        else:
            total_cond = total_cond | full_cond

    if add_notches:
        # adding notches
        for transition_point in notch_points:
            # neg for left side, 0 for on edge, pos for on right side
            # inside polygon IF all neg or IF all pos
            shifted_vertex_pairs = [
                (v + transition_point, nextv + transition_point)
                for (v, nextv) in vertex_pairs
            ]  # [(v1, v2), (v2, v3), ..., (vn, v1)]
            # source: https://inginious.org/course/competitive-programming/geometry-pointinconvex
            # source: https://www.eecs.umich.edu/courses/eecs380/HANDOUTS/PROJ2/InsidePoly.html
            side_conds = [
                (y - v.y) * (nextv.x - v.x) - (x - v.x) * (nextv.y - v.y)
                for (v, nextv) in shifted_vertex_pairs
            ]
            # construct both cases inside
            cond1 = true  # init to sympy.true since we're cascading Ands
            cond2 = true  # init to sympy.true since we're cascading Ands
            for side_cond in side_conds:
                cond1 = cond1 & (side_cond <= 0)
                cond2 = cond2 & (side_cond >= 0)
            total_cond = total_cond | cond1 | cond2

    if print_latex:
        print(latex(total_cond))

    return total_cond
