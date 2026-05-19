from enum import unique
from tkinter import Y
from setuptools import Command
from sympy import *
import itertools
import time
from typing import Tuple, List, Dict, Set
from safe_region_objects import *

from safe_region_utils import compute_polygon_angles, slope_sym

# global debug if printing
PRINTS = False


def find_transitions_symbolic(trajectory, angles, x, y, domain=S.Complexes):
    """Identifies transition points of a trajectory (points where trajectory is
    parallel to any of the angles). Trajectory MUST be an expression (not equation)
    and have two variables. When used in compute_unsafe_cond() with single-var
    trajectories, the function passes -y + f(x) or -x + f(y) to find_transitions().

    Args:
        trajectory (Sympy expression): two-variable expression representing trajectory as f(x,y). Never Piecewise.
        angles (list[float]): List of angles (corresponding to sides of a)
        x: Sympy symbolic variable
        y: Sympy symbolic variable
        domain (Interval, optional): Sympy interval over which to find transitions. Defaults to S.Complexes.

    Returns:
        set: Set of all TransitionPoint objects
    """
    transitions: set = {}
    # Compute slope symbolically
    df_dy, df_dx = slope_sym(trajectory, x, y)
    for angle in angles:
        # 2 vectors <x1, y1> <x2, y2>
        # parallel iff y1*x2 = x1*y2
        # use vectors <dfdx, dfdy> <cos(theta), sin(theta)>
        # print(angle)
        # print(df_dx * sin(angle))
        # print(df_dy * cos(angle), "\n")
        try:
            soln = solveset(
                Eq(df_dx * sin(angle), df_dy * cos(angle)), x, domain=domain
            )
        except:
            soln = solve(Eq(df_dx * sin(angle), df_dy * cos(angle)), x)
        # print(f"solveset solution: {soln.doit()}")b
        if soln is S.EmptySet:
            soln = solveset(
                Eq(df_dx * sin(angle), df_dy * cos(angle)), y, domain=domain
            )
            if soln != S.EmptySet and type(soln) != list:
                # In this case, type(soln): S.FiniteSet
                soln = [{y: soln_elem} for soln_elem in list(soln)]
        else:
            # Pack into list of dict so it's clear which variable has been solved for
            # print(f"solution when finding transitions for angle {angle}: {soln}")
            if type(soln) is FiniteSet:
                soln = [{x: soln_elem} for soln_elem in list(soln)]
            elif type(soln) is Complement:
                # discard rhs of complmement (shows up in Adler examples for circle path)
                soln = [{x: soln_elem} for soln_elem in list(soln.args[0])]
            elif type(soln) is list:
                soln = [{x: soln_elem} for soln_elem in soln]
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

    set_of_transitions = set()
    traj_eqn = Eq(trajectory, 0)
    for angle, solns in transitions.items():
        for pair in solns:
            # print(f"pair used for transition point finding: {pair}")
            # pair should always be a dictionary
            assert type(pair) == dict, "Solution element was not a dictionary!"
            # pair looks like {x: f(y)} or {y: f(x)}
            # before going further, figure out the variable for
            # which pair contains a solution
            soln_var = [k for k in pair][0]  # variable is the dict key
            other_var = y if soln_var == x else x

            # remove one variable from equation by substituting pair into traj_eqn
            if type(pair[soln_var]) == Intersection:
                if Reals in pair[soln_var].args:
                    pair = {soln_var: list(pair[soln_var].args[1])[0]}
                    # print(type(pair[soln_var]))
                    # print(pair)
            traj_eqn_single_var = traj_eqn.subs(pair)

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
                        pass
                        # print("x-coord:", solved_eqn.rhs)
                        # print("y-coord:", single_var_soln)
                    coord = Point(solved_eqn.rhs, single_var_soln)
                elif soln_var == y:
                    coord = Point(single_var_soln, solved_eqn.rhs)

                # Construct TransitionPoint object and add
                set_of_transitions.add(TransitionPoint(coord, angle, traj_eqn))
    # print(set_of_transitions)
    return set_of_transitions


def compute_all_transitions(
    x,
    y,
    poly: Polygon,
    trajectory,  # piecewise
    domain,
    intervals=None,
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
        raise Exception("Trajectory had too many variables!")

    # construct a large set of transitions
    # set_of_transitions = TransitionPointSet()
    set_of_transitions = set()
    unique_transition_points = set()
    lookup_dict = {}
    if type(trajectory) == Piecewise:
        # For piecewise trajectories, we need to find transitions for each piece
        # for (subtraj, subcond) in trajectory.as_expr_set_pairs():
        # as_expr_set_pairs doesn't work for multivariate (symbolic) trajectories
        for i, (subtraj, subcond) in enumerate(trajectory.args):
            # trim domain by computing intersection, so we don't find transitions
            # outside of the domain for each piece of the trajectory
            # TODO fix hardcoding
            # subdomain = subcond.intersect(domain)  # invalid for symbolic
            subdomain = Reals  # for symbolic just leave as Reals??
            # print(f"\nPiecewise: {subtraj} for condition {subcond}")

            if y not in subtraj.free_symbols:  # form y=f(x)
                # TODO: domain? for finding outside of piecewise range
                subset_of_transitions = find_transitions_symbolic(
                    -y + subtraj, angles, x, y, domain=domain
                )
                # add piecewise boundary
                if (
                    intervals[i].inf.is_imaginary
                    or subtraj.subs(func_var, intervals[i].inf).is_imaginary
                ):
                    print(
                        "imaginary!",
                        intervals[i].inf,
                        subtraj.subs(func_var, intervals[i].inf),
                    )
                    left_bound = None
                else:
                    left_bound = Point(
                        intervals[i].inf, subtraj.subs(func_var, intervals[i].inf)
                    )
                if (
                    intervals[i].sup.is_imaginary
                    or subtraj.subs(func_var, intervals[i].sup).is_imaginary
                ):
                    print(
                        "imaginary!",
                        intervals[i].sup,
                        subtraj.subs(func_var, intervals[i].sup),
                    )
                    right_bound = None
                else:
                    right_bound = Point(
                        intervals[i].sup, subtraj.subs(func_var, intervals[i].sup)
                    )
            elif x not in subtraj.free_symbols:  # form x=f(y)
                # TODO: domain? for finding outside of piecewise range
                subset_of_transitions = find_transitions_symbolic(
                    -x + subtraj, angles, x, y, domain=domain
                )
                # add piecewise boundary
                if (
                    subtraj.subs(func_var, intervals[i].inf).is_imaginary
                    or intervals[i].inf.is_imaginary
                ):
                    left_bound = None
                else:
                    left_bound = Point(
                        subtraj.subs(func_var, intervals[i].inf), intervals[i].inf
                    )
                if (
                    subtraj.subs(func_var, intervals[i].sup).is_imaginary
                    or intervals[i].sup.is_imaginary
                ):
                    right_bound = None
                else:
                    right_bound = Point(
                        subtraj.subs(func_var, intervals[i].sup), intervals[i].sup
                    )
            # set_of_transitions.update(subset_of_transitions)
            for elem in subset_of_transitions:
                if elem.point not in unique_transition_points:
                    set_of_transitions.add(elem)
                    unique_transition_points.add(elem.point)
                point_var = getattr(elem.point, str(func_var))
                if point_var in lookup_dict:
                    lookup_dict[point_var].append(subtraj)
                else:
                    lookup_dict[point_var] = [subtraj]

            # if left_bound.x.is_finite and left_bound.y.is_finite:
            #     set_of_transitions.add(left_bound)
            # if right_bound.x.is_finite and right_bound.y.is_finite:
            #     set_of_transitions.add(right_bound)
            if left_bound:
                if left_bound not in unique_transition_points:
                    set_of_transitions.add(TransitionPoint(left_bound, None, subtraj))
                    unique_transition_points.add(left_bound)
                point_var = getattr(left_bound, str(func_var))
                if point_var in lookup_dict:
                    lookup_dict[point_var].append(subtraj)
                else:
                    lookup_dict[point_var] = [subtraj]

            if right_bound:
                if right_bound not in unique_transition_points:
                    set_of_transitions.add(TransitionPoint(right_bound, None, subtraj))
                    unique_transition_points.add(right_bound)
                point_var = getattr(right_bound, str(func_var))
                if point_var in lookup_dict:
                    lookup_dict[point_var].append(subtraj)
                else:
                    lookup_dict[point_var] = [subtraj]
    # Non-Piecewise case
    else:
        if y not in trajectory.free_symbols:
            subset_of_transitions = find_transitions_symbolic(
                -y + trajectory, angles, x, y, domain=domain
            )
            # Add left and right boundaries to check for notch there too
            left_bound = Point(domain.inf, trajectory.subs(func_var, domain.inf))
            right_bound = Point(domain.sup, trajectory.subs(func_var, domain.sup))
        elif x not in trajectory.free_symbols:
            subset_of_transitions = find_transitions_symbolic(
                -x + trajectory, angles, x, y, domain=domain
            )
            # Add left and right boundaries to check for notch there too
            left_bound = Point(trajectory.subs(func_var, domain.inf), domain.inf)
            right_bound = Point(trajectory.subs(func_var, domain.sup), domain.sup)

        # set_of_transitions.update(subset_of_transitions)
        for elem in subset_of_transitions:
            if elem.point not in unique_transition_points:
                set_of_transitions.add(elem)
                unique_transition_points.add(elem.point)
            point_var = getattr(elem.point, str(func_var))
            if point_var in lookup_dict:
                lookup_dict[point_var].append(trajectory)
            else:
                lookup_dict[point_var] = [trajectory]

        # if left_bound.x.is_finite and left_bound.y.is_finite:
        #     set_of_transitions.add(left_bound)
        # if right_bound.x.is_finite and right_bound.y.is_finite:
        #     set_of_transitions.add(right_bound)
        if left_bound:
            if left_bound not in unique_transition_points:
                set_of_transitions.add(TransitionPoint(left_bound, None, trajectory))
                unique_transition_points.add(left_bound)
            point_var = getattr(left_bound, str(func_var))
            if point_var in lookup_dict:
                lookup_dict[point_var].append(trajectory)
            else:
                lookup_dict[point_var] = [trajectory]

        if right_bound:
            if right_bound not in unique_transition_points:
                set_of_transitions.add(TransitionPoint(right_bound, None, trajectory))
                unique_transition_points.add(right_bound)
            point_var = getattr(right_bound, str(func_var))
            if point_var in lookup_dict:
                lookup_dict[point_var].append(trajectory)
            else:
                lookup_dict[point_var] = [trajectory]

        # set_of_transitions.add(TransitionPoint(left_bound, None, trajectory))
        # set_of_transitions.add(TransitionPoint(right_bound, None, trajectory))
    return set_of_transitions, lookup_dict, func_var


def sort_or_order(transitions: Set[Point2D], lookup_dict: Dict, func_var: Symbol):
    """In order to identify which corners are active over which intervals,
    sort transitions and identify active corners at the midpoints of the intervals
    defined by transition points.


    Args:
        transitions (Set[Point2D]): set of all transition points
        func_var (Symbol): trajectory variable (either x or y)
    """
    # try:
    #     print("sorting was successful!")
    #     sorted_transitions: list = sorted(
    #         transitions,
    #         key=lambda tp: getattr(tp.point, str(func_var)),
    #     )
    #     return [sorted_transitions]
    # except TypeError as error:
    # construct dict mapping x-val (or y-val) to transition object
    var_to_transition: Dict = {}
    for transition in transitions:
        lookup_coord = getattr(transition.point, str(func_var))
        if lookup_coord in var_to_transition:
            var_to_transition[lookup_coord].append(transition)
        else:
            var_to_transition[lookup_coord] = [transition]

    add_neg_inf = False
    add_pos_inf = False

    # include all possible combinations of true Transition points in/out of set
    """TODO!: add docstring to explain because this is complicated"""
    boundary_coords = set(
        map(
            lambda tp: getattr(tp.point, str(func_var)),
            filter(lambda t: t.is_bound, transitions),
        )
    )
    if -oo in boundary_coords:
        add_neg_inf = True
        boundary_coords.remove(-oo)
    if oo in boundary_coords:
        add_pos_inf = True
        boundary_coords.remove(oo)

    true_transition_coords = set(
        map(
            lambda tp: getattr(tp.point, str(func_var)),
            filter(lambda t: not t.is_bound, transitions),
        )
    )
    possible_orderings = []
    num_tested = 0
    for i in range(len(true_transition_coords) + 1):
        for comb in itertools.combinations(true_transition_coords, i):
            orderings = itertools.permutations(boundary_coords | set(comb))
            for ordering in orderings:
                num_tested += 1
                if check_ordering(
                    ordering,
                    var_to_transition,
                    lookup_dict,
                    add_neg_inf,
                    add_pos_inf,
                ):
                    possible_orderings.append(ordering)
                    if len(possible_orderings) % 100 == 0:
                        print(len(possible_orderings))
                    # if num_tested % 100 == 0:
                    #     print(num_tested)

    possible_transitions = []
    for ordering in possible_orderings:
        possible_transition_ordering = reconstruct_transition_points(
            var_to_transition,
            ordering,
            add_neg_inf,
            add_pos_inf,
        )
        # may return empty list if invalid (doesn't start or end with Boundary)
        # TODO!: probably removable
        if possible_transition_ordering:
            possible_transitions.append(possible_transition_ordering)
    return possible_transitions


def check_ordering(
    ordering, var_to_transition, lookup_dict, add_neg_inf, add_pos_inf, verbose=False
):
    # make sure start and end of ordering are boundary points
    if not add_neg_inf and not var_to_transition[ordering[0]][0].is_bound:
        return False

    if not add_pos_inf and not var_to_transition[ordering[-1]][0].is_bound:
        return False

    if add_neg_inf:
        left_funcs = set(lookup_dict[-oo])
        right_funcs = set(lookup_dict[ordering[0]])
        common_funcs = left_funcs.intersection(right_funcs)
        if len(common_funcs) == 0:
            return False

    if add_pos_inf:
        left_funcs = set(lookup_dict[ordering[-1]])
        right_funcs = set(lookup_dict[oo])
        common_funcs = left_funcs.intersection(right_funcs)
        if len(common_funcs) == 0:
            return False

    # TODO: make this more efficient (nlogn somehow)?
    for i, elem1 in enumerate(ordering):
        for elem2 in ordering[i + 1 :]:
            try:
                if verbose:
                    print(f"ensuring {elem1} < {elem2}")
                # check if each element smaller than successive ones
                if elem1 > elem2:
                    return False
            except TypeError:
                pass

    for i in range(len(ordering) - 1):
        # make sure functions match for successive entries
        left_funcs = set(lookup_dict[ordering[i]])
        right_funcs = set(lookup_dict[ordering[i + 1]])
        # interval endpoints may come from multiple functions (if piecewise boundary)
        # but there should only be one common function, hence set intersection below
        common_funcs = left_funcs.intersection(right_funcs)
        if len(common_funcs) == 0:
            return False

    return True


def reconstruct_transition_points(
    var_to_transition, ordering, add_neg_inf, add_pos_inf
):
    ordered_transitions = []
    # use extend because var_to_transition maps coordinates (x or y) to lists
    # Must start with -oo if it exists
    if add_neg_inf:
        ordered_transitions.extend(var_to_transition[-oo])
    # elif not var_to_transition[ordering[0]][0].is_bound:
    #     # must begin with a boundary type transition point
    #     return []
    # elif not var_to_transition[ordering[-1]][0].is_bound:
    #     # must also end with a boundary type transition point
    #     return []

    for coord in ordering:
        ordered_transitions.extend(var_to_transition[coord])

    # +oo must go at the end if it is to be added
    if add_pos_inf:
        ordered_transitions.extend(var_to_transition[oo])

    assert (
        ordered_transitions[0].is_bound and ordered_transitions[-1].is_bound
    ), "Ordering did not start and end with Boundary points"

    return ordered_transitions


def generate_clause(
    x,
    y,
    poly: Polygon,
    trajectory,
    sorted_transitions: List,
    lookup_dict: Dict,
    add_notches=True,
    print_latex=False,
    notches_only=False,
):
    _, vertex_pairs = compute_polygon_angles(poly)
    verts = poly.vertices
    if y not in trajectory.free_symbols:
        func_var = x
        keyfunc = lambda p: p.x
    elif x not in trajectory.free_symbols:
        func_var = y
        keyfunc = lambda p: p.y
    else:
        raise Exception("Trajectory had too many variables!")

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

    if PRINTS:
        print(sorted_transitions)
    func_var_transitions = [
        getattr(tp.point, str(func_var)) for tp in sorted_transitions
    ]
    # print(f"func_var_transitions: {func_var_transitions}")

    # # Map each midpoint angle to the active corner. Because we're only supporting
    # # symmetric polygons so far, we only need to find a single active corner here,
    # # and we can use its opposing corner (due to central symmetry) later.
    # active_corners: dict = {}
    # for midpoint_angle in midpoint_angles:
    #     for k, v in corners_to_angles.items():
    #         print(midpoint_angle)
    #         print(v)
    #         # NOTE: fails for symbolic trajectory parameters
    #         if midpoint_angle % (2 * pi) in v:
    #             active_corners[midpoint_angle] = k
    var_intervals = list(
        zip(func_var_transitions[:-1], func_var_transitions[1:])
    )  # same len as midpoints
    assert var_intervals, f"var_intervals: {var_intervals}. Evaluated to False."

    # print(f"var intervals: {var_intervals}")

    # Construct the full boolean formulation of the *unsafe* region
    total_cond = None
    for i, var_interval in enumerate(var_intervals):
        # skip intervals with measure 0
        if var_interval[0] == var_interval[1]:
            continue

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

        # Ensure this check only applies between the transition points (plus/minus width)
        # Discard nonsense > -inf or < +inf
        if func_var == x:
            if x_left is -oo:
                func_var_cond = x <= (x_right + w)
            elif x_right is oo:
                func_var_cond = x >= (x_left - w)
            else:
                func_var_cond = (x >= (x_left - w)) & (x <= (x_right + w))
        elif func_var == y:
            # note that w is actually height when found above
            if y_left is -oo:
                func_var_cond = y <= (y_right + w)
            elif y_right is oo:
                func_var_cond = y >= (y_left - w)
            else:
                func_var_cond = (y >= (y_left - w)) & (y <= (y_right + w))

        # construct g functions
        # same trajectory over this interval, held constant outside of it
        if type(trajectory) == Piecewise:
            if func_var == x:
                left_funcs = set(lookup_dict[x_left])
                right_funcs = set(lookup_dict[x_right])
                # interval endpoints may come from multiple functions (if piecewise boundary)
                # but there should only be one common function, hence set intersection below
                common_funcs = left_funcs.intersection(right_funcs)
                if len(common_funcs) != 1:
                    print(
                        f"No common function over interval [{x_left}, {x_right}]: functions were {left_funcs} and {right_funcs} with overlap {common_funcs}"
                    )
                    print(trajectory)
                    print(var_intervals)
                    print(sorted_transitions)
                    return []
                assert (
                    len(common_funcs) == 1
                ), f"Ambiguous which function to use here, given function intersection {common_funcs}"
                current_fn = common_funcs.pop()

                # Discard nonsense infinities
                # TODO(nishant): pull this logic out into a different function
                if x_left is -oo:
                    g = y - Piecewise(
                        (current_fn, x <= x_right),
                        (y_right, x > x_right),
                    )
                elif x_right is oo:
                    g = y - Piecewise(
                        (y_left, x < x_left),
                        (current_fn, x <= x_right),
                    )
                else:
                    g = y - Piecewise(
                        (y_left, x < x_left),
                        (current_fn, x <= x_right),
                        (y_right, x > x_right),
                    )
            elif func_var == y:
                # Use an open interval in case two cases hold exactly at y_left
                left_funcs = set(lookup_dict[y_left])
                right_funcs = set(lookup_dict[y_right])
                common_funcs = left_funcs.intersection(right_funcs)
                assert len(common_funcs) == 1, "Ambiguous which function to use here!"
                current_fn = common_funcs.pop()

                if y_left is -oo:
                    g = x - Piecewise(
                        (current_fn, y <= y_right),
                        (x_right, y > y_right),
                    )
                elif y_right is oo:
                    g = x - Piecewise(
                        (x_left, y < y_left),
                        (current_fn, y <= y_right),
                    )
                else:
                    g = x - Piecewise(
                        (x_left, y < y_left),
                        (current_fn, y <= y_right),
                        (x_right, y > y_right),
                    )
        else:
            if func_var == x:
                # Discard nonsense infinities
                if x_left is -oo:
                    g = y - Piecewise(
                        (trajectory, x <= x_right),
                        (y_right, x > x_right),
                    )
                elif x_right is oo:
                    g = y - Piecewise(
                        (y_left, x < x_left),
                        (trajectory, x <= x_right),
                    )
                else:
                    g = y - Piecewise(
                        (y_left, x < x_left),
                        (trajectory, x <= x_right),
                        (y_right, x > x_right),
                    )
            elif func_var == y:
                # Discard nonsense infinities
                if y_left is -oo:
                    g = x - Piecewise(
                        (trajectory, y <= y_right),
                        (x_right, y > y_right),
                    )
                elif y_right is oo:
                    g = x - Piecewise(
                        (x_left, y < y_left),
                        (trajectory, y <= y_right),
                    )
                else:
                    g = x - Piecewise(
                        (x_left, y < y_left),
                        (trajectory, y <= y_right),
                        (x_right, y > y_right),
                    )

        # Assume symmetric polygon and test all active corner pairs
        full_cond = None
        for vert in verts[0 : len(verts) // 2]:
            # TODO!: handle infinities! one sided??
            if (
                x_left is not oo
                and x_left is not -oo
                and x_right is not oo
                and x_right is not -oo
                and y_left is not oo
                and y_left is not -oo
                and y_right is not oo
                and y_right is not -oo
                # x_left.is_finite
                # and y_left.is_finite
                # and x_right.is_finite
                # and y_right.is_finite
            ):
                poly_center_left = Point(x_left, y_left)
                poly_center_right = Point(x_right, y_right)

                line_left = Line(poly_center_left, poly_center_left + vert)
                line_right = Line(poly_center_right, poly_center_right + vert)
                # using Line.equation() creates duplicate x,y variables and ruins .subs() call later
                # Ensure this check only applies between the active corners at the start and end of the interval
                left_a, left_b, left_c = line_left.coefficients
                right_a, right_b, right_c = line_right.coefficients
                left_eq = left_a * x + left_b * y + left_c
                right_eq = right_a * x + right_b * y + right_c
                corner_cond = left_eq * right_eq <= 0
            elif (x_left is -oo and func_var == x) or (y_left is -oo and func_var == y):
                poly_center_right = Point(x_right, y_right)

                # TODO!: add some prints to debug here
                line_right = Line(poly_center_right, poly_center_right + vert)
                # using Line.equation() creates duplicate x,y variables and ruins .subs() call later
                # Ensure this check only applies between the active corners at the start and end of the interval
                right_a, right_b, right_c = line_right.coefficients
                right_eq = right_a * x + right_b * y + right_c
                # TODO: this is slow
                # construct left-hand inequality (out to -inf)
                ineq_rhs = solve(right_eq, func_var)
                # ineq_rhs = solveset(right_eq, func_var, Reals)
                if ineq_rhs:
                    # if soln exists
                    # if type(ineq_rhs) == Intersection:
                    #     if Reals in ineq_rhs.args:
                    #         ineq_rhs = list(ineq_rhs.args[1])[0]
                    #         print(ineq_rhs)
                    corner_cond = func_var <= ineq_rhs[0]
                else:
                    if PRINTS:
                        print(
                            f"Skipping non-finite interval between ({x_left}, {y_left}) and ({x_right}, {y_right})"
                        )
                    corner_cond = True
            elif (x_right is oo and func_var == x) or (y_right is oo and func_var == y):
                poly_center_left = Point(x_left, y_left)

                # TODO!: add some prints to debug here
                line_left = Line(poly_center_left, poly_center_left + vert)
                # using Line.equation() creates duplicate x,y variables and ruins .subs() call later
                # Ensure this check only applies between the active corners at the start and end of the interval
                left_a, left_b, left_c = line_left.coefficients
                left_eq = left_a * x + left_b * y + left_c
                # construct right-hand inequality (out to +inf)
                # TODO: this is slow
                ineq_rhs = solve(left_eq, func_var)
                # ineq_rhs = solveset(left_eq, func_var, Reals)
                if ineq_rhs:
                    corner_cond = func_var >= ineq_rhs[0]
                else:
                    if PRINTS:
                        print(
                            f"Skipping non-finite interval between ({x_left}, {y_left}) and ({x_right}, {y_right})"
                        )
                    corner_cond = True
            else:
                # Set to True, effectively ignoring it in the And
                if PRINTS:
                    print(
                        f"Skipping non-finite interval between ({x_left}, {y_left}) and ({x_right}, {y_right})"
                    )
                corner_cond = True

            g1 = g.subs(x, x - vert.x).subs(y, y - vert.y)
            g2 = g.subs(x, x + vert.x).subs(y, y + vert.y)
            if full_cond is None:
                full_cond = func_var_cond & corner_cond & (g1 * g2 <= 0)
            else:
                full_cond = full_cond | (func_var_cond & corner_cond & (g1 * g2 <= 0))

        if total_cond is None:
            # print(full_cond)
            total_cond = full_cond
        else:
            # print(full_cond, "not none")
            total_cond = total_cond | full_cond

    if add_notches:
        # TODO remove me!!
        if notches_only:
            total_cond = False
        # adding notches
        for transition_point in sorted_transitions:
            if (
                transition_point.point.x is not oo
                and transition_point.point.x is not -oo
                and transition_point.point.y is not oo
                and transition_point.point.y is not -oo
            ):
                # neg for left side, 0 for on edge, pos for on right side
                # inside polygon IF all neg or IF all pos
                shifted_vertex_pairs = [
                    (v + transition_point.point, nextv + transition_point.point)
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
            else:
                if PRINTS:
                    print(f"Skipping non-finite notch point {transition_point}")

    if print_latex:
        print(latex(total_cond))

    return total_cond


def compute_unsafe_conds_symbolic(
    x,
    y,
    poly,
    trajectory,
    domain=Reals,
    intervals=None,
    add_notches=True,
    print_runtime=False,
    print_orderings=False,
    print_latex=False,
    notches_only=False,
):
    transitions, lookup, func_var = compute_all_transitions(
        x,
        y,
        poly,
        trajectory,
        domain=domain,
        intervals=intervals,
    )

    print(transitions)

    t0 = time.time()
    transition_orderings = sort_or_order(transitions, lookup, func_var)
    if print_orderings:
        for ordering in transition_orderings:
            for i, tp in enumerate(ordering):
                if i == 0:
                    print(f"[{tp},")
                elif i == len(ordering) - 1:
                    print(f"{tp}]")
                else:
                    print(f"{tp},")

        print("")
    if print_runtime:
        print(
            f"Took {time.time()-t0} seconds to compute",
            f"{len(transition_orderings)} possible orderings.",
        )

    t0 = time.time()
    # clauses = [
    #     generate_clause(
    #         x,
    #         y,
    #         poly,
    #         trajectory,
    #         sorted_transitions,
    #         lookup,
    #         add_notches=add_notches,
    #         notches_only=notches_only,
    #         print_latex=print_latex,
    #     )
    #     for sorted_transitions in transition_orderings
    # ]

    # generate all clauses
    clauses = []
    for i, sorted_transitions in enumerate(transition_orderings):
        if print_runtime and i % 3 == 0:
            print(f"{i} out of {len(transition_orderings)} done")
        clause = generate_clause(
            x,
            y,
            poly,
            trajectory,
            sorted_transitions,
            lookup,
            add_notches=add_notches,
            notches_only=notches_only,
            print_latex=print_latex,
        )
        # some may be empty if they fail
        if clause:
            clauses.append(clause)
        else:
            print(f"failure to generate clause for ordering {sorted_transitions}")

    if print_runtime:
        print(f"Took {time.time()-t0} seconds to compute {len(clauses)} clauses")

    return clauses, ExplicitFormulationSymbolic(
        clauses, transition_orderings, lookup, trajectory, intervals, func_var
    )
