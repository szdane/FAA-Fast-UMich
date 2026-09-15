from sympy import *
from typing import Tuple, List, Dict, Set


class TransitionPoint:
    # TODO(nishant): add documentation here
    # May not always represent a transition point (sometimes is piecewise bound)
    def __init__(self, point: Point, angle, function):
        self.point = point
        self.angle = angle
        self.function = function
        self.is_bound = angle is None  # used to see if this is a piecewise bound or not

    def __repr__(self):
        if self.is_bound:
            # return f"Boundary({self.point.x}, {self.point.y}) [func {self.function}]"
            return f"Boundary({self.point.x}, {self.point.y})"
        else:
            # return f"Transition({self.point.x}, {self.point.y}) [angle {self.angle}, func {self.function}]"
            return f"Transition({self.point.x}, {self.point.y})"


class TransitionPointSet:
    def __init__(self):
        self.objects: Dict = {}
        self.unique_points: set = set()

    def add(self, tp: TransitionPoint):
        if tp.point in self.unique_points:
            return
        else:
            self.objects[tp.point] = tp
            self.unique_points.add(tp.point)

    def update(self, tps: set):
        for tp in tps:
            self.add(tp)

    # TODO: make sortable


class ExplicitFormulationSymbolic:
    def __init__(
        self,
        clauses: List,
        orderings: List,
        lookup: Dict,
        trajectory,
        intervals,
        func_var: Symbol,
    ):
        self.clauses = clauses
        self.orderings = orderings
        self.lookup = lookup
        self.trajectory = trajectory  # sympy expression, maybe piecewise
        self.intervals = intervals
        self.func_var = func_var

    def instantiate(self, values: List[Tuple[Symbol, Number]]):
        """Return a single, correct, numeric instance of this explicit formulation given parameters."""
        if len(self.clauses) == 1:
            return ExplicitFormulationNumeric(
                self.orderings[0], self.clauses[0].subs(values)
            )

        piecewise_funcs = [piece[0] for piece in self.trajectory.args]
        intervals_num = [
            piecewise_interval.subs(values) for piecewise_interval in self.intervals
        ]
        longest_ordering_idx = -1
        longest_ordering_len = -1
        for i, ordering in enumerate(self.orderings):
            # Step through each ordering and figure out if it's valid - pick the one with the MOST valid points
            # Check if a transition point lies inside a valid piecewise condition
            if check_ordering_numeric(
                ordering,
                self.lookup,
                piecewise_funcs,
                intervals_num,
                values,
                self.func_var,
            ):
                if len(ordering) > longest_ordering_len:
                    longest_ordering_idx = i
                    longest_ordering_len = len(ordering)

        return ExplicitFormulationNumeric(
            self.orderings[longest_ordering_idx],
            self.clauses[longest_ordering_idx].subs(values),
        )


class ExplicitFormulationNumeric:
    def __init__(self, ordering: List[TransitionPoint], clause):
        self.ordering = ordering
        self.clause = clause


def check_ordering_numeric(
    ordering, lookup, piecewise_funcs, piecewise_intervals_num, values, func_var
):
    for tp in ordering:
        if not tp.is_bound:  # if it's a transition point
            interval_index = piecewise_funcs.index(
                lookup[getattr(tp.point, str(func_var))][0]
            )
            tp_interval = piecewise_intervals_num[interval_index]
            point_loc_num = getattr(tp.point, str(func_var)).subs(values)
            if point_loc_num not in tp_interval:
                return False
    return True
