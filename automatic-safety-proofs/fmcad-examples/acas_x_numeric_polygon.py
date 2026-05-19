import time
from sympy import *
from safe_region_utils import *
from symbolic_utils import *
import os, psutil
import resource


def full_acas_x_example(params=[]):
    initial_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2

    t0 = time.time()
    x, y = symbols("x y")
    w = Symbol("w", real=True, positive=True, nonzero=True)
    h = Symbol("h", real=True, positive=True, nonzero=True)
    rect_points: list = [
        geometry.Point(val) for val in [[w, -h], [w, h], [-w, h], [-w, -h]]
    ]
    rect_param: geometry.Polygon = Polygon(*rect_points)
    c = Symbol("c", real=True, nonzero=True)  # curvature
    b = Symbol("b", real=True, positive=True)  # boundary
    # b = Symbol('b', real=True) # boundary

    traj_piecewise = Piecewise(
        (c * x**2, x < b),
        (2 * b * c * (x - b) + b**2 * c, x >= b),
    )

    piecewise_intervals = [Interval(-oo, b), Interval(b, oo)]
    piecewise_intervals = [sub_int.subs(params) for sub_int in piecewise_intervals]

    clauses_acas, explicit_acas = compute_unsafe_conds_symbolic(
        x,
        y,
        rect_param.subs(params),
        traj_piecewise.subs(params),
        domain=Reals,
        intervals=piecewise_intervals,
    )
    symbolic_time = time.time() - t0
    print(f"Took {symbolic_time} seconds to compute symbolic safe region")
    inst_begin_time = time.time()

    numeric_params = dict([(c, 0.25), (b, 2), (w, 2), (h, 1)])
    for (k, v) in params:
        if k in numeric_params:
            numeric_params.pop(k)

    numeric_acas = explicit_acas.instantiate(list(numeric_params.items()))
    inst_duration = time.time() - inst_begin_time
    total_duration = time.time() - t0
    #     print(f"Took {time.time() - t0} seconds to instantiate")
    #     print(numeric_acas.ordering)
    #     numeric_acas.clause
    print(
        f"Took {inst_duration} seconds to instantiate and {total_duration} seconds total."
    )
    final_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"RAM usage: {final_ram - initial_ram} MB")
    return symbolic_time, inst_duration, total_duration


if __name__ == "__main__":
    w = Symbol("w", real=True, positive=True, nonzero=True)
    h = Symbol("h", real=True, positive=True, nonzero=True)

    params = [(w, 2), (h, 1)]
    print("numeric rectangle")
    full_acas_x_example()
