import time
from sympy import *
from safe_region_utils import *
from symbolic_utils import *
import resource


def full_adler(params):
    initial_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    t0 = time.time()

    x, y = symbols("x y")

    R = Symbol("R", real=True, positive=True, nonzero=True)
    theta = Symbol("theta", real=True, positive=True, nonzero=True)
    w = Symbol("w", real=True, positive=True, nonzero=True)

    hexagon = RegularPolygon(Point(0, 0), w, 6)

    bound = R / sqrt(tan(theta) ** 2 + 1)

    traj_piecewise = Piecewise(
        (sqrt(R**2 - x**2), x > bound),
        (-1 / tan(theta) * (x - R * cos(theta)) + R * sin(theta), x <= bound),
    )

    piecewise_intervals = [Interval(bound, R), Interval(-oo, bound)]
    piecewise_intervals = [sub_int.subs(params) for sub_int in piecewise_intervals]

    clauses_hex, explicit_hex = compute_unsafe_conds_symbolic(
        x,
        y,
        hexagon.subs(params),
        traj_piecewise.subs(params),
        domain=Reals,
        intervals=piecewise_intervals,
    )

    symbolic_time = time.time() - t0
    print(f"Took {symbolic_time} seconds to compute symbolic safe region")
    inst_begin_time = time.time()

    numeric_params = dict([(R, 4), (theta, pi / 3), (w, 2)])
    for (k, v) in params:
        if k in numeric_params:
            numeric_params.pop(k)

    numeric_hex = explicit_hex.instantiate(list(numeric_params.items()))
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
    print("numeric hexagon")
    full_adler([(w, 2)])
