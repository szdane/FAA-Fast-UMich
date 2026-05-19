import time
from sympy import *
from safe_region_utils import *
from symbolic_utils import *
import resource


def dubins():
    initial_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    t0 = time.time()

    R = Symbol("R", real=True, positive=True, nonzero=True)
    theta = Symbol("theta", real=True, positive=True, nonzero=True)
    p = Symbol("p", real=True)
    b = Symbol("b", real=True)
    x, y = symbols("x y")

    bound = R / sqrt(tan(theta) ** 2 + 1)

    traj_piecewise = Piecewise(
        (sqrt(R**2 - x**2), x > bound),
        (-1 / tan(theta) * (x - R * cos(theta)) + R * sin(theta), x > p),
        (
            -sqrt(Abs(p / cos(theta)) ** 2 - (x - 2 * p) ** 2)
            + R / sin(theta)
            - p / tan(theta)
            + Abs(p * tan(theta)),
            x > b,
        ),
        (
            (
                (-2 * p + x)
                * Abs(cos(theta))
                / sqrt(p**2 - (b - 2 * p) ** 2 * cos(theta) ** 2)
            )
            * (x - b)
            + R / sin(theta)
            - p / tan(theta)
            - sqrt(
                -(b**2) * cos(theta) ** 2
                + 4 * b * p * cos(theta) ** 2
                - 4 * p**2 * cos(theta) ** 2
                + p**2
            )
            / Abs(cos(theta))
            + Abs(p * tan(theta)),
            x <= b,
        ),
    )

    h = Symbol("h", real=True, positive=True, nonzero=True)
    hexagon_param = RegularPolygon(Point(0, 0), w, 6)

    piecewise_intervals = [
        Interval(bound, R),
        Interval(p, bound),
        Interval(b, p),
        Interval(-oo, b),
    ]

    clauses_dubins, explicit_dubins = compute_unsafe_conds_symbolic(
        x,
        y,
        hexagon_param,
        traj_piecewise,
        domain=Reals,
        intervals=piecewise_intervals,
        print_orderings=True,
        print_runtime=True,
    )
    symbolic_time = time.time() - t0
    print(f"Took {symbolic_time} seconds to compute symbolic safe region")
    inst_begin_time = time.time()

    params = dict([(R, 10), (theta, pi / 3), (p, 3), (b, 1)])

    numeric_dubins = explicit_dubins.instantiate(list(params.items()))
    inst_duration = time.time() - inst_begin_time
    total_duration = time.time() - t0
    print(
        f"Took {inst_duration} seconds to instantiate and {total_duration} seconds total."
    )

    final_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    print(f"RAM usage: {final_ram - initial_ram} MB")


if __name__ == "__main__":
    print("dubins symbolic hexagon")
    dubins()
