#!/usr/bin/env python
# coding: utf-8
from sympy import *
from safe_region_utils import *

init_printing()


x, y = symbols("x y")

## Second example: Application 1 ##
# Define a polygon
w = 0.75
rect_points: list = [
    geometry.Point(val) for val in [[2 * w, -w], [2 * w, w], [-2 * w, w], [-2 * w, -w]]
]
rectangle: geometry.Polygon = Polygon(*rect_points)
print("Plotting object. Close plot to continue example...\n")
plot_polygon(rectangle)

# Define a trajectory
traj_piecewise = Piecewise((x ** 2 / 16, x < 4), (x / 2 - 1, x >= 4))
print("Plotting trajectory. Close plot to continue example...\n")
plot(traj_piecewise)

# Define domain and plotting bounds
domain = Interval(-6, 15)
xbounds = (domain.inf - 3, domain.sup + 3)
ybounds = (-2, 9)
example_name = "ACAS X Climb with Rectangle"

cond = compute_unsafe_cond(
    x,
    y,
    rectangle,
    traj_piecewise,
    domain=domain,
)
print("Boolean condition for unsafe region:\n", cond)

print(
    "\nPlotting dot grid visualization of safe and unsafe regions. This may take up to 20 seconds to plot.\nOnce displayed, close plot to continue example...\n"
)

plot_condition(x, y, cond, xbounds, ybounds, title=example_name)
mathematica_output = print_mathematica(
    x, y, cond, xbounds, ybounds, traj_piecewise, rectangle
)
print("Mathematica command for plotting:\n", mathematica_output)
