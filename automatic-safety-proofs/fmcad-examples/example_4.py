#!/usr/bin/env python
# coding: utf-8
from sympy import *
from safe_region_utils import *

init_printing()


x, y = symbols("x y")


## Fourth example: f(y) example that is part of example 2 ##
# Define a polygon
w = 0.5
square_points: list = [
    geometry.Point(val) for val in [[w, -w], [w, w], [-w, w], [-w, -w]]
]
square: geometry.Polygon = Polygon(*square_points)
print("Plotting object. Close plot to continue example...\n")
plot_polygon(square)

# Define a trajectory
# This is a function of y, equivalent to example 2 for x >= 0
# NOTE: the trajectory is not plotted in this example because Sympy does not support
# plotting of piecewise functions of y in the x-y plane.
traj_piecewise = Piecewise((4 * sqrt(y), y < 1), (2 * y + 2, y >= 1))

# Define domain and plot bounds
# domain here is a range of y-values, not x-values as in past examples
domain = Interval(0, 4)
xbounds = [-2, 12]
ybounds = [-2, 6]

# Run algorithm
example_name = "$f(y)$ Trajectory Example"
cond = compute_unsafe_cond(
    x,
    y,
    square,
    traj_piecewise,
    domain,
)

print("Boolean condition for unsafe region:\n", cond)

print(
    "\nPlotting dot grid visualization of safe and unsafe regions. This may take up to 20 seconds to plot.\nOnce displayed, close plot to continue example...\n"
)

plot_condition(x, y, cond, xbounds, ybounds, title=example_name, resolution=0.25)
mathematica_output = print_mathematica(
    x, y, cond, xbounds, ybounds, traj_piecewise, square
)
print("Mathematica command for plotting:\n", mathematica_output)
