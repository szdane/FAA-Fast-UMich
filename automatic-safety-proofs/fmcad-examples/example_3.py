#!/usr/bin/env python
# coding: utf-8
from sympy import *
from safe_region_utils import *

init_printing()


x, y = symbols("x y")

## Third example: Application 2 ##
# Define a polygon
rp = 2
hexagon = RegularPolygon(Point(0, 0), rp, 6)

print("Plotting object. Close plot to continue example...\n")
plot_polygon(hexagon)

# Define a trajectory

R = 10
theta = pi / 3
bound = R / sqrt(tan(theta) ** 2 + 1)

traj_piecewise = Piecewise(
    (sqrt(R ** 2 - x ** 2), x > bound),
    (-1 / tan(theta) * (x - R * cos(theta)) + R * sin(theta), x <= bound),
)
print("Plotting trajectory. Close plot to continue example...\n")
plot(traj_piecewise)


# Define domain and plot bounds
domain = Interval(-12, 10)
xbounds = (domain.inf - 3, domain.sup + 3)
ybounds = (-3, 22)


# Run algorithm
example_name = "Top-Down UAV Trajectory"

cond = compute_unsafe_cond(
    x,
    y,
    hexagon,
    traj_piecewise,
    domain=domain,
)
print("Boolean condition for unsafe region:\n", cond)

print(
    "\nPlotting dot grid visualization of safe and unsafe regions. This may take up to 20 seconds to plot.\nOnce displayed, close plot to continue example...\n"
)

plot_condition(x, y, cond, xbounds, ybounds, title=example_name, resolution=0.5)
mathematica_output = print_mathematica(
    x, y, cond, xbounds, ybounds, traj_piecewise, hexagon
)
print("Mathematica command for plotting:\n", mathematica_output)
