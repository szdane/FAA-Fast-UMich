#!/usr/bin/env python
# coding: utf-8

# ── Imports ──────────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add Active Corner Method/ to path so imports work when run directly

import matplotlib
matplotlib.use('TkAgg')              # use Tkinter backend instead of Qt (avoids Qt platform plugin error)

from sympy import *                  # symbolic math library (symbols, Piecewise, sin, etc.)
from safe_region_utils import *      # Active Corner Method functions (compute_unsafe_cond, plot_condition, etc.)

init_printing()                      # display symbolic expressions in a readable format

x, y = symbols("x y")               # declare x and y as symbolic variables (not numbers)


# ── 1. Define the polygon (aircraft shape) ───────────────────────────────────
w = 0.5                              # half-width of the square (so the full square is 1×1)
square_points: list = [
    geometry.Point(val) for val in [[w, -w], [w, w], [-w, w], [-w, -w]]
]                                    # list the 4 corners of the square, centered at the origin
square: geometry.Polygon = Polygon(*square_points)  # build the sympy Polygon object from those corners

print("Plotting object. Close plot to continue example...\n")
plot_polygon(square)                 # visualize the square shape before running the algorithm


# ── 2. Define the trajectory (path the aircraft flies) ───────────────────────
traj_piecewise = Piecewise(
    (sin(x / 2), x < 0),            # for x < 0: follow a sine curve  y = sin(x/2)
    (x / 2,      x >= 0),           # for x ≥ 0: follow a straight line  y = x/2
)                                    # the two pieces join smoothly at x=0, y=0

print("Plotting trajectory. Close plot to continue example...\n")
plot(traj_piecewise)                 # visualize the trajectory before running the algorithm


# ── 3. Set the domain and plot bounds ────────────────────────────────────────
domain  = Interval(-12, 9)          # x-range over which the aircraft actually flies
xbounds = [-15, 12]                 # x-range for the output visualization plot (wider than domain)
ybounds = [-3, 6]                   # y-range for the output visualization plot
example_name = "Sine-Linear with Square"  # title shown on the plot


# ── 4. Run the Active Corner Method algorithm ────────────────────────────────
cond = compute_unsafe_cond(
    x,                               # symbolic x variable
    y,                               # symbolic y variable
    square,                          # the polygon (aircraft shape)
    traj_piecewise,                  # the trajectory (path flown)
    domain,                          # the x-range to analyze
)                                    # returns a symbolic boolean formula: True = intruder is UNSAFE

print("Boolean condition for unsafe region:\n", cond)   # print the formula (for inspection)


# ── 5. Visualize the safe vs unsafe regions ──────────────────────────────────
print(
    "\nPlotting dot grid visualization of safe and unsafe regions. "
    "This may take up to 20 seconds to plot.\n"
    "Once displayed, close plot to continue example...\n"
)
plot_condition(x, y, cond, xbounds, ybounds, title=example_name)
# evaluates `cond` at every grid point in the bounds and colors dots:
#   blue  = safe   (intruder here will NOT be hit)
#   orange = unsafe (intruder here WILL be hit)


# ── 6. Export to Mathematica for a higher-quality plot ───────────────────────
mathematica_output = print_mathematica(
    x, y, cond, xbounds, ybounds, traj_piecewise, square
)                                    # converts `cond` into a Mathematica RegionPlot[] command
print("Mathematica command for plotting:\n", mathematica_output)  # paste this into Mathematica
