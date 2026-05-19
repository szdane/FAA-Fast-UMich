import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import *


def plot_condition(
    x,
    y,
    cond,
    xbounds=(-5, 5),
    ybounds=(-5, 5),
    resolution=0.25,
    title="Safe Region Plot",
    alpha=1,
    savefig=False,
):
    # First, check if we can plot `cond` at all
    if len(cond.free_symbols) > 2:
        print(
            "Can't plot `cond` since it has too many variables.",
            "Please use .subs() to plot an instantation of this condition.",
        )
        print(
            f"Extraneous variables to substitute out are {set(map(str, cond.free_symbols)) - {'x', 'y'}}."
        )
        return

    fig = plt.figure()
    fig.set_dpi(200)
    ax = fig.gca()

    # nelem = (xbounds[1] - xbounds[0]) * (ybounds[1] - ybounds[0]) / (resolution ** 2)
    # count = 0
    SAFE_COLOR = "#1f77b4"
    UNSAFE_COLOR = "#ff7f0e"
    if resolution < 0.5:
        dotscale = 6
    elif resolution < 1:
        dotscale = 4
    else:
        dotscale = 3

    xpoints_safe = []
    xpoints_unsafe = []
    ypoints_safe = []
    ypoints_unsafe = []
    for x0 in np.arange(xbounds[0], xbounds[1], resolution):
        for y0 in np.arange(ybounds[0], ybounds[1], resolution):
            # count += 1
            is_safe = (~cond).subs([(x, x0), (y, y0)])
            # print(is_safe, x0, y0)
            if is_safe:
                xpoints_safe.append(x0)
                ypoints_safe.append(y0)
            else:
                xpoints_unsafe.append(x0)
                ypoints_unsafe.append(y0)
    ax.scatter(xpoints_safe, ypoints_safe, s=resolution * dotscale, c=SAFE_COLOR)
    ax.scatter(
        xpoints_unsafe,
        ypoints_unsafe,
        s=resolution * dotscale,
        c=UNSAFE_COLOR,
        marker="^",
    )
    ax.legend(["safe", "unsafe"])
    ax.axis("equal")

    ax.set_title(title)
    if savefig:
        plt.savefig(title)
    plt.show()


def plot_polygon(poly: sympy.Polygon):
    # Draw a polygon by plotting vertices as points and edges as lines
    fig = plt.figure()
    ax = fig.gca()

    verts = poly.vertices

    for p in verts:
        ax.scatter(p.x, p.y, c="r")
        plt.text(p.x + 0.05, p.y + 0.05, f"({p.x},{p.y})")

    for p, nextp in zip(verts, verts[1:] + verts[:1]):
        x = np.linspace(float(p.x), float(nextp.x), 100, dtype=float)
        y = np.linspace(float(p.y), float(nextp.y), 100, dtype=float)
        ax.plot(x, y, c="b")

    ax.axis("equal")

    plt.show()


def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    # Fix for > sympy v1.5
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines["right"].set_color("none")
    backend.ax.spines["bottom"].set_position("zero")
    backend.ax.spines["top"].set_color("none")
    plt.close(backend.fig)
