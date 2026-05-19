#!/usr/bin/env python
# coding: utf-8
"""
Debug test for automatic proof generation with lemma generation.
"""

import argparse
from sympy import symbols, Point, Polygon, RegularPolygon, Interval, oo, Piecewise
from pvs_utils import (
    generate_complete_proof_package,
    log_proof_to_file,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Example for automatic proof generation"
    )
    parser.add_argument(
        "output_filename",
        nargs="?",
        default="example_5.pvs",
        help="Output filename for the proof (default: example_5.pvs)",
    )

    args = parser.parse_args()

    # Define symbols
    x = symbols("x")

    # Create a polygon
    w = 1.0
    rect_points = [
        Point(val) for val in [[2 * w, -w], [2 * w, w], [-2 * w, w], [-2 * w, -w]]
    ]
    polygon = Polygon(*rect_points)

    # Trajectory function: should be a SymPy expression in terms of x only
    trajectory_expr = Piecewise((x**2, x <= 4), (8 * x - 16, x > 4))

    # Domain
    domain = Interval(-oo, oo)

    package = generate_complete_proof_package(trajectory_expr, polygon, domain)

    log_proof_to_file(package, args.output_filename)

    print(f"Proof generated successfully and saved to: {args.output_filename}")


if __name__ == "__main__":
    main()
