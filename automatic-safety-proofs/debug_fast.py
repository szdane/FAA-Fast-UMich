#!/usr/bin/env python
# coding: utf-8
"""
Debug test for automatic proof generation with lemma generation.
"""

# from sympy import symbols, Point, Polygon, RegularPolygon, Interval, oo, Piecewise
# from pvs_utils import (
#     generate_complete_proof_package,
#     print_prooflite,
#     log_proof_to_file,
# )

# # Define symbols
# x = symbols("x")

# # Create a rectangle polygon with width 4 (half-width 2)
# w = 1.0
# rect_points = [
#     Point(val) for val in [[2 * w, -w], [2 * w, w], [-2 * w, w], [-2 * w, -w]]
# ]
# polygon = Polygon(*rect_points)
# # diamond_points = [Point(val) for val in [[0, 1], [1, 0], [0, -1], [-1, 0]]]
# # polygon = Polygon(*diamond_points)
# # polygon = RegularPolygon((0, 0), 1, n=6)  # hexagon

# # trajectory_expr = x**2
# trajectory_expr = Piecewise((182.0*x, (x < 1.0) & (x >= 0)), (182.0, (x >= 1.0) & (x < 2.0)), (426.2 - 122.1*x, (x >= 2.0) & (x < 3.0)), (59.9, (x >= 3.0) & (x < 7.0)), (35.6*x - 189.3, (x >= 7.0) & (x < 8.0)), (1551.5 - 182.0*x, (x >= 8.0) & (x < 9.0)), (146.4*x - 1404.1, (x >= 9.0) & (x < 10.0)), (59.9, (x >= 10.0) & (x < 11.0)), (718.8 - 59.9*x, (x >= 11.0) & (x < 12.0)), (0, (x >= 12.0) & (x <= 13.0)))
# # Domain (infinite)
# domain = Interval(0,13)
# # domain = Interval(-oo, 10)
# # domain = Interval(0, 10)

# print("Debug test with y = x^2 trajectory and lemma generation...")
# print(f"Trajectory: {trajectory_expr}")
# print(f"Domain: {domain}")

# # Add debug prints to understand the transition points
# print(f"\nDomain bounds: inf={domain.inf}, sup={domain.sup}")

# try:
#     # Generate complete proof package including lemmas
#     print(f"\n{'='*60}")
#     print("GENERATING COMPLETE PROOF PACKAGE:")
#     print(f"{'='*60}")

#     package = generate_complete_proof_package(
#         trajectory_expr, polygon, domain
#     )

#     print(f"Package keys: {list(package.keys())}")
#     print(f"Number of proof calls: {len(package['proof_calls'])}")
#     print(f"Number of lemmas: {len(package['lemmas'])}")
#     print(f"Number of proof scripts: {len(package['proof_scripts'])}")

#     # Display proof calls with lemma information
#     print(f"\n{'='*60}")
#     print("PROOF CALLS WITH LEMMA INFORMATION:")
#     print(f"{'='*60}")

#     for i, call in enumerate(package["proof_calls"]):
#         print(f"\nProof call {i+1}:")
#         print(f"  Case labels: {call['case_labels']}")
#         print(f"  Deriv lemma: {call['deriv_lemma']}")
#         print(f"  Max right: {call['max_right']}")
#         print(f"  Min left: {call['min_left']}")
#         print(f"  Domain definition: {call['domain_definition']}")
#         print(f"  Interval: {call['interval']}")
#         print(f"  Active corner: {call['active_corner']}")
#         print(f"  Lemma name: {call.get('lemma_name', 'N/A')}")
#         print(f"  Deriv clause: {call.get('deriv_clause1', 'N/A')}")
#         print(f"  Domain start: {call.get('domain_start', 'N/A')}")
#         print(f"  Domain end: {call.get('domain_end', 'N/A')}")

#     # Display lemmas and proof scripts together
#     print(f"\n{'='*60}")
#     print("LEMMAS AND PROOF SCRIPTS TOGETHER:")
#     print(f"{'='*60}")

#     print(print_prooflite(package))

#     log_proof_to_file(package, "debug_fast.pvs")

#     # print(f"Debug: Number of lemmas: {len(package['lemmas'])}")
#     # print(f"Debug: Number of proof scripts: {len(package['proof_scripts'])}")
#     # print(f"Debug: Lemmas type: {type(package['lemmas'])}")
#     # print(
#     #     f"Debug: First lemma: {package['lemmas'][0] if package['lemmas'] else 'None'}"
#     # )

#     for i in range(len(package["lemmas"])):
#         print(f"\n{'='*60}")
#         print(f"CASE {i+1}:")
#         print(f"{'='*60}")

#         # Print the lemma
#         print(f"\nLEMMA {i+1}:")
#         print(f"{'='*40}")
#         print(package["lemmas"][i])
#         print(f"{'='*40}")

#         # Print the corresponding proof script
#         print(f"\nPROOF SCRIPT {i+1}:")
#         print(f"{'='*40}")
#         print(package["proof_scripts"][i])
#         print(f"{'='*40}")
# except Exception as e:
#     print(f"Error: {e}")
#     import traceback

#     traceback.print_exc()



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
        default="example_2.pvs",
        help="Output filename for the proof (default: example_2.pvs)",
    )

    args = parser.parse_args()

    # Define symbols
    x = symbols("x")

    # Create a polygon
    w = 1.0
    square_points = [Point(val) for val in [[w, w], [w, -w], [-w, -w], [-w, w]]]
    polygon = Polygon(*square_points)

    trajectory_expr = Piecewise((182.0*x, (x < 1.0) & (x >= 0)), (182.0, (x >= 1.0) & (x < 2.0)), (426.2 - 122.1*x, (x >= 2.0) & (x < 3.0)), (59.9, (x >= 3.0) & (x < 7.0)), (35.6*x - 189.3, (x >= 7.0) & (x < 8.0)), (1551.5 - 182.0*x, (x >= 8.0) & (x < 9.0)), (146.4*x - 1404.1, (x >= 9.0) & (x < 10.0)), (59.9, (x >= 10.0) & (x < 11.0)), (718.8 - 59.9*x, (x >= 11.0) & (x < 12.0)), (0, (x >= 12.0) & (x <= 13.0)))
    # Domain (infinite)
    domain = Interval(0,13)

    package = generate_complete_proof_package(trajectory_expr, polygon, domain)

    log_proof_to_file(package, args.output_filename)

    print(f"Proof generated successfully and saved to: {args.output_filename}")


if __name__ == "__main__":
    main()