#!/usr/bin/env python
# coding: utf-8
"""
Debug test for automatic proof generation with lemma generation.
"""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from sympy import Rational
from sympy import symbols, Point, Polygon, RegularPolygon, Interval, oo, Piecewise
from pvs_utils import (
    generate_complete_proof_package,
    print_prooflite,
    log_proof_to_file,
)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Debug test for automatic proof generation"
    )
    parser.add_argument(
        "output_filename",
        nargs="?",
        default="debug_proof.pvs",
        help="Output filename for the proof (default: debug_proof.pvs)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--print_prooflite",
        action="store_true",
        help="Print the prooflite output",
    )

    args = parser.parse_args()

    # Define symbols
    x = symbols("x")

    # Create a rectangle polygon with width 4 (half-width 2)
    w = 1.0
    rect_points = [
        Point(val) for val in [[2 * w, -w], [2 * w, w], [-2 * w, w], [-2 * w, -w]]
    ]
    polygon = Polygon(*rect_points)
    # square_points = [Point(val) for val in [[w, w], [w, -w], [-w, -w], [-w, w]]]
    # polygon = Polygon(*square_points)
    # diamond_points = [Point(val) for val in [[0, 1], [1, 0], [0, -1], [-1, 0]]]
    # polygon = Polygon(*diamond_points)
    # polygon = RegularPolygon((0, 0), 1, n=6)  # hexagon

    # trajectory_expr = x
    # trajectory_expr = -x
    # trajectory_expr = x**2 + 2 * x + 1
    # # trajectory_expr = -(x**2)
    # trajectory_expr = Piecewise((14.0 - 1.0*x, (x < 4.0)), 
    #                             (10.4 - 0.1*x, (x >= 4.0) & (x < 21.0)), 
    #                             (14.6 - 0.3*x, (x >= 21.0)))

    def qnum(x, places=12):
        d = Decimal(str(x)).quantize(Decimal("1." + "0"*places), rounding=ROUND_HALF_UP)
        return Rational(str(d))
    
    # trajectory_expr = Piecewise(
    # (qnum(14)  - qnum(1)  * x, x < qnum(4)),
    # (qnum(10.48) - qnum(0.12) * x, x < qnum(21)),
    # (qnum(14.89) - qnum(0.33) * x, x >= qnum(21)),
    # )

    trajectory_expr = Piecewise((73*x - 3011, (x <= 4)), (81*x - 3043, (x >= 4)))
                                                          
    # trajectory_expr = Piecewise((81*x - 3043,(x <= 21)), (71*x - 2833, (x > 21)))
    
    # Piecewise((14 - x, (x < 4)), (10, x>=4))
    print(trajectory_expr)
    # trajectory_expr = Piecewise((x**2 + 2 * x + 1, x <= 3), (8 * x - 8, x > 3))
    # trajectory_expr = Piecewise((-(x**2), x <= 4), (-8 * x + 16, x > 4))

    # Domain (infinite)
    domain = Interval(-oo, oo)
    # domain = Interval(-oo, 10)
    # domain = Interval(-10, 10)
    # domain = Interval(0, 10)

    try:
        # Generate complete proof package including lemmas
        if args.verbose:
            print(f"\n{'='*60}")
            print("GENERATING COMPLETE PROOF PACKAGE:")
            print(f"{'='*60}")

        package = generate_complete_proof_package(trajectory_expr, polygon, domain)

        if args.verbose:
            print(f"Package keys: {list(package.keys())}")
            print(f"Number of proof calls: {len(package['proof_calls'])}")
            print(f"Number of lemmas: {len(package['lemmas'])}")
            print(f"Number of proof scripts: {len(package['proof_scripts'])}")

            # Display proof calls with lemma information
            print(f"\n{'='*60}")
            print("PROOF CALLS WITH LEMMA INFORMATION:")
            print(f"{'='*60}")

            for i, call in enumerate(package["proof_calls"]):
                print(f"\nProof call {i+1}:")
                print(f"  Case labels: {call['case_labels']}")
                print(f"  Deriv lemma: {call['deriv_lemma']}")
                print(f"  Max right: {call['max_right']}")
                print(f"  Min left: {call['min_left']}")
                print(f"  Domain definition: {call['domain_definition']}")
                print(f"  Interval: {call['interval']}")
                print(f"  Lemma name: {call.get('lemma_name', 'N/A')}")
                print(f"  Deriv bound 1: {call.get('deriv_bound1', 'N/A')}")
                print(f"  Deriv bound 2: {call.get('deriv_bound2', 'N/A')}")
                print(f"  Domain start: {call.get('domain_start', 'N/A')}")
                print(f"  Domain end: {call.get('domain_end', 'N/A')}")

            # Display lemmas and proof scripts together
            print(f"\n{'='*60}")
            print("LEMMAS AND PROOF SCRIPTS TOGETHER:")
            print(f"{'='*60}")

        if args.print_prooflite:
            print(print_prooflite(package))

        log_proof_to_file(package, args.output_filename)

        print(f"Proof generated successfully and saved to: {args.output_filename}")

        # print(f"Debug: Number of lemmas: {len(package['lemmas'])}")
        # print(f"Debug: Number of proof scripts: {len(package['proof_scripts'])}")
        # print(f"Debug: Lemmas type: {type(package['lemmas'])}")
        # print(
        #     f"Debug: First lemma: {package['lemmas'][0] if package['lemmas'] else 'None'}"
        # )

        # for i in range(len(package["lemmas"])):
        #     print(f"\n{'='*60}")
        #     print(f"CASE {i+1}:")
        #     print(f"{'='*60}")

        #     # Print the lemma
        #     print(f"\nLEMMA {i+1}:")
        #     print(f"{'='*40}")
        #     print(package["lemmas"][i])
        #     print(f"{'='*40}")

        #     # Print the corresponding proof script
        #     print(f"\nPROOF SCRIPT {i+1}:")
        #     print(f"{'='*40}")
        #     print(package["proof_scripts"][i])
        #     print(f"{'='*40}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
