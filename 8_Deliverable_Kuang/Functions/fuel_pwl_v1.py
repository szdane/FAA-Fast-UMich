"""Local, continuous piecewise-linear approximation of OpenAP enroute fuel flow.

Each grid cell is split into six tetrahedra. A segment selects one tetrahedron
and interpolates only its four vertices; a global convex hull is not used.
The reference evaluator matches compute_fuel_and_emission: constant mass,
OpenAP's default thrust limits, and no longitudinal acceleration input.
"""

from itertools import permutations, product                                                                                    # Build axis orders and grid combinations.

import gurobipy as gp                                                                                                          # Create Gurobi variables and constraints.
import numpy as np                                                                                                             # Work with arrays and numerical calculations.
import openap                                                                                                                  # Calculate reference aircraft fuel flow.
from openap import prop                                                                                                        # Read aircraft properties, including maximum takeoff mass.


FT_TO_M = 0.3048                                                                                                               # Convert feet to metres.
KNOT_TO_MPS = 1852.0 / 3600.0                                                                                                  # Convert knots to metres per second.
FPM_TO_MPS = FT_TO_M / 60.0                                                                                                    # Convert feet per minute to metres per second.
NORM_FACETS = 32                                                                                                               # Use 32 measuring directions, not 32 allowed flight headings.
NORM_COSINE = float(np.cos(np.pi / NORM_FACETS))                                                                               # Correction based on half the angle between directions.
NORM_DIRECTIONS = np.column_stack((                                                                                            # Store east and north components for each direction.
    np.cos(np.arange(NORM_FACETS) * 2 * np.pi / NORM_FACETS),                                                                  # East components, spaced 11.25 degrees apart.
    np.sin(np.arange(NORM_FACETS) * 2 * np.pi / NORM_FACETS),                                                                  # North components of the same directions.
)) / NORM_COSINE                                                                                                               # Scale up so the speed estimate never falls below true speed.


def horizontal_speed_pwl(east_mps, north_mps):
    """Upper approximation to speed, between the true norm and norm/cos(pi/32)."""
    return float(np.max(NORM_DIRECTIONS @ [east_mps, north_mps]))


def add_horizontal_speed(model, east_mps, north_mps, name):                                                                    # Add the speed approximation to Gurobi.
    """A bounded max-of-linear speed norm, avoiding nonconvex quadratic equality."""
    projections = model.addVars(NORM_FACETS, lb=-500, ub=500, name=name + "_projection")                                       # One signed measurement per direction.
    for direction_index, (east, north) in enumerate(NORM_DIRECTIONS):                                                          # Visit all 32 measuring directions.
        model.addConstr(projections[direction_index] == east * east_mps + north * north_mps)                                   # Project velocity onto this direction.
    speed = model.addVar(lb=0, ub=300, name=name + "_mps")                                                                     # Limit approximate speed to 300 m/s.
    model.addGenConstrMax(speed, list(projections.values()), name=name + "_norm")                                              # Require speed to equal the largest measurement.
    return speed                                                                                                               # Return a solver variable, not a solved number.


class FuelFlowTable:                                                                                                           # Store OpenAP samples and build their linear interpolation model.
    def __init__(self, aircraft="B737", mass_fraction=0.8):                                                                    # Build a table for one aircraft and fixed mass.
        self.aircraft = aircraft                                                                                               # Remember the aircraft type.
        self.mass_kg = mass_fraction * prop.aircraft(aircraft)["mtow"]                                                         # Use a fixed fraction of maximum takeoff mass.
        self.reference = openap.FuelFlow(aircraft)                                                                             # Create the original OpenAP fuel calculator.
        self.axes = (                                                                                                          # Define sample locations for the three operating conditions.
            np.r_[np.arange(120.0, 581.0, 40.0) * KNOT_TO_MPS, 300.0],                                                         # Speed: 120 to 560 knots in 40-knot steps, plus 300 m/s.
            np.r_[0.0, np.arange(8000.0, 40001.0, 4000.0)] * FT_TO_M,                                                          # Altitude: 0 ft, then 8000 to 40000 ft in 4000-ft steps.
            np.arange(-1000.0, 1001.0, 500.0) * FPM_TO_MPS,                                                                    # Vertical speed: -1000 to +1000 ft/min in 500-ft/min steps.
        )                                                                                                                      # All three axes are stored in metres and seconds.
        self.shape = tuple(len(axis) for axis in self.axes)                                                                    # Count the sample locations along each axis.
        self.points = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1).reshape(-1, 3)                                 # List every speed/altitude/vertical-speed combination.
        self.flows = self.exact(self.points[:, 0], self.points[:, 1], self.points[:, 2])                                       # Sample OpenAP fuel flow at every grid point.
        if not np.isfinite(self.flows).all() or (self.flows < 0).any():                                                        # Reject missing, infinite, or negative fuel values.
            raise ValueError("OpenAP produced invalid fuel-flow grid values.")                                                 # Stop if the table cannot be trusted.
        simplices = []                                                                                                         # Collect tetrahedra: small 3D pieces with four corners each.
        for cell in product(*(range(size - 1) for size in self.shape)):                                                        # Visit every box between neighbouring grid points.
            for order in permutations(range(3)):                                                                               # Split each box into six tetrahedra using six axis orders.
                vertex = np.array(cell)                                                                                        # Start at the box's lower corner on all three axes.
                vertices = [np.ravel_multi_index(tuple(vertex), self.shape)]                                                   # Save that corner's index in the flat point list.
                for axis_index in order:                                                                                       # Move along each axis in the chosen order.
                    vertex[axis_index] += 1                                                                                    # Step to the next grid point on this axis.
                    vertices.append(np.ravel_multi_index(tuple(vertex), self.shape))                                           # Save the next corner's index.
                simplices.append(vertices)                                                                                     # Store the four corners of this tetrahedron.
        self.simplices = np.asarray(simplices)                                                                                 # Keep all tetrahedra in one array for later use.

    def exact(self, speed_mps, altitude_m, vertical_mps):                                                                      # Evaluate original OpenAP, without the PWL approximation.
        """Return kg/s with the same units and defaults as the existing report."""
        return np.asarray(self.reference.enroute(                                                                              # Ask OpenAP for fuel flow at these conditions.
            mass=self.mass_kg,                                                                                                 # Keep aircraft mass fixed in kilograms.
            tas=np.asarray(speed_mps) / KNOT_TO_MPS,                                                                           # Convert speed to knots for OpenAP.
            alt=np.asarray(altitude_m) / FT_TO_M,                                                                              # Convert altitude to feet for OpenAP.
            vs=np.asarray(vertical_mps) / FPM_TO_MPS,                                                                          # Convert vertical speed to feet per minute.
        ), dtype=float)                                                                                                        # Return fuel flow in kg/s as numerical values.

    def interpolate(self, speed_mps, altitude_m, vertical_mps):                                                                # Calculate PWL fuel flow for known numerical inputs.
        """Evaluate exactly the same simplicial interpolant without a solver."""
        vertices, weights = self._simplex_weights(speed_mps, altitude_m, vertical_mps)                                         # Find the four surrounding corners and their shares.
        return float(weights @ self.flows[vertices])                                                                           # Blend those four fuel-flow samples using their shares.

    def _simplex_weights(self, speed_mps, altitude_m, vertical_mps):                                                           # Locate one point inside the tetrahedral grid.
        values = (speed_mps, altitude_m, vertical_mps)                                                                         # Keep inputs in the same order as the table axes.
        cell, fractions = [], []                                                                                               # Store the containing box and the position within it.
        for axis, value in zip(self.axes, values):                                                                             # Check each operating condition against its axis.
            if not np.isfinite(value) or value < axis[0] - 1e-7 or value > axis[-1] + 1e-7:                                    # Allow only tiny rounding errors at the edges.
                raise ValueError("Fuel evaluation outside the calibrated grid.")                                               # Do not estimate fuel beyond the sampled range.
            value = np.clip(value, axis[0], axis[-1])                                                                          # Remove any tiny edge overshoot allowed above.
            index = min(max(int(np.searchsorted(axis, value, side="right")) - 1, 0), len(axis) - 2)                            # Find the lower end of the containing interval.
            cell.append(index)                                                                                                 # Save this axis's box index.
            fractions.append((value - axis[index]) / (axis[index + 1] - axis[index]))                                          # Measure progress from 0 to 1 across this interval.
        order = np.argsort(-np.asarray(fractions), kind="stable")                                                              # Largest progress first identifies the tetrahedron.
        ordered = np.asarray(fractions)[order]                                                                                 # Arrange the three progress values in descending order.
        weights = np.r_[1.0 - ordered[0], ordered[:-1] - ordered[1:], ordered[-1]]                                             # Compute four nonnegative corner shares that sum to one.
        vertex = np.array(cell)                                                                                                # Start at the containing box's lower corner.
        vertices = [np.ravel_multi_index(tuple(vertex), self.shape)]                                                           # Record the first corner's flat index.
        for axis_index in order:                                                                                               # Follow the axis order for this tetrahedron.
            vertex[axis_index] += 1                                                                                            # Move to its next corner.
            vertices.append(np.ravel_multi_index(tuple(vertex), self.shape))                                                   # Record that corner's flat index.
        return vertices, weights                                                                                               # Return the four sample indices and their interpolation shares.

    def add_segment(self, model, speed_mps, altitude_m, vertical_mps, inactive,                                                # Accept solver expressions for one flight interval.
                    duration_seconds, name, altitude_bounds=None, start_point=None):                                           # Also accept its duration, bounds, and optional starting guess.
        """Charge a full flown interval; inactive is the PREVIOUS arrival flag."""
        if duration_seconds <= 0:                                                                                              # Fuel needs an interval with positive length.
            raise ValueError("Segment duration must be positive.")                                                             # Reject zero or negative durations.
        simplices = self.simplices                                                                                             # Begin with every tetrahedron in the table.
        if altitude_bounds is not None:                                                                                        # Reduce model size when reachable altitude limits are known.
            heights = self.points[simplices, 1]                                                                                # Get the four corner altitudes of each tetrahedron.
            lower, upper = altitude_bounds                                                                                     # Read the reachable altitude range for this interval.
            simplices = simplices[(heights.max(axis=1) >= lower) & (heights.min(axis=1) <= upper)]                             # Keep pieces whose altitude ranges overlap it.
        vertex_ids = np.unique(simplices)                                                                                      # Collect the distinct corners used by the remaining pieces.
        choose = model.addVars(len(simplices), vtype=gp.GRB.BINARY, name=name + "_tet")                                        # Give each tetrahedron an on/off selection variable.
        blend = model.addVars(vertex_ids.tolist(), lb=0.0, ub=1.0, name=name + "_blend")                                       # Give each corner a share between zero and one.
        model.addConstr(choose.sum() == 1 - inactive, name=name + "_one_tet")                                                  # Choose one piece while flying, none after arrival.
        model.addConstr(blend.sum() == 1 - inactive, name=name + "_unit_blend")                                                # Shares total one while flying and zero when inactive.
        incident = {vertex: [] for vertex in vertex_ids}                                                                       # Track which tetrahedra contain each corner.
        for simplex_index, simplex in enumerate(simplices):                                                                    # Visit each available tetrahedron.
            for vertex in simplex:                                                                                             # Visit its four corners.
                incident[vertex].append(choose[simplex_index])                                                                 # Link this corner to this piece's selection variable.
        for vertex in vertex_ids:                                                                                              # Restrict every corner's interpolation share.
            model.addConstr(blend[vertex] <= gp.quicksum(incident[vertex]), name=f"{name}_local_{vertex}")                     # Only corners of the selected piece may have nonzero shares.
        states = [gp.quicksum(self.points[vertex, dimension] * blend[vertex] for vertex in vertex_ids)                         # Blend corner conditions with the same shares as fuel.
                  for dimension in range(3)]                                                                                   # Build speed, altitude, and vertical speed in that order.
        model.addConstr(speed_mps == states[0], name=name + "_speed")                                                          # Match interval speed; inactive intervals must have zero speed.
        model.addConstr(vertical_mps == states[2], name=name + "_vertical_speed")                                              # Match vertical speed; inactive intervals must have zero vertical speed.
        model.addGenConstrIndicator(inactive, 0, altitude_m == states[1], name=name + "_altitude")                             # Match altitude only while flying, not after arrival.
        fuel = model.addVar(lb=0.0, ub=duration_seconds * self.flows.max(), name=name + "_kg")                                 # Create interval fuel in kg with a table-based upper bound.
        model.addConstr(fuel == duration_seconds * gp.quicksum(                                                                # Fuel equals the interval length times blended flow.
            self.flows[vertex] * blend[vertex] for vertex in vertex_ids                                                        # Weight each corner's kg/s value by its share.
        ), name=name + "_integrate")                                                                                           # This also forces zero fuel when all shares are zero.
        if start_point is not None:                                                                                            # Supply optional solver hints, not extra constraints.
            vertices, weights = self._simplex_weights(*start_point)                                                            # Find the starting guess's corners and shares.
            selected = np.flatnonzero(np.all(np.sort(simplices, axis=1) == np.sort(vertices), axis=1))                         # Find that tetrahedron among the retained pieces.
            if selected.size:                                                                                                  # Apply hints only if that piece is still available.
                for simplex_index, variable in choose.items():                                                                 # Set starting values for piece selections.
                    variable.Start = float(simplex_index == selected[0])                                                       # Suggest selecting only the starting piece.
                start_weights = dict(zip(vertices, weights))                                                                   # Match starting corner indices to their shares.
                for vertex, variable in blend.items():                                                                         # Set starting values for all corner shares.
                    variable.Start = start_weights.get(vertex, 0.0)                                                            # Suggest zero for corners outside the starting piece.
                fuel.Start = duration_seconds * self.interpolate(*start_point)                                                 # Suggest fuel consistent with the starting conditions.
        return fuel                                                                                                            # Return the constrained fuel variable for use in the objective.

    def validation_samples(self, count=500, seed=16):                                                                          # Compare the approximation with OpenAP at random points.
        """Return reproducible held-out points and approximation errors."""
        import pandas as pd                                                                                                    # Arrange the comparison as a table.

        random = np.random.default_rng(seed)                                                                                   # Use a fixed seed so checks can be repeated.
        points = np.column_stack([random.uniform(axis[0], axis[-1], count) for axis in self.axes])                             # Sample conditions throughout the table's range.
        exact = self.exact(points[:, 0], points[:, 1], points[:, 2])                                                           # Calculate original OpenAP flow at those points.
        estimated = np.array([self.interpolate(*point) for point in points])                                                   # Calculate PWL flow at the same points.
        return pd.DataFrame({                                                                                                  # Return inputs, both predictions, and their differences.
            "speed_mps": points[:, 0], "altitude_m": points[:, 1], "vertical_mps": points[:, 2],                               # Save each test point's conditions.
            "openap_kg_s": exact, "pwl_kg_s": estimated,                                                                       # Save both fuel-flow estimates in kg/s.
            "relative_error_percent": 100 * (estimated - exact) / np.maximum(exact, 1e-9),                                     # Positive means PWL is higher; guard against division by zero.
        })                                                                                                                     # Finish the validation table.


def write_fuel_report(table, waypoints, flights, solved_fuel, output_dir, show_plot=True):                                     # Report fuel on the original solved schedule.
    """Reevaluate the SAME timed segments, without NLP retiming or double-counting."""
    import pandas as pd                                                                                                        # Build and save interval and flight summary tables.

    rows = []                                                                                                                  # Collect one report row per flight interval.
    for flight_index, flight in flights.iterrows():                                                                            # Process each flight in the model's order.
        prefix = f"f{flight_index + 1}"                                                                                        # Find this flight's columns, such as f1_x.
        for step in range(1, len(waypoints)):                                                                                  # Each pair of neighbouring waypoints defines one interval.
            before, after = waypoints.iloc[step - 1], waypoints.iloc[step]                                                     # Read the interval's start and end.
            active = before[f"{prefix}_status"] == 1                                                                           # Count fuel if flying at the start, including the arrival interval.
            duration = float(after["t"] - before["t"])                                                                         # Calculate the interval length in seconds.
            speed = float(np.hypot(after[f"{prefix}_x"] - before[f"{prefix}_x"],                                               # Use east displacement for exact horizontal distance.
                                   after[f"{prefix}_y"] - before[f"{prefix}_y"]) / duration)                                   # Include north displacement and divide distance by time.
            altitude = float((before[f"{prefix}_z"] + after[f"{prefix}_z"]) / 2)                                               # Use the average endpoint altitude in metres.
            vertical = float((after[f"{prefix}_z"] - before[f"{prefix}_z"]) / duration)                                        # Use altitude change per second; descent is negative.
            model_speed = float(after.get(f"{prefix}_fuel_speed_mps", speed))                                                  # Read the solver's approximate speed, or fall back to true speed.
            predicted = float(solved_fuel.get((flight_index, step), 0.0))                                                      # Read solved PWL fuel; omitted intervals default to zero.
            reference = float(table.exact(speed, altitude, vertical)) * duration if active else 0.0                            # Evaluate OpenAP using true speed on the same interval.
            independent = table.interpolate(model_speed, altitude, vertical) * duration if active else 0.0                     # Recheck the PWL calculation without Gurobi.
            if not np.isclose(predicted, independent, atol=0.02, rtol=1e-5):                                                   # Allow only small numerical differences in the PWL check.
                raise ValueError(f"Fuel interpolation/accounting mismatch for {flight['acId']} step {step}.")                  # Stop if solved fuel disagrees with its own approximation.
            rows.append({                                                                                                      # Save this interval's conditions and fuel results.
                "acId": flight["acId"], "step": step, "active": bool(active),                                                  # Identify the flight, interval, and whether it is flown.
                "start_t_seconds": float(before["t"]), "end_t_seconds": float(after["t"]),                                     # Save the original interval timing.
                "flight_seconds": duration if active else 0.0,                                                                 # Exclude waiting and post-arrival time from flight duration.
                "speed_mps": speed, "fuel_model_speed_mps": model_speed,                                                       # Keep true and approximate speeds for comparison.
                "altitude_m": altitude, "vertical_mps": vertical,                                                              # Record mean altitude and climb or descent rate.
                "pwl_fuel_kg": predicted, "openap_fuel_kg": reference,                                                         # Store both fuel estimates in kilograms.
                "relative_error_percent": 100 * (predicted - reference) / reference if reference else 0.0,                     # Report approximation error, not fuel savings.
            })                                                                                                                 # Finish this interval's row.
    segments = pd.DataFrame(rows)                                                                                              # Turn all interval rows into a table.
    summary = segments.groupby("acId", sort=False)[["flight_seconds", "pwl_fuel_kg", "openap_fuel_kg"]].sum()                  # Add up time and fuel for each flight.
    summary["relative_error_percent"] = 100 * (summary["pwl_fuel_kg"] - summary["openap_fuel_kg"]) / summary["openap_fuel_kg"] # Compare each flight's total PWL fuel with OpenAP.
    segments.to_csv(output_dir / "fuel_segments.csv", index=False)                                                             # Save the detailed interval results.
    summary.to_csv(output_dir / "fuel_summary.csv")                                                                            # Save flight totals, including aircraft IDs in the index.
    print("\nFuel on the v16 schedule (kg), with fixed mass and no wind:")                                                     # Label the comparison and its assumptions.
    print(summary.round(3).to_string())                                                                                        # Print flight totals to three decimal places.
    if show_plot:                                                                                                              # Create and display a comparison plot only when requested.
        import matplotlib.pyplot as plt                                                                                        # Load plotting tools when needed.

        figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")                                                     # Make a plot with space for its labels.
        summary[["pwl_fuel_kg", "openap_fuel_kg"]].plot.bar(                                                                   # Draw paired fuel bars for each flight.
            ax=axis, color=["#267a78", "#c24b38"], rot=30                                                                      # Use separate colours and tilt flight labels for readability.
        )                                                                                                                      # Finish drawing the bars.
        axis.set_title("V16 Fuel: Identical Waypoints and Timing")                                                             # Make clear that both estimates use the same schedule.
        axis.set_xlabel("Flight")                                                                                              # Label the horizontal axis.
        axis.set_ylabel("Fuel consumed (kg)")                                                                                  # Label the vertical axis and its units.
        figure.savefig(output_dir / "fuel_comparison.png", dpi=160)                                                            # Save the comparison image.
        plt.show()                                                                                                             # Display the plot; this may wait until its window is closed.
    return segments, summary                                                                                                   # Return the detailed and per-flight results to the caller.
