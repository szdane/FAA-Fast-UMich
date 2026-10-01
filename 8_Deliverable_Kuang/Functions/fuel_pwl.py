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

    def add_segment(self, model, speed_mps, altitude_m, vertical_mps, inactive,                                                # Accept solver expressions for one flight interval.
                    duration_seconds, name, altitude_bounds=None):                                                             # Accept duration.
        """Charge a full flown interval; inactive is the PREVIOUS arrival flag."""

        # 1. Validate the segment duration.
        if duration_seconds <= 0:                                                                                              # Fuel needs an interval with positive length.
            raise ValueError("Segment duration must be positive.")                                                             # Reject zero or negative durations.

        # 2. Keep tetrahedra that overlap the reachable altitude range.
        simplices = self.simplices                                                                                             # Begin with every tetrahedron in the table.
        if altitude_bounds is not None:                                                                                        # Reduce model size when reachable altitude limits are known.
            heights = self.points[simplices, 1]                                                                                # Get the four corner altitudes of each tetrahedron.
            lower, upper = altitude_bounds                                                                                     # Read the reachable altitude range for this interval.
            simplices = simplices[(heights.max(axis=1) >= lower) & (heights.min(axis=1) <= upper)]                             # Keep pieces whose altitude ranges overlap it.

        # 3. Create tetrahedron selectors and interpolation weights, active only in flight.
        vertex_ids = np.unique(simplices)                                                                                      # Collect the distinct corners used by the remaining pieces.
        choose = model.addVars(len(simplices), vtype=gp.GRB.BINARY, name=name + "_tet")                                        # Give each tetrahedron an on/off selection variable.
        blend = model.addVars(vertex_ids.tolist(), lb=0.0, ub=1.0, name=name + "_blend")                                       # Give each corner a share between zero and one.
        model.addConstr(choose.sum() == 1 - inactive, name=name + "_one_tet")                                                  # Choose one piece while flying, none after arrival.
        model.addConstr(blend.sum() == 1 - inactive, name=name + "_unit_blend")                                                # Shares total one while flying and zero when inactive.

        # 4. Allow nonzero weights only at corners of the selected tetrahedron.
        incident = {vertex: [] for vertex in vertex_ids}                                                                       # Track which tetrahedra contain each corner.
        for simplex_index, simplex in enumerate(simplices):                                                                    # Visit each available tetrahedron.
            for vertex in simplex:                                                                                             # Visit its four corners.
                incident[vertex].append(choose[simplex_index])                                                                 # Link this corner to this piece's selection variable.
        for vertex in vertex_ids:                                                                                              # Restrict every corner's interpolation share.
            model.addConstr(blend[vertex] <= gp.quicksum(incident[vertex]), name=f"{name}_local_{vertex}")                     # Only corners of the selected piece may have nonzero shares.

        # 5. Match interpolated speed, altitude, and vertical speed to the flight segment.
        states = [gp.quicksum(self.points[vertex, dimension] * blend[vertex] for vertex in vertex_ids)                         # Blend corner conditions with the same shares as fuel.
                  for dimension in range(3)]                                                                                   # Build speed, altitude, and vertical speed in that order.
        model.addConstr(speed_mps == states[0], name=name + "_speed")                                                          # Match interval speed; inactive intervals must have zero speed.
        model.addConstr(vertical_mps == states[2], name=name + "_vertical_speed")                                              # Match vertical speed; inactive intervals must have zero vertical speed.
        model.addGenConstrIndicator(inactive, 0, altitude_m == states[1], name=name + "_altitude")                             # Match altitude only while flying, not after arrival.

        # 6. Multiply interpolated fuel flow by duration and return fuel consumed in kg.
        fuel = model.addVar(lb=0.0, ub=duration_seconds * self.flows.max(), name=name + "_kg")                                 # Create interval fuel in kg with a table-based upper bound.
        model.addConstr(fuel == duration_seconds * gp.quicksum(                                                                # Fuel equals the interval length times blended flow.
            self.flows[vertex] * blend[vertex] for vertex in vertex_ids                                                        # Weight each corner's kg/s value by its share.
        ), name=name + "_integrate")                                                                                           # This also forces zero fuel when all shares are zero.
        return fuel                                                                                                            # Return the constrained fuel variable for use in the objective.
