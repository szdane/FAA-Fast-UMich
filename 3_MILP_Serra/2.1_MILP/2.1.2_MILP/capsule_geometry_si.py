"""Metre-based capsule constraints with bounded unit directions and indicators.

Unlike the slope formulation, neither direction component can blow up near
an axis. All conditional inequalities have units of metres. Nonlinear
equalities define auxiliary variables; indicators themselves remain linear.
"""
import math

from gurobipy import GRB


def build_segment(model, dx, dy, max_length, tag, eps=1e-3):
    length = model.addVar(ub=max_length, name=f"length_{tag}")
    east = model.addVar(lb=-1, ub=1, name=f"unit_east_{tag}")
    north = model.addVar(lb=-1, ub=1, name=f"unit_north_{tag}")
    model.addConstr(east * east + north * north == 1, name=f"unit_norm_{tag}")
    model.addConstr(length * east == dx, name=f"segment_east_{tag}")
    model.addConstr(length * north == dy, name=f"segment_north_{tag}")
    moves = model.addVar(vtype=GRB.BINARY, name=f"moves_{tag}")
    model.addConstr((moves == 1) >> (length >= eps))
    model.addConstr((moves == 0) >> (length <= eps))
    return east, north, moves


def add_waypoint(model, waypoint, point, delta, enabled, tag, shape="octagon"):
    """Outside a circle or its circumscribed square/octagon, when enabled."""
    dx, dy = point[0] - waypoint[0], point[1] - waypoint[1]
    if shape == "circle":
        distance = model.addVar(name=f"distance_{tag}")
        model.addConstr(distance * distance == dx * dx + dy * dy)
        model.addConstr((enabled == 1) >> (distance >= delta))
        return
    faces = [dx, -dx, dy, -dy]
    if shape == "octagon":
        faces += [(dx + dy) / math.sqrt(2), (-dx - dy) / math.sqrt(2),
                  (dx - dy) / math.sqrt(2), (-dx + dy) / math.sqrt(2)]
    elif shape != "square":
        raise ValueError(f"Unknown waypoint shape: {shape}")
    selected = model.addVars(len(faces), vtype=GRB.BINARY, name=f"faces_{tag}")
    model.addConstr(selected.sum() == enabled)
    for face, value in enumerate(faces):
        model.addConstr((selected[face] == 1) >> (value >= delta))


def add_capsule(model, geometry, near, far, point, delta, enabled, tag,
                shape="octagon"):
    east, north, moves = geometry
    cases = model.addVars(4, vtype=GRB.BINARY, name=f"cases_{tag}")
    model.addConstr(cases.sum() == enabled)
    model.addConstr(cases[0] + cases[1] <= moves)
    # Project offsets onto unit vectors: these quantities are metres,
    # whereas unnormalised dot/cross products would have units of m^2.
    perp = model.addVar(lb=-GRB.INFINITY, name=f"perp_m_{tag}")
    along_far = model.addVar(lb=-GRB.INFINITY, name=f"along_far_m_{tag}")
    along_near = model.addVar(lb=-GRB.INFINITY, name=f"along_near_m_{tag}")
    model.addConstr(perp == east * (point[1] - near[1]) - north * (point[0] - near[0]))
    model.addConstr(along_far == east * (point[0] - far[0]) + north * (point[1] - far[1]))
    model.addConstr(along_near == east * (point[0] - near[0]) + north * (point[1] - near[1]))
    model.addConstr((cases[0] == 1) >> (perp >= delta))
    model.addConstr((cases[1] == 1) >> (perp <= -delta))
    model.addConstr((cases[2] == 1) >> (along_far >= 0))
    model.addConstr((cases[3] == 1) >> (along_near <= 0))
    add_waypoint(model, far, point, delta, cases[2], f"far_{tag}", shape)
    add_waypoint(model, near, point, delta, cases[3], f"near_{tag}", shape)
