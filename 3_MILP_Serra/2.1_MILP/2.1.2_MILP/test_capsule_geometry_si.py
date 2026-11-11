"""Solver checks at the actual 152.4 m separation scale.

Run: python3 -m unittest test_capsule_geometry_si.py
"""
import math
import unittest

from gurobipy import Env, GRB, Model

from capsule_geometry_si import add_capsule, build_segment
from milp_multiple_Debug_SI import latlon_to_xy, xy_to_latlon


class SICapsuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env(params={"OutputFlag": 0})

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def feasible(self, dx, dy, px, py, enabled=1, shape="octagon"):
        with Model(env=self.env) as model:
            model.Params.NonConvex = 2
            vx = model.addVar(lb=dx, ub=dx)
            vy = model.addVar(lb=dy, ub=dy)
            geometry = build_segment(model, vx, vy, max(1, math.hypot(dx, dy)), "test")
            active = model.addVar(vtype=GRB.BINARY, lb=enabled, ub=enabled)
            add_capsule(model, geometry, (0, 0), (dx, dy), (px, py),
                        152.4, active, "test", shape)
            model.optimize()
            self.assertIn(model.Status, (GRB.OPTIMAL, GRB.INFEASIBLE))
            return model.Status == GRB.OPTIMAL

    def test_directions_strip_and_endcaps(self):
        for dx, dy in [(10000, 10000), (-10000, 10000), (-10000, -10000),
                       (10000, -10000), (10000, 0), (-10000, 0),
                       (0, 10000), (0, -10000), (0.01, 10000), (10000, 0.01)]:
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            # Well inside/outside, away from face/tolerance boundaries.
            for along, perp, expected in [(length/2, 100, False), (length/2, 200, True),
                                           (-100, 0, False), (-200, 0, True),
                                           (length+100, 0, False), (length+200, 0, True)]:
                with self.subTest(direction=(dx, dy), along=along, perp=perp):
                    self.assertEqual(self.feasible(dx, dy, ux*along-uy*perp,
                                                   uy*along+ux*perp), expected)

    def test_stationary(self):
        for shape in ("circle", "square", "octagon"):
            with self.subTest(shape=shape):
                self.assertFalse(self.feasible(0, 0, 0, 0, shape=shape))
                self.assertFalse(self.feasible(0, 0, 100, 0, shape=shape))
                self.assertTrue(self.feasible(0, 0, 200, 0, shape=shape))

    def test_disabled_branch_allows_overlap(self):
        self.assertTrue(self.feasible(10000, 0, 5000, 0, enabled=0))
        self.assertTrue(self.feasible(0, 0, 0, 0, enabled=0))

    def test_octagon_diagonal_faces(self):
        self.assertFalse(self.feasible(0, 0, 100, 100))
        self.assertTrue(self.feasible(0, 0, 120, 120))

    def test_projection_round_trip(self):
        for lat, lon in [(42.2125, -83.3534), (41.673, -82.9405), (42.915, -83.602)]:
            x, y = latlon_to_xy(lat, lon)
            actual_lat, actual_lon = xy_to_latlon(x, y)
            self.assertAlmostEqual(float(actual_lat), lat, places=7)
            self.assertAlmostEqual(float(actual_lon), lon, places=7)


if __name__ == "__main__":
    unittest.main()
