###################################
# modified from OpenAP.top.cruise #
###################################
#.conda/Lib/site-packeges/openap/top/cruise

import warnings
from math import pi

import casadi as ca

import numpy as np
import openap.casadi as oc
import pandas as pd
from openap.extra.aero import fpm, ft, kts

from Functions.fuel_emission_analysis_base import Base #from openap.top.base import Base


class Cruise_with_Multi_Waypoints(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fix_mach = False
        self.fix_alt = False
        self.fix_track = False

        ############################################ Modified By Kuang ############################################
        self.allow_descent = True #False
        # enabling descent at pre-TRACON region
        ###########################################################################################################


    def fix_mach_number(self):
        self.fix_mach = True

    def fix_cruise_altitude(self):
        self.fix_alt = False

    def fix_track_angle(self):
        self.fix_track = True

    def allow_cruise_descent(self):
        self.allow_descent = True

    def init_conditions(self, **kwargs):
        """Initialize direct collocation bounds and guesses."""

        # Convert lat/lon to cartisian coordinates.
        xp_0, yp_0 = self.proj(self.lon1, self.lat1)
        xp_f, yp_f = self.proj(self.lon2, self.lat2)

        ############################################ Modified By Kuang ############################################
        z_0 = self.alt1 * ft
        z_f = self.alt2 * ft
        self.z_0 = z_0
        self.z_f = z_f
        # convert altitude from ft to meters
        ###########################################################################################################

        ############################################ Modified By Kuang ############################################
        x_min = min(xp_0, xp_f) - 10000
        x_max = max(xp_0, xp_f) + 10000
        y_min = min(yp_0, yp_f) - 300000 #10000
        y_max = max(yp_0, yp_f) + 300000 #10000
        # increase playground size to let generated optimized trajectory not be restricted by bouaries of small 
        # optimization playground
        ###########################################################################################################


        ts_min = 0
        ts_max = 24 * 3600

        h_max = kwargs.get("h_max", self.aircraft["limits"]["ceiling"])
        h_min = kwargs.get("h_min", 15_000 * ft)

        hdg = oc.aero.bearing(self.lat1, self.lon1, self.lat2, self.lon2)
        psi = hdg * pi / 180

        # Initial conditions - Lower upper bounds
        self.x_0_lb = [xp_0, yp_0, h_min, self.mass_init, ts_min]
        self.x_0_ub = [xp_0, yp_0, h_max, self.mass_init, ts_min]

        # Final conditions - Lower and upper bounds
        self.x_f_lb = [xp_f, yp_f, h_min, self.oew, ts_min]
        self.x_f_ub = [xp_f, yp_f, h_max, self.mass_init, ts_max]

        # States - Lower and upper bounds
        self.x_lb = [x_min, y_min, h_min, self.oew, ts_min]
        self.x_ub = [x_max, y_max, h_max, self.mass_init, ts_max]

        # Control init - lower and upper bounds
        self.u_0_lb = [0.5, -500 * fpm, psi - pi / 4]
        self.u_0_ub = [self.mach_max, 500 * fpm, psi + pi / 4]

        # Control final - lower and upper bounds
        self.u_f_lb = [0.5, -500 * fpm, psi - pi / 4]
        self.u_f_ub = [self.mach_max, 500 * fpm, psi + pi / 4]

        # Control - Lower and upper bound
        self.u_lb = [0.5, -500 * fpm, psi - pi / 2]
        self.u_ub = [self.mach_max, 500 * fpm, psi + pi / 2]

        ############################################ Modified By Kuang ############################################
        # Initial guess - states
        middle_waypoints = kwargs.get("middle_waypoints", [])
        full_waypoints = [self.origin] + middle_waypoints + [self.destination]
        self.x_guess = self.initial_guess_through_waypoints(full_waypoints)
        # self.x_guess = self.initial_guess()
        # middle_waypoints = kwargs.get("middle_waypoints", [])
        # full_waypoints = [self.origin] + middle_waypoints + [self.destination]
        #print(full_waypoints)
        #full_waypoints = [(self.lat1, self.lon1, self.alt1, )] + middle_waypoints + [(self.lat2, self.lon2, self.alt2)]
        #self.x_guess = self.initial_guess_through_waypoints(full_waypoints)
        ###########################################################################################################

        # Initial guess - controls
        self.u_guess = [0.7, 0, psi]

    def trajectory(self, objective="time", **kwargs) -> pd.DataFrame:
        """
        Computes the optimal trajectory for the aircraft based on the given objective.

        Parameters:
        - objective (str): The objective of the optimization, default is "fuel".
        - **kwargs: Additional keyword arguments.
            - max_fuel (float): Customized maximum fuel constraint.
            - initial_guess (pd.DataFrame): Initial guess for the trajectory. This is
                usually a exsiting flight trajectory.
            - return_failed (bool): If True, returns the DataFrame even if the
                optimization fails. Default is False.

        Returns:
        - pd.DataFrame: A DataFrame containing the optimized trajectory.

        Note:
        - The function uses CasADi for symbolic computation and optimization.
        - The constraints and bounds are defined based on the aircraft's performance
            and operational limits.
        """


        ############################################ Modified By Kuang ############################################
        # Parse optional middle waypoint
        middle_waypoints = kwargs.get("middle_waypoints", [])  # list of tuples [(lat1, lon1, alt1), (lat2, lon2, alt2), ...]
        middle_radius = kwargs.get("middle_radius", 10_000)
        margin_alt = kwargs.get("middle_alt_margin", 10 * ft) # altitude tolerance # kwargs are defined in the brackets in function(...)
        # Add middle waipoint into to keyword arguments (kwargs) dictionary, so that optimizer can find middle 
        # point information from the kwargs dictionar later
        ###########################################################################################################


        # Initialize conditions
        self.init_conditions(**kwargs)

        self.init_model(objective, **kwargs)
        
        # arguments passed init_condition to overwright h_min and h_max
        self.init_conditions(**kwargs)

        self.init_model(objective, **kwargs)

        customized_max_fuel = kwargs.get("max_fuel", None)

        initial_guess = None #kwargs.get("initial_guess", None)
        if initial_guess is not None:
            self.x_guess = self.initial_guess(initial_guess)

        return_failed = kwargs.get("return_failed", False)

        C, D, B = self.collocation_coeff()

        # Start with an empty NLP
        w = []  # Containing all the states & controls generated
        w0 = []  # Containing the initial guess for w
        lbw = []  # Lower bound constraints on the w variable
        ubw = []  # Upper bound constraints on the w variable
        J = 0  # Objective function
        g = []  # Constraint function
        lbg = []  # Constraint lb value
        ubg = []  # Constraint ub value

        # For plotting x and u given w
        X = []
        U = []

        # Apply initial conditions
        # Create Xk such that it is the same length as x
        nstates = self.x.shape[0]
        Xk = ca.MX.sym("X0", nstates, self.x.shape[1])
        w.append(Xk)
        lbw.append(self.x_0_lb)
        ubw.append(self.x_0_ub)
        w0.append(self.x_guess[0])
        X.append(Xk)

        # Formulate the NLP
        for k in range(self.nodes):
            # New NLP variable for the control
            Uk = ca.MX.sym("U_" + str(k), self.u.shape[0])
            U.append(Uk)
            w.append(Uk)

            if k == 0:
                lbw.append(self.u_0_lb)
                ubw.append(self.u_0_ub)
            elif k == self.nodes - 1:
                lbw.append(self.u_f_lb)
                ubw.append(self.u_f_ub)
            else:
                lbw.append(self.u_lb)
                ubw.append(self.u_ub)

            w0.append(self.u_guess)

            # State at collocation points
            Xc = []
            for j in range(self.polydeg):
                Xkj = ca.MX.sym("X_" + str(k) + "_" + str(j), nstates)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
                w0.append(self.x_guess[k])

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1, self.polydeg + 1):
                # Expression for the state derivative at the collocation point
                xpc = C[0, j] * Xk
                for r in range(self.polydeg):
                    xpc = xpc + C[r + 1, j] * Xc[r]

                # Append collocation equations
                fj, qj = self.func_dynamics(Xc[j - 1], Uk)
                g.append(self.dt * fj - xpc)
                lbg.append([0] * nstates)
                ubg.append([0] * nstates)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                # J = J + B[j] * qj * dt
                J = J + B[j] * qj

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym("X_" + str(k + 1), nstates)
            w.append(Xk)
            X.append(Xk)

            # lbw.append(x_lb)
            # ubw.append(x_ub)

            if k < self.nodes - 1:
                lbw.append(self.x_lb)
                ubw.append(self.x_ub)
            else:
                # Final conditions
                lbw.append(self.x_f_lb)
                ubw.append(self.x_f_ub)

            w0.append(self.x_guess[k])

            # Add equality constraint
            g.append(Xk_end - Xk)
            lbg.append([0] * nstates)
            ubg.append([0] * nstates)

        w.append(self.ts_final)
        lbw.append([0])
        ubw.append([ca.inf])
        w0.append([self.range * 1000 / 200])


        ############################################ Modified By Kuang ############################################        
        from geopy.distance import geodesic

        def arc_length_along_waypoints(waypoints):
            total = 0
            for i in range(1, len(waypoints)):
                total += geodesic(waypoints[i-1], waypoints[i]).meters
            return total

        # Step 0: Compute total geodesic distance from origin to destination
        all_waypoints = [ (self.lat1, self.lon1) ] + \
                [ (lat, lon) for lat, lon, _, _ in middle_waypoints ] + \
                [ (self.lat2, self.lon2) ]

        total_arc_length = arc_length_along_waypoints(all_waypoints)

        # Step 1: Rebuild the loop for constraint placement
        for i, (lat_wp, lon_wp, alt_wp_ft, t_wp) in enumerate(middle_waypoints):
            # 1. Find its index in the full path (offset by 1 for origin)
            full_index = i + 1

            # 2. Compute cumulative arc length to this point
            arc_dist_to_wp = arc_length_along_waypoints(all_waypoints[:full_index + 1])  # +1 to include this point

            # 3. Compute ratio
            ratio_along_path = arc_dist_to_wp / total_arc_length
            ratio_along_path = min(ratio_along_path, 0.999)  # Avoid going out of bounds

            # 4. Assign node index
            wp_index = int(ratio_along_path * (self.nodes - 1))
            self.wp_node_indices[i] = wp_index # record the critical waypoitns that are constrained during optimization
            
            # Project lat/lon to Cartesian coordinates
            x_wp, y_wp = self.proj(lon_wp, lat_wp)
            alt_wp_m = alt_wp_ft * ft  # convert to meters

            # Position constraints (x, y)
            dx = X[wp_index][0] - x_wp
            dy = X[wp_index][1] - y_wp
            dist_squared = dx ** 2 + dy ** 2
            g.append(dist_squared)
            lbg.append([0])
            ubg.append([middle_radius ** 2])

            # Altitude constraint (z)
            # margin_alt = 1000 * ft  # ±1000 ft tolerance
            g.append(X[wp_index][2])
            lbg.append([alt_wp_m - margin_alt])
            ubg.append([alt_wp_m + margin_alt])

            # Altitude constraint on initial node
            g.append(X[0][2])
            lbg.append([self.z_0 - margin_alt])
            ubg.append([self.z_0 + margin_alt])

            # Altitude constraint on final node
            g.append(X[self.nodes][2])
            lbg.append([self.z_f - margin_alt])
            ubg.append([self.z_f + margin_alt])

            # time_margin = 60  # seconds of tolerance (increased from 20 to reduce zigzag)
            # g.append(X[wp_index][4])
            # lbg.append([t_wp - time_margin])
            # ubg.append([t_wp + time_margin])

            # # Soft time constraint penalty
        #     t_node = X[wp_index][4]  # time at that node
        #     time_penalty_weight = 1e4  # You can tune this, eg. 1e-3
        #     J += time_penalty_weight * (t_node - t_wp) ** 2
        # # assign waypoint constraint with proximity, and assign them at corresponding node in the trajectory
        ###########################################################################################################


        # aircraft performane constraints
        for k in range(self.nodes):
            S = self.aircraft["wing"]["area"]
            mass = X[k][3]
            v = oc.aero.mach2tas(U[k][0], X[k][2])
            tas = v / kts
            alt = X[k][2] / ft
            rho = oc.aero.density(X[k][2])
            thrust_max = self.thrust.cruise(tas, alt)

            # max_thrust * 95% > drag (5% margin)
            g.append(thrust_max * 0.95 - self.drag.clean(mass, tas, alt))
            lbg.append([0])
            ubg.append([ca.inf])

            # max lift * 80% > weight (20% margin)
            drag_max = thrust_max * 0.9
            cd_max = drag_max / (0.5 * rho * v**2 * S + 1e-10)
            cd0 = self.drag.polar["clean"]["cd0"]
            ck = self.drag.polar["clean"]["k"]
            cl_max = ca.sqrt(ca.fmax(1e-10, (cd_max - cd0) / ck))
            L_max = cl_max * 0.5 * rho * v**2 * S
            g.append(L_max * 0.8 - mass * oc.aero.g0)
            lbg.append([0])
            ubg.append([ca.inf])
        
        ############################################ Modified By Kuang ############################################
        # # ==================== Avoid TRACON Polygon Constraint ====================
        # from shapely.geometry import Polygon, Point

        # tracon_polygon = kwargs.get("forbidden_region", None)

        # if tracon_polygon is not None:
        #     # Project polygon into Cartesian coordinates
        #     poly_lon, poly_lat = tracon_polygon.exterior.xy
        #     poly_x, poly_y = self.proj(poly_lon, poly_lat)  # Same projection as X, Y
        #     tracon_poly_xy = Polygon(zip(poly_x, poly_y))

        #     avoid_buffer = 3000  # meters away from boundary
        #     # Create polygon edges as line segments
        #     edges = list(zip(poly_x, poly_y, poly_x[1:] + poly_x[:1], poly_y[1:] + poly_y[:1]))
            
        #     for k in range(self.nodes):
        #         xk = X[k][0]
        #         yk = X[k][1]
                
        #         for x1, y1, x2, y2 in edges:
        #             # Use cross product for signed distance from line
        #             dx = x2 - x1
        #             dy = y2 - y1
        #             norm_sq = dx**2 + dy**2 + 1e-6
        #             proj = ((xk - x1)*dx + (yk - y1)*dy) / norm_sq
        #             proj = ca.fmin(ca.fmax(proj, 0), 1)
        #             x_proj = x1 + proj * dx
        #             y_proj = y1 + proj * dy
        #             dist_sq = (xk - x_proj)**2 + (yk - y_proj)**2
        #             g.append(dist_sq)
        #             lbg.append([(avoid_buffer)**2])
        #             ubg.append([ca.inf])
        ##########################################################################################################

        # ts and dt should be consistent
        for k in range(self.nodes - 1):
            g.append(X[k + 1][4] - X[k][4] - self.dt)
            lbg.append([0]) #lbg.append([-1])
            ubg.append([0]) #ubg.append([1])

        # # smooth Mach number change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][0] - U[k][0])
        #     lbg.append([-0.2])
        #     ubg.append([0.2])  # to be tunned

        # # smooth vertical rate change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][1] - U[k][1])
        #     lbg.append([-500 * fpm])
        #     ubg.append([500 * fpm])  # to be tunned


        ############################################ Modified By Kuang ############################################ 
        # # smooth heading change
        # for k in range(self.nodes - 1):
        #     g.append(U[k + 1][2] - U[k][2])
        #     # lbg.append([-45 * pi / 180]) 
        #     # ubg.append([45 * pi / 180]) # tunned smooth heading rate change
        #     lbg.append([-15 * pi / 180])
        #     ubg.append([15 * pi / 180]) 
    
        # # Altitude Constraint on first node
        # #alt_target_first = 35000 * ft  # desired altitude in meters
        # # margin_first = 1000 * ft # soft constraint (within a range, e.g. ±1000 ft)
        # g.append(X[0][2]) 
        # lbg.append([self.z_0 - margin_alt])
        # ubg.append([self.z_0 + margin_alt])

        # # Altitude constraint on final node
        # # alt_target_final = 15000 * ft  # desired altitude in meters
        # # margin_final = 2000 * ft # soft constraint (within a range, e.g. ±1000 ft)
        # g.append(X[self.nodes][2])
        # lbg.append([self.z_f - margin_alt])
        # ubg.append([self.z_f + margin_alt])

        # # OPTIONAL: Hard constraint to prevent climbing (uncomment to enforce strict descent)
        # # enforce_monotonic_descent = kwargs.get("enforce_monotonic_descent", True)
        # # if enforce_monotonic_descent:
        # #     for k in range(self.nodes - 1):
        # #         # Force altitude to decrease or stay same: X[k+1][2] <= X[k][2]
        # #         g.append(X[k + 1][2] - X[k][2])
        # #         lbg.append([-ca.inf])  # Can decrease by any amount
        # #         ubg.append([0])         # Cannot increase (max change = 0)

        # # --- Time constraint on final node ---
        # final_time = kwargs.get("final_time", None) # final_time = 1200 
        # if final_time is not None:
        #     final_time_margin = 5 #kwargs.get("final_time_margin", 20)  # seconds tolerance
        #     g.append(X[self.nodes][4])  # time component of final node
        #     lbg.append([final_time - final_time_margin])
        #     ubg.append([final_time + final_time_margin])
            
        # Added / Tunned Optimization Constraints
        ###########################################################################################################


        ############################################ Modified By Kuang ############################################ 
        # #Penalize heading rate change (optional but recommended for smoother turns)
        # penalty_weight_heading_rate = 1e-6d # small weight, tune as needed
        # for k in range(self.nodes - 1):
        #     dpsi = U[k + 1][2] - U[k][2]
        #     J += penalty_weight_heading_rate * (dpsi ** 2)

        # # Penalize rapid change in heading (second derivative of heading angle)
        # penalty_weight_heading_accel = 5 #5  # tune as needed
        # for k in range(1, self.nodes - 1):
        #     d2psi = U[k + 1][2] - 2 * U[k][2] + U[k - 1][2]
        #     J += penalty_weight_heading_accel * (d2psi ** 2)
        # # NEW: Penalize position acceleration (2nd derivative of x, y) to reduce zigzag
        # penalty_weight_position_accel = 1000  # tune as needed (higher = smoother but potentially slower)
        # for k in range(1, self.nodes - 1):
        #     # x acceleration
        #     d2x = X[k + 1][0] - 2 * X[k][0] + X[k - 1][0]
        #     J += penalty_weight_position_accel * (d2x ** 2)
        #     # y acceleration  
        #     d2y = X[k + 1][1] - 2 * X[k][1] + X[k - 1][1]
        #     J += penalty_weight_position_accel * (d2y ** 2)
        # # penalize too small vertical rate (DESCENT only - we WANT descent)
        # min_vrate = 1000 * fpm  # ~1.52 m/s — tune as needed
        # penalty_weight_vrate = 0.2  # Tune based on importance
        # for k in range(self.nodes):
        #     vrate = U[k][1]
        #     # Only penalize if NOT descending enough (vrate should be negative for descent)
        #     J += penalty_weight_vrate * ca.fmax(0, min_vrate + vrate) ** 2  # Changed: penalize if vrate > -min_vrate

        # # FIXED: Penalize CLIMBING (positive vertical rate)
        # penalty_weight_climb = 5000  # Large penalty for climbing
        # for k in range(self.nodes):
        #     vrate = U[k][1]
        #     # Heavily penalize positive vertical rates (climbing)
        #     J += penalty_weight_climb * ca.fmax(0, vrate) ** 2
        
        # # FIXED: Penalize altitude increases between nodes
        # penalty_weight_alt_increase = 10000  # Very large penalty
        # for k in range(self.nodes - 1):
        #     # Penalize if next altitude is higher than current
        #     alt_increase = X[k + 1][2] - X[k][2]
        #     J += penalty_weight_alt_increase * ca.fmax(0, alt_increase) ** 2
        # # Added / Tunned Optimization Penalties

        # # Encourage shorter flight time (faster route)
        # time_weight = kwargs.get("time_weight", 0.0)  # set >0 in main to prioritize speed
        # J += time_weight * self.ts_final

        # # Smooth Mach changes (1st + 2nd differences)
        # w_dmach = kwargs.get("w_dmach", 500.0)  # increased from 200.0
        # w_d2mach = kwargs.get("w_d2mach", 200.0)  # increased from 50.0

        # for k in range(self.nodes - 1):
        #     dM = U[k + 1][0] - U[k][0]
        #     J += w_dmach * (dM ** 2)

        # for k in range(1, self.nodes - 1):
        #     d2M = U[k + 1][0] - 2 * U[k][0] + U[k - 1][0]
        #     J += w_d2mach * (d2M ** 2)

        # # Smooth vertical speed changes (1st + 2nd differences)
        # w_dvs = kwargs.get("w_dvs", 200.0)  # increased from 5.0
        # w_d2vs = kwargs.get("w_d2vs", 100.0)  # increased from 1.0

        # for k in range(self.nodes - 1):
        #     dVS = U[k + 1][1] - U[k][1]
        #     J += w_dvs * (dVS ** 2)

        # for k in range(1, self.nodes - 1):
        #     d2VS = U[k + 1][1] - 2 * U[k][1] + U[k - 1][1]
        #     J += w_d2vs * (d2VS ** 2)
        ############################################################################################################

        # optional constraints
        if self.fix_mach:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][0] - U[k][0])
                lbg.append([0])
                ubg.append([0])

        if self.fix_alt:
            for k in range(self.nodes - 1):
                g.append(X[k + 1][2] - X[k][2])
                lbg.append([0])
                ubg.append([0])

        if self.fix_track:
            for k in range(self.nodes - 1):
                g.append(U[k + 1][2] - U[k][2])
                lbg.append([0])
                ubg.append([0])

        if not self.allow_descent:
            for k in range(self.nodes):
                g.append(U[k][1])
                lbg.append([0])
                ubg.append([ca.inf])

        # add fuel constraint
        g.append(X[0][3] - X[-1][3])
        lbg.append([0])
        ubg.append([self.fuel_max])

        if customized_max_fuel is not None:
            g.append(X[0][3] - X[-1][3] - customized_max_fuel)
            lbg.append([-ca.inf])
            ubg.append([0])

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        X = ca.horzcat(*X)
        U = ca.horzcat(*U)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create an NLP solver
        nlp = {"f": J, "x": w, "g": g}

        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)
        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        df = self.to_trajectory(ts_final, x_opt, u_opt)

        df_copy = df.copy()

        # check if the optimizer has failed
        if df.altitude.max() < 5000:
            warnings.warn("max altitude < 5000 ft, optimization seems to have failed.")
            return None

        if df is not None:
            final_mass = df.mass.iloc[-1]

            if final_mass < self.oew:
                warnings.warn("final mass condition violated (smaller than OEW).")
                df = None

        if return_failed:
            return df_copy
        
        ############################################ Modified By Kuang ############################################
        df.attrs["wp_node_indices"] = self.wp_node_indices #store critical nodes as global comment
        ###########################################################################################################
        return df
       