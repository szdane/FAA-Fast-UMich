### ### ### ### ### ### ### ### ### ### ### ##
# modified from OpenAP.top.cruise #
### ### ### ### ### ### ### ### ### ### ### ##
#.conda/Lib/site-packeges/openap/top/cruise

import warnings
from math import pi
from pathlib import Path

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

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        self.allow_descent = True #False
        # enabling descent at pre-TRACON region
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


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

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        z_0 = self.alt1 * ft
        z_f = self.alt2 * ft
        self.z_0 = z_0
        self.z_f = z_f
        # convert altitude from ft to meters
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        x_min = min(xp_0, xp_f) - 10000
        x_max = max(xp_0, xp_f) + 10000
        y_min = min(yp_0, yp_f) - 300000 #10000
        y_max = max(yp_0, yp_f) + 300000 #10000
        # increase playground size to let generated optimized trajectory not be restricted by bouaries of small 
        # optimization playground
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


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

        # Min and Max vertical rates. 
        vs_min = kwargs.get("vs_min", -3000 * fpm)   # realistic arrival descent rate (3000 fpm)
        vs_max = kwargs.get("vs_max",   0 * fpm) # No climbing
        # Control init - lower and upper bounds
        self.u_0_lb = [0.5, vs_min, psi - pi / 4]
        self.u_0_ub = [self.mach_max, vs_max, psi + pi / 4]
        # Control final - lower and upper bounds
        self.u_f_lb = [0.5, vs_min, psi - pi / 4]
        self.u_f_ub = [self.mach_max, vs_max, psi + pi / 4]
        # Control - Lower and upper bound
        self.u_lb   = [0.5, vs_min, psi - pi / 2]
        self.u_ub   = [self.mach_max, vs_max, psi + pi / 2]

        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        # Initial guess - states
        middle_waypoints = kwargs.get("middle_waypoints", [])
        full_waypoints = [self.origin] + middle_waypoints + [self.destination]
        # self.x_guess = self.initial_guess() 
        # self.x_guess = self.initial_guess_through_waypoints(full_waypoints)
        self.x_guess = self.initial_guess_smooth_through_waypoints(full_waypoints)
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

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


        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        # Parse optional middle waypoint
        middle_waypoints = kwargs.get("middle_waypoints", [])  # list of tuples [(lat1, lon1, alt1), (lat2, lon2, alt2), ...]
        middle_radius = kwargs.get("middle_radius", 10_000)
        margin_alt = kwargs.get("middle_alt_margin", 10 * ft) # altitude tolerance # kwargs are defined in the brackets in function(...)
        # Add middle waipoint into to keyword arguments (kwargs) dictionary, so that optimizer can find middle 
        # point information from the kwargs dictionar later
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


        # Initialize conditions
        # self.init_conditions(**kwargs)
        # self.init_model(objective, **kwargs)
        # arguments passed init_condition to overwright h_min and h_max
        self.init_conditions(**kwargs)
        self.init_model(objective, **kwargs)

        customized_max_fuel = kwargs.get("max_fuel", None)

        initial_guess = kwargs.get("initial_guess", None) # None #
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

### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
                # Add contribution to quadrature function — disabled; using smoothness objective
                # J = J + B[j] * qj * dt
                # J = J + B[j] * qj
                J = J + B[j] * qj * self.dt
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


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
        w0.append([self.range / 200])

### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        # # Smooth trajectory objective: penalize squared control changes between nodes
        # # Controls: U[k] = [mach (-), vertical_rate (m/s), heading (rad)]
        W_MACH = 100.0     # Mach number smoothness weight
        W_VS   = 1.0       # vertical rate smoothness weight
        W_PSI  = 10000.0   # heading smoothness weight — high to prevent zigzag
        for k in range(self.nodes - 1):
            J += W_MACH * (U[k + 1][0] - U[k][0]) ** 2
            J += W_VS   * (U[k + 1][1] - U[k][1]) ** 2
            J += W_PSI  * (U[k + 1][2] - U[k][2]) ** 2  
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
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

            # Altitude constraint (z) on the waypoint node
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


        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##


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


        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## 
        ## Forced ROC to ZERO and therefore none of these are needed anymore
        # # OPTIONAL: Hard constraint to prevent climbing (uncomment to enforce strict descent)
        # enforce_monotonic_descent = kwargs.get("enforce_monotonic_descent", True)
        # if enforce_monotonic_descent:
        #     for k in range(self.nodes - 1):
        #         # Force altitude to decrease or stay same: X[k+1][2] <= X[k][2]
        #         g.append(X[k + 1][2] - X[k][2])
        #         lbg.append([-ca.inf])  # Can decrease by any amount
        #         ubg.append([0])         # Cannot increase (max change = 0)
        # # Soft penalty: heavily penalize any altitude increase between nodes
        # # ca.fmax(0, dh) is 0 when descending/level, positive when climbing
        # penalty_weight_climb = kwargs.get("penalty_weight_climb", 1e6)
        # for k in range(self.nodes - 1):
        #     dh = X[k + 1][2] - X[k][2]          # altitude change (m), positive = climb
        #     J += penalty_weight_climb * ca.fmax(0, dh) ** 2


        # --- Time constraint on final node ---
        final_time = kwargs.get("final_time")  # flight duration in seconds (NOT UTC time)
        if final_time is not None:
            final_time_margin = kwargs.get("final_time_margin", 0.10 * final_time)
            g.append(X[self.nodes][4])  # elapsed time at final node
            lbg.append([final_time - final_time_margin])
            ubg.append([final_time + final_time_margin])
            
        # Added / Tunned Optimization Constraints
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##



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
        w = ca.vertcat(*w) # w is the vector of all decision variables (states and controls at all nodes, plus final time)
        g = ca.vertcat(*g) # g is the vector of all constraints (dynamics, performance, waypoint, etc.)
        X = ca.horzcat(*X) # X is the matrix of state variables at all nodes (each column corresponds to a node)
        U = ca.horzcat(*U) # U is the matrix of control variables at all nodes (each column corresponds to a node)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        # Create an NLP solver
        nlp = {"f": J, "x": w, "g": g}

        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.solver_options)
        self.solution = self.solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        status = self.solver.stats()["return_status"]
        print(f"  IPOPT status: {status}")
        print(f"  range = {self.range:.1f}  |  ts_final guess = {self.range / 200:.1f} s")
        # if status not in ("Solve_Succeeded", "Solved_To_Acceptable_Level"):
        #     warnings.warn(f"NLP did not converge for {kwargs.get('flight_id')}: {status}")
        #     return None

        # final timestep
        ts_final = self.solution["x"][-1].full()[0][0]

        # Function to get x and u from w
        output = ca.Function("output", [w], [X, U], ["w"], ["x", "u"])
        x_opt, u_opt = output(self.solution["x"])

        # --- Save NLP decision variables and per-interval cost to CSV ---
        flight_id  = kwargs.get("flight_id", "unknown")
        output_dir = kwargs.get("output_dir", Path(__file__).parent.parent / "Output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        x_np = np.array(x_opt.full())   # shape (5, nodes+1): x, y, h, mass, ts
        u_np = np.array(u_opt.full())   # shape (3, nodes):   mach, vs, psi
        W_MACH_val, W_VS_val, W_PSI_val = W_MACH, W_VS, W_PSI

        nlp_rows = []
        n_nodes = u_np.shape[1]  # number of control nodes
        for k in range(x_np.shape[1]):  # iterate over all state nodes (nodes+1)
            u_k     = u_np[:, k]     if k < n_nodes else np.full(3, np.nan)
            u_next  = u_np[:, k+1]  if k + 1 < n_nodes else np.full(3, np.nan)
            d_mach  = float(u_next[0] - u_k[0]) if (k < n_nodes and k+1 < n_nodes) else np.nan
            d_vs    = float(u_next[1] - u_k[1]) if (k < n_nodes and k+1 < n_nodes) else np.nan
            d_psi   = float(u_next[2] - u_k[2]) if (k < n_nodes and k+1 < n_nodes) else np.nan
            cost_mach = W_MACH_val * d_mach**2 if not np.isnan(d_mach) else np.nan
            cost_vs   = W_VS_val   * d_vs**2   if not np.isnan(d_vs)   else np.nan
            cost_psi  = W_PSI_val  * d_psi**2  if not np.isnan(d_psi)  else np.nan
            cost_total = (cost_mach + cost_vs + cost_psi) if not np.isnan(d_mach) else np.nan
            fuel_kg    = float(x_np[3, k] - x_np[3, k+1]) if k + 1 < x_np.shape[1] else np.nan  # mass drop over interval k → k+1
            nlp_rows.append({
                "flight":       flight_id,
                "node":         k,
                # State variables
                "x_cart_m":     float(x_np[0, k]),
                "y_cart_m":     float(x_np[1, k]),
                "h_m":          float(x_np[2, k]),
                "h_ft":         float(x_np[2, k]) / ft,
                "mass_kg":      float(x_np[3, k]),
                "ts_s":         float(x_np[4, k]),
                # Control variables (NaN at last state-only node)
                "mach":         float(u_k[0])  if k < n_nodes else np.nan,
                "vs_mps":       float(u_k[1])  if k < n_nodes else np.nan,
                "psi_rad":      float(u_k[2])  if k < n_nodes else np.nan,
                "vs_fpm":       float(u_k[1]) / fpm if k < n_nodes else np.nan,
                "psi_deg":      float(u_k[2]) * 180 / pi if k < n_nodes else np.nan,
                # Inter-interval cost terms (NaN at last interval)
                "d_mach":       d_mach,
                "d_vs_mps":     d_vs,
                "d_psi_rad":    d_psi,
                "cost_mach":    cost_mach,
                "cost_vs":      cost_vs,
                "cost_psi":     cost_psi,
                "fuel_kg":      fuel_kg,         # kg burned in this interval (= mass drop)
                "cost_total":   cost_total,      # smoothness cost only; fuel cost is in fuel_kg
            })
        df_nlp = pd.DataFrame(nlp_rows)
        nlp_path = Path(output_dir) / f"nlp_variables_{flight_id}.csv"
        df_nlp.to_csv(nlp_path, index=False)
        print(f"  NLP variables saved to {nlp_path}  (obj={float(self.solution['f']):.4f})")
        # --- End NLP CSV save ---

        df = self.to_trajectory(ts_final, x_opt, u_opt)

        # # Print all states and controls at each node
        # state_cols   = ["x", "y", "h", "mass", "ts"]
        # control_cols = ["mach", "vertical_rate", "heading"]
        # print("=== States (x) at each node ===")
        # print(df[state_cols].to_string())
        # print("=== Controls (u) at each node ===")
        # print(df[control_cols].to_string())

        # # Plot states and controls separately
        # import matplotlib.pyplot as plt
        # nodes = range(len(df))
        # fig, axes = plt.subplots(len(state_cols), 1, figsize=(10, 10), sharex=True)
        # for ax, col in zip(axes, state_cols):
        #     ax.plot(nodes, df[col], label=col)
        #     ax.set_ylabel(col)
        #     ax.legend(loc="upper right")
        # axes[-1].set_xlabel("Node")
        # fig.suptitle("States (x) per Node")
        # plt.tight_layout()

        # fig, axes = plt.subplots(len(control_cols), 1, figsize=(10, 6), sharex=True)
        # for ax, col in zip(axes, control_cols):
        #     ax.plot(nodes, df[col], label=col)
        #     ax.set_ylabel(col)
        #     ax.legend(loc="upper right")
        # axes[-1].set_xlabel("Node")
        # fig.suptitle("Controls (u) per Node")
        # plt.tight_layout()

        # # Third plot: derived quantities — using exact same formulas as drag.clean() and dynamics
        # from openap.extra.aero import g0
        # mass_arr = df["mass"].values
        # tas_arr  = df["tas"].values          # kts  (same units as drag.clean / _cl)
        # alt_arr  = df["altitude"].values     # ft   (same units as drag.clean / _cl)
        # vs_arr   = df["vertical_rate"].values  # fpm (same units as drag.clean / _cl)

        # # gamma — matches _cl() and dynamics: arctan2(vs*fpm, tas*kts)
        # tas_ms    = tas_arr * kts            # m/s
        # vs_ms     = vs_arr  * fpm            # m/s
        # gamma_rad = np.arctan2(vs_ms, tas_ms)
        # gamma_deg = np.rad2deg(gamma_rad)

        # # Drag — reuse self.drag (already configured with wave_drag flag)
        # drag_vals = self.drag.clean(mass=mass_arr, tas=tas_arr, alt=alt_arr, vs=vs_arr)

        # # Thrust (max cruise) — reuse self.thrust
        # thrust_vals = self.thrust.cruise(tas=tas_arr, alt=alt_arr)

        # # Aerodynamic quantities — matching _cl() internals exactly
        # alt_m    = alt_arr * ft
        # S        = self.aircraft["wing"]["area"]
        # rho_vals = np.array([oc.aero.density(h) for h in alt_m])
        # qS_vals  = 0.5 * rho_vals * tas_ms**2 * S
        # qS_safe  = np.maximum(qS_vals, 1e-3)
        # L_vals   = mass_arr * g0 * np.cos(gamma_rad)
        # cl_vals  = L_vals / qS_safe
        # cd_vals  = drag_vals / qS_safe

        # # Gravity component along flight path (acc=0 assumed)
        # grav_path_vals = mass_arr * g0 * np.sin(gamma_rad)   # m·g·sin(γ), N

        # # Required thrust from dynamics: T = D + m·g·sin(γ)  (acc=0)
        # T_required_vals = drag_vals + grav_path_vals

        # derived = {
        #     "TAS (kts)":              df["tas"],
        #     "gamma (deg)":            gamma_deg,
        #     "drag (N)":               drag_vals,
        #     "T_max cruise (N)":       thrust_vals,
        #     "T_required (N)":         T_required_vals,
        #     "rho (kg/m³)":            rho_vals,
        #     "qS (N)":                 qS_vals,
        #     "L (N)":                  L_vals,
        #     "CL (-)":                 cl_vals,
        #     "CD (-)":                 cd_vals,
        #     "m·g·sin(γ) (N)":         grav_path_vals,
        # }
        # fig, axes = plt.subplots(len(derived), 1, figsize=(10, 22), sharex=True)
        # for ax, (label, data) in zip(axes, derived.items()):
        #     ax.plot(nodes, data, label=label)
        #     ax.set_ylabel(label)
        #     ax.legend(loc="upper right")
        # axes[-1].set_xlabel("Node")
        # fig.suptitle("Derived: TAS, gamma, Drag, Thrust, rho, qS, L, CL, CD, m·g·sin(γ) per Node")
        # plt.tight_layout()
        # plt.show()

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
        
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ## Modified By Kuang ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        df.attrs["wp_node_indices"] = self.wp_node_indices #store critical nodes as global comment
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
        return df
       