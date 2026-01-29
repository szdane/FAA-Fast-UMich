# Fuel & Emission Estimation For Flights _ Gurobi Version
# Kuang Sun, September 2025

# import packages
import numpy as np
import pandas as pd
import math

# Gamma compute function
def compute_gamma(vs, tas, m, limit=True):
    """
    Compute the flight path angle (gamma) in radians.
    # Coding logic from openap.drag._cl
    
    Args:
        vs (Gurobi Var): Vertical speed variable
        tas (Gurobi Var): True airspeed variable  
        model (Gurobi Model): The optimization model
        limit (bool): Whether to limit gamma to ±20 degrees
        
    Returns:
        gamma (Gurobi Var): Flight path angle variable
    """
    
    # 1. calculate gamma with GUROBI version code
        # Create variables
    gamma = m.addVar(lb=-math.pi/2, ub=math.pi/2, name="gamma")
    ratio = m.addVar(lb=-3.0, ub=3.0, name="vs_tas_ratio")
    
        # Piecewise-linear approximation of arctan
    ratio_min = -2.0
    ratio_max = 2.0
    num_points = 100
    ratio_points = np.linspace(ratio_min, ratio_max, num_points)
    gamma_values = np.arctan(ratio_points)
    
        # Define ratio
    m.addConstr(vs == ratio * tas, name="ratio_definition") # This is equivalent to: vs = ratio * tas
    
        # Add equality constraint on gamma
    m.addGenConstrPWL(ratio, gamma, ratio_points.tolist(), gamma_values.tolist(), name="arctan_approximation")

    # 2. limit gamma to -20 to 20 degrees (0.175 radians)
    if limit:
        # Update the variable bounds to limit gamma
        gamma.lb = max(gamma.lb, -0.175)  # -20 degrees in radians
        gamma.ub = min(gamma.ub, 0.175)   # +20 degrees in radians
        # For Gurobi, we handle limits through variable bounds, not np.where

    return gamma  # Return the Gurobi variable, not a constant!

# Atmosphere calculation function
def compute_atmosphere(h, m, dT=0):
    """
    Compute air pressure, density, and temperature at a given altitude.
    # Coding logic from openap.extra.aero.atoms

    Args:
        h (float): Altitude in meters.
        dT (float): Temperature shift from ISA in K. Defaults to 0.
        m (Gurobi Model): The optimization model.

    Returns:
        p (float): Air pressure in Pa.
        rho (float): Air density in kg/m^3.
        T (float): Air temperature in K.
    """
    # 1. Input constants
    beta = -0.0065  # [K/m] ISA temp gradient below tropopause
    T0 = 288.15  # K, temperature, sea level ISA
    rho0 = 1.225  # kg/m3, air density, sea level ISA
    R = 287.05287  # m2/(s2 x K), gas constant, sea level ISA
    
    # 2. Compute temperature
        # calculate sea-level temperatur
    dT = np.maximum(-15, np.minimum(dT, 15)) # sea level temperature shift
    T0_shift = T0 + dT # default dT = 0

        # create variables
    T = m.addVar(name="T")  # Auxiliary variable for the maximum
    T_tropopause = m.addVar(name="Temp_tropopause")  # temperature at tropopause (11km)
    T_h = m.addVar(name="T_h")  # temperature at altitude h above sea level

        # Define T_tropopause and T_h
    m.addConstr(T_tropopause == 216.65 + dT, name="T_tropopause")
    m.addConstr(T_h == T0_shift + beta * h, name="T_h")

        # Add lower constraint on T
    m.addGenConstrMax(T, [T_tropopause, T_h], name="define_T") #ensure T does not go below T_tropopause, which means h shouldn't be above tropopause (11km) 
   
    # 3. Compute density
        # define density at tropopause
    rhotrop = rho0 * (T / T0_shift) ** 4.

        # create variables
    alt_diff_tropopause = m.addVar(name="alt_diff_tropopause")  # Altitude difference between set altitude and tropopause, when set altitude is at tropopause
    alt_diff_h = m.addVar(name="alt_diff_h")  # Altitude difference between set altitude and tropopause, when set altitude is at tropopause
    dhstrat = m.addVar(name="dhstrat")  # difference between set altitude and tropopause height (11km)

        # Define altitude differences
    m.addConstr(alt_diff_tropopause == 0.0, name="alt_diff_tropopause")
    m.addConstr(alt_diff_h == h - 11000.0, name="alt_diff_h") # alt_diff_h should be a negative value if h < 11k

        # Add higher constraint (alt_diff = 0) constraint on dhstrat # same as dhstrat = np.maximum(0.0, h - 11000.0)
    m.addGenConstrMax(dhstrat, [alt_diff_tropopause, alt_diff_h], name="define_h") #constraint set altitude to be below tropopause (maximum if 0 at tropopause)

        # create variables
    rho = m.addVar(name="rho")  # Variable for density
    exp_e = m.addVar(name="exp_e")  # exponentiaal term for e
    e = m.addVar(name="e")  # Natural constant

        # define e
    e = math.e

        # calculate e^x and rho (by adding equality constraints)
    m.addConstr(exp_e == e ** (-dhstrat / 6341.552161), name="calculate_exp_e")
    m.addConstr(rho == rhotrop * exp_e) # density at altitude

    # 4. Compute pressure
    p = rho * R * T
    
    return p, rho, T

# Drag calculation function
def compute_drag(mass, tas, alt, cd0, k, vs, S, m):
    """
    Compute the drag force using a simple parabolic drag polar.
    # Coding logic from openap.drag._calc_drag
    Args:
        mass (float): Aircraft mass in kg. 
        tas (float): True airspeed in m/s.
        alt (float): Altitude in meters.
        cd0 (float): Zero-lift drag coefficient.
        k (float): Induced drag factor.
        vs (float): Vertical speed in m/s.
        S (float): Wing area in m^2.
        m (Gurobi Model): The optimization model.

    Returns:
        D (float): Drag force in N.
    """
    # 1. calculate cd0
    # Coding logic from openap.drag.clean
    dCdw = 0 # assuming no wave drag
    cd0 = cd0 + dCdw

    # 2. input constants
    # Coding logic from openap.drag._cl & openap.extra.aero
    v = tas # in m/s
    h = alt # in m
    vs = vs # in m/s

    # 3. calculate gamma
    gamma = compute_gamma(vs, tas, m, limit=False)

    # 4. calculate rho (air density)
    # coding logic from openap.extra.aero.density
    _, rho, _ = compute_atmosphere(h, m)

    # 5. calculate qS (dynamic pressure times wing area)
    # Coding logic from openap.drag._cl
        # Create variables
    qS = m.addVar(name="qS")  # Variable for qS
    const_qS = m.addVar(name="const_qS")  # Variable for the constant term

        # Define variables
    m.addConstr(const_qS == 1e-3, name="const_constraint")

        # calculate qS = 0.5 * rho * v^2 * S
    m.addConstr(qS == 0.5 * rho * v**2 * S, name="qS_constraint")
    m.addGenConstrMax(qS, [qS, const_qS], name="qS_max_constraint")  # qS = np.maximum(qS, 1e-3)  # avoid zero division
   
    # 6. calculate L (lift)
    # Coding logic from openap.drag._cl
    g0 = 9.80665  # m/s2, Sea level gravity constant

    # calculate cos(gamma) using piecewise-linear approximation
    gamma_min = -np.pi / 2
    gamma_max = np.pi / 2
    num_points = 100  # Number of points for the piecewise-linear approximation

    gamma_points = np.linspace(gamma_min, gamma_max, num_points)
    cos_values = np.cos(gamma_points)

    cos_gamma = m.addVar(name="cos_gamma")  # create variable for cosine(gamma)
    L = m.addVar(name="L")  # Create variable for lift

    m.addGenConstrPWL(gamma, cos_gamma, gamma_points.tolist(), cos_values.tolist(), name="cos_pwl_constraint") # calculate cos(gamma)
    m.addConstr(L == mass * g0 * cos_gamma, name="L_constraint") # calculate lift L = mass * g0 * np.cos(gamma)

    # 7. calculate cl (lift coefficient)
    # Coding logic from openap.drag._calc_drag
    cl = L / qS

    # 8. calculate cd (drag coefficient)
    # Coding logic from openap.drag._calc_drag
    cd = cd0 + k * cl**2

    # 9. calculate D (drag)
    # Coding logic from openap.drag._calc_drag
    D = cd * qS
    
    return D

# Acceleration limit check function
def limit_acceleration(acc):
    """
    Limit the acceleration to under 5 m/s^2.
    # Coding logic from openap.fuel.enroute
    
    Args:
        acc (float): Acceleration in m/s^2.

    Returns:
        acc (float): Limited acceleration in m/s^2.
    """
    
    # 1. limit acc to under 5 m/s^2
    acc = np.where(acc < -5, -5, acc)
    acc = np.where(acc > 5, 5, acc)

    return acc

# Thrust calculation function
def compute_thrust(D, mass, gamma, acc, model):
    """
    Compute the required thrust.
    # Coding logic from openap.fuel.enroute

    Args:
        D (float): Drag force in N.
        mass (float): Aircraft mass in kg.
        gamma (float): Flight path angle in radians.
        acc (float): Acceleration in m/s^2.

    Returns:
        T (float): Required thrust in N.
    """
    
    # 1. calculate thrust
    # T = D + mass * 9.81 * np.sin(gamma) + mass * acc

    gamma_min = -np.pi / 2
    gamma_max = np.pi / 2
    num_points = 100  # Number of points for the piecewise-linear approximation

    gamma_points = np.linspace(gamma_min, gamma_max, num_points)
    sin_values = np.sin(gamma_points)

    
    sin_gamma = model.addVar(name="sin_gamma")  # Create variable for sine
    T = model.addVar(name="T") # Create variable for thrust

    model.addGenConstrPWL(gamma, sin_gamma, gamma_points.tolist(), sin_values.tolist(), name="sin_pwl_constraint") # calculate sin(gamma) using piecewise-linear approximation
    model.addConstr(T == D + mass * 9.81 * sin_gamma + mass * acc, name="T_constraint") # calculate Thrust

    return T

# Fuel flow calculation function
def compute_fuel_flow_from_thrust(T, tsfc):
    """
    Compute the fuel flow based on the required thrust and altitude.
    # Assume a constant at different altitude tsfc for simplicity. 
    # (This is a very bold assumption indeed, might need to be improved in the future)
    
    Args:
        T (float): Required thrust in N.
        tsfc (float): Thrust specific fuel consumption in kg/Ns.

    Returns:
        fuel_flow (float): Fuel flow in kg/s.
    """

    # 1. calculate the fuel flow
    fuel_flow = T * tsfc

    return fuel_flow

# compute emission function
def compute_emission_from_fuel_flow(fuel_flow):
    """
    Compute the emissions based on the fuel flow.
    # Coding logic from openap.emission
    # reference: openap original linear relationship 
    # and https://ansperformance.eu/economics/cba/standard-inputs/latest/chapters/amount_of_emissions_released_by_fuel_burn.html
    
    Args:
        fuel_flow (float): Fuel flow in kg/s.

    Returns:
        CO2_flow (float): CO2 emission in g/s.
        H2O_flow (float): H2O emission in g/s.
        Soot_flow (float): Soot emission in g/s.
        SOx_flow (float): SOx emission in g/s.
        NOx_flow (float): NOx emission in g/s.
        CO_flow (float): CO emission in g/s.
        HC_flow (float): HC emission in g/s.
    """
    
    # 1. input constants
    ffac = fuel_flow # Fuel flow for all engines in kg/s

    # 2. compute emissions
    CO2_flow = ffac * 3160 # CO2 emission from all engines in g/s
    H2O_flow = ffac * 1230 # H2O emission from all engines in g/s
    Soot_flow = ffac * 0.03 # Soot emission from all engines in g/s
    SOx_flow = ffac * 1.2 # SOx emission from all engines in g/s
    NOx_flow = ffac * 0.0148 * 1000 # NOx emission from all engines in g/s
    CO_flow = ffac * 0.00325 * 1000 # CO emission from all engines in g/s
    HC_flow = ffac * 0.00032 * 1000 # HC emission from all engines in g/s

    return CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow

# Compute fuel and emission flow at a timestep
def compute_fuel_emission_flow(tas, alt, vs, mass, S, cd0, k, tsfc, m, limit=True, cal_emission=False):
    """
    Compute the fuel and emission flow at a timestep.

    Args:
        tas (float): True airspeed in m/s.
        alt (float): Altitude in meters.
        vs (float): Vertical speed in m/s.
        mass (float): Aircraft mass in kg.
        S (float): Wing area in m^2.
        cd0 (float): Zero-lift drag coefficient.
        k (float): Induced drag factor.
        tsfc (float): Thrust specific fuel consumption in kg/Ns.
        m (Gurobi Model): The Gurobi optimization model.
        limit (bool): Whether to limit acceleration. Defaults to True.
        cal_emission (bool): Whether to calculate emissions. Defaults to True.

    Returns:
        fuel_flow (float): Fuel flow in kg/s.
        CO2_flow (float): CO2 emission in g/s.
        H2O_flow (float): H2O emission in g/s.
        Soot_flow (float): Soot emission in g/s.
        SOx_flow (float): SOx emission in g/s.
        NOx_flow (float): NOx emission in g/s.
        CO_flow (float): CO emission in g/s.
        HC_flow (float): HC emission in g/s.
    """

    # 1. calculate drag
    D = compute_drag(mass, tas, alt, cd0, k, vs, S, m)

    # 2. calculate gamma
    gamma = compute_gamma(vs, tas, m, limit=False)

    # 3. limit acceleration
    acc = 0 # assume zero acceleration at a timestep
    if limit:
        acc = limit_acceleration(acc)

    # 4. calculate thrust
    T = compute_thrust(D, mass, gamma, acc, m)

    # 5. calculate fuel flow
    fuel_flow = compute_fuel_flow_from_thrust(T, tsfc)

    # 6. calculate emission flow
    if cal_emission:
        CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow = compute_emission_from_fuel_flow(fuel_flow)

    # if cal_emission:
    #     return fuel_flow, CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow
    # else:
    return fuel_flow
    
# Compute fuel and emission for a flight
def compute_fuel_emission_for_flight(df, S, mtow, tsfc, cd0, k, m, limit=True, cal_emission=True):
    """
    Compute the fuel and emission for a flight.

    Args:
        df (DataFrame): Flight data, including columns: d_ts (s), groundSpeed (kts), alt (ft), rateOfClimb (ft/min).
        S (float): Wing area in m^2.
        mtow (float): Maximum takeoff weight in kg.
        tsfc (float): Thrust specific fuel consumption in kg/Ns.
        cd0 (float): Zero-lift drag coefficient.
        k (float): Induced drag factor.

    Returns:
        total_fuel (float): Total fuel consumed in kg.
        total_emissions (dict): Dictionary containing total emissions in g.
    """

    # Loop through each time step in the flight data
    total_fuel = 0

    total_CO2 = 0
    total_H2O = 0
    total_Soot = 0
    total_SOx = 0
    total_NOx = 0
    total_CO = 0
    total_HC = 0

    # 1. Initialize variables
    mass = 0.8 * mtow # initial mass is assumed to be MTOW 
    # (This is a bold assumption, might need to be improved in the future)

    for index, row in df.iterrows():
        dt = row.d_ts # time step in seconds
        tas = row.groundSpeed * 0.514444 # convert kts to m/s
        alt = row.alt * 0.3048 # convert ft to m
        vs = row.rateOfClimb * 0.00508 # convert ft/min to m/s
        acc = (row.groundSpeed - df.iloc[index - 1].groundSpeed) * 0.514444 / dt if index > 0 else 0 # convert acceleration from ft/min^2 to m/s^2

        # 2. calculate drag
        D = compute_drag(gamma, mass, tas, alt, cd0, k, vs, S, m)

        # 3. calculate gamma
        gamma = compute_gamma(vs, tas, m, limit=False)

        # 4. limit acceleration
        if limit:
            acc = limit_acceleration(acc)

        # 5. calculate thrust
        T = compute_thrust(D, mass, gamma, acc, m)

        # 6. calculate fuel flow
        fuel_flow = compute_fuel_flow_from_thrust(T, tsfc)

        # 7. calculate emissions
        if cal_emission:
            CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow = compute_emission_from_fuel_flow(fuel_flow)

        # 8. update total fuel and emissions
        total_fuel += fuel_flow * dt # in kg

        if cal_emission:
            total_CO2 += CO2_flow * dt # in g
            total_H2O += H2O_flow * dt # in g
            total_Soot += Soot_flow * dt # in g
            total_SOx += SOx_flow * dt # in g
            total_NOx += NOx_flow * dt # in g
            total_CO += CO_flow * dt # in g
            total_HC += HC_flow * dt # in g

        # 9. update mass
        mass -= fuel_flow * dt # see how it's done on website!!!

    if cal_emission:
        return total_fuel, total_CO2, total_H2O, total_Soot, total_SOx, total_NOx, total_CO, total_HC
    else:
        return total_fuel