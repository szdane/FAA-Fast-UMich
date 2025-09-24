# Fuel & Emission Estimation For Flights
# Kuang Sun, September 2025

# import packages
import numpy as np
import pandas as pd

# Gamma compute function
def compute_gamma(vs, tas, limit=True):
    """
    Compute the flight path angle (gamma) in radians.
    # Coding logic from openap.drag._cl
    
    Args:
        vs (float): Vertical speed in m/s.
        tas (float): True airspeed in m/s.

    Returns:
        gamma (float): Flight path angle in radians.
    """
    
    # 1. calculate gamma
    # coding logic from openap.drag._cl
    gamma = np.arctan2(vs, tas)

    # 2. limit gamma to -20 to 20 degrees (0.175 radians)
    # Coding logic from openap.fuel.enroute
    if limit:
        gamma = np.where(gamma < -0.175, -0.175, gamma)
        gamma = np.where(gamma > 0.175, 0.175, gamma)

    return gamma

# Atmosphere calculation function
def compute_atmosphere(h, dT=0):
    """
    Compute air pressure, density, and temperature at a given altitude.
    # Coding logic from openap.extra.aero.atoms

    Args:
        h (float): Altitude in meters.
        dT (float): Temperature shift from ISA in K. Defaults to 0.

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
    dT = np.maximum(-15, np.minimum(dT, 15)) # limit dT to -15 to 15 K
    T0_shift = T0 + dT # shifted sea-level temperature
    T = np.maximum(T0_shift + beta * h, 216.65 + dT) # limit T to tropopause

    # 3. Compute density
    rhotrop = rho0 * (T / T0_shift) ** 4. # density at tropopause
    dhstrat = np.maximum(0.0, h - 11000.0) # height above tropopause
    rho = rhotrop * np.exp(-dhstrat / 6341.552161) # density at altitude

    # 4. Compute pressure
    p = rho * R * T
    
    return p, rho, T

# Drag calculation function
def compute_drag(mass, tas, alt, cd0, k, vs, S):
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
    gamma = compute_gamma(vs, tas, limit=False)

    # 4. calculate rho (air density)
    # coding logic from openap.extra.aero.density
    _, rho, _ = compute_atmosphere(h)

    # 5. calculate qS (dynamic pressure times wing area)
    # Coding logic from openap.drag._cl
    qS = 0.5 * rho * v**2 * S
    qS = np.maximum(qS, 1e-3)  # avoid zero division

    # 6. calculate L (lift)
    # Coding logic from openap.drag._cl
    g0 = 9.80665  # m/s2, Sea level gravity constant
    L = mass * g0 * np.cos(gamma)

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
def compute_thrust(D, mass, gamma, acc):
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
    T = D + mass * 9.81 * np.sin(gamma) + mass * acc

    return T

# Fuel flow calculation function
def compute_fuel_flow(T, tsfc):
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
def compute_emission(fuel_flow):
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
    
    ffac = fuel_flow # Fuel flow for all engines in kg/s
    # 1. compute CO2 emission
    CO2_flow = ffac * 3160 # CO2 emission from all engines in g/s

    # 2. compute H2O emission
    H2O_flow = ffac * 1230 # H2O emission from all engines in g/s

    # 3. compute soot emission
    Soot_flow = ffac * 0.03 # Soot emission from all engines in g/s

    # 4. compute SOx emission
    SOx_flow = ffac * 1.2 # SOx emission from all engines in g/s

    # 5. compute NOx emission
    NOx_flow = ffac * 0.0148 * 1000 # NOx emission from all engines in g/s

    # 6. compute CO emission
    CO_flow = ffac * 0.00325 * 1000 # CO emission from all engines in g/s

    # 7. compute HC emission
    HC_flow = ffac * 0.00032 * 1000 # HC emission from all engines in g/s

    return CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow

# Compute fuel and emission for a flight
def compute_fuel_emission_for_flight(df, S, mtow, tsfc, cd0, k, limit=True, cal_emission=True):
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
        D = compute_drag(mass, tas, alt, cd0, k, vs, S)

        # 3. calculate gamma
        gamma = compute_gamma(vs, tas, limit=False)

        # 4. limit acceleration
        if limit:
            acc = limit_acceleration(acc)

        # 5. calculate thrust
        T = compute_thrust(D, mass, gamma, acc)

        # 6. calculate fuel flow
        fuel_flow = compute_fuel_flow(T, tsfc)

        # 7. calculate emissions
        if cal_emission:
            CO2_flow, H2O_flow, Soot_flow, SOx_flow, NOx_flow, CO_flow, HC_flow = compute_emission(fuel_flow)

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


# Main function to test the above functions
if __name__ == "__main__":

    def main():
        # Example flight data (5 time steps)
        df = pd.DataFrame({
            'd_ts': [60, 60, 60, 60, 60],              # seconds
            'groundSpeed': [250, 255, 260, 265, 270],   # knots
            'alt': [10000, 12000, 14000, 16000, 18000], # feet
            'rateOfClimb': [500, 600, 700, 800, 900]    # ft/min
        })

        # print(df)
        print(df)

        # Example aircraft parameters
        S = 122.6      # m^2 (wing area)
        mtow = 70000   # kg (max takeoff weight)
        tsfc = 0.00003 # kg/Ns (thrust specific fuel consumption)
        cd0 = 0.02     # zero-lift drag coefficient
        k = 0.045      # induced drag factor

        # Calculate total fuel usage only
        total_fuel = compute_fuel_emission_for_flight(df, S, mtow, tsfc, cd0, k, limit=True, cal_emission=False)
        print(f"Total fuel used: {total_fuel:.2f} kg")

        return 0

main()

