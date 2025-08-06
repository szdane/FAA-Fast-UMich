# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 12:44:42 2025

@author: Eric

Multi-Aircraft pre-TRACON Re-routing Model
Simplified Test Version
"""

#%% IMPORT PACKAGES
import pandas as pd
import ast
from shapely.geometry import Polygon
import gurobipy as gp
from gurobipy import GRB

#%% DATA INPUTS (TEST CASE)
"""
The test case involves some number of duplicates of the same flight arriving at the same time, to ensure that the model works properly

List of necessary inputs for all flights:
	1. arrival time of flight to pre-TRACON, converted into a timestep
	2. 3D entry coordinate of flight to pre-TRACON
	3. maximum velocity of flight in planar directions (x,y,z)
	4. scheduled/historical entry time to TRACON
	5. metering fixes that can be utilized by the flight
"""

# Items 2, 1, and 4
def read_in_flight_info(df_path, acID):
	df = pd.read_csv(df_path) #'Yuwei 4-18\\flight_entry_times_with_coords.csv'
	df.set_index('acId', inplace = True)
	
	flight_info = df.loc[acID]
	entry_coord = ast.literal_eval(flight_info['enter_pre_tracon_coord']) # String to tuple
	entry_time = pd.to_datetime(flight_info['enter_pre_tracon'])
	exit_time = pd.to_datetime(flight_info['enter_tracon'])
	
	return entry_coord, entry_time, exit_time

# INPUT
flight_df_path = 'Yuwei 5-12\\flight_entry_times_with_coords.csv'
acID = 'SKW4010_KIADtoKDTW' # This will be our example flight to duplicate for the test case
preTRA_entry_coord, preTRA_entry_datetime, hist_TRA_entry_datetime = read_in_flight_info(flight_df_path, acID)
# Add altitude to the preTRA_entry_coord, for this example
preTRA_entry_coord = (preTRA_entry_coord[0], preTRA_entry_coord[1], 30000) # Using 30000 as a placeholder

#%% DATA INPUTS: VELOCITY ESTIMATES
#(from Serra)
max_velocityEst_lon = .5 # in degrees per minute
max_velocityEst_lat = .144
max_velocityEst_alt = 2000 # in feet per minute

# SAFE SEPARATION REQUIREMENTS
min_horiz_dist = 5 / 60 # Aircraft must be seperated by 5 nautical miles, converted into degrees
min_vert_dist = 1000 # OR, Aircraft must be seperated by 1000ft vertically

#%% DATA INPUTS: DETROIT METERING FIXES (from Yuwei)
star_fixes = {
    "BONZZ": (-82.7972, 41.7483), "CRAKN": (-82.9405, 41.6730), "CUUGR": (-83.0975, 42.3643),
    "FERRL": (-82.6093, 42.4165), "GRAYT": (-83.6020, 42.9150), "HANBL": (-84.1773, 41.7375),
    "HAYLL": (-84.2975, 41.9662), "HTROD": (-83.3442, 42.0278), "KKISS": (-83.7620, 42.5443),
    "KLYNK": (-82.9888, 41.8793), "LAYKS": (-83.5498, 42.8532), "LECTR": (-84.0217, 41.9183),
    "RKCTY": (-83.9603, 42.6869), "VCTRZ": (-84.0670, 41.9878)
}

tracon_polygon = Polygon(star_fixes.values()).convex_hull
metering_fix_coords = tracon_polygon.exterior.coords[:-1] # Don't include the duplicate "closure point" at the end

dtw_lat, dtw_lon, dtw_alt = 42.2125, -83.3534, 650
preTRACON_radius = 3 # degrees surrounding the airport, datasets were pre-calculated using this preTRA radius

# Add altitude estimate to the metering fixes
altitude_fix_estimation = 10000 #ft, estimate from Serra
for index, fix_coord in enumerate(metering_fix_coords):
	metering_fix_coords[index] = (fix_coord[0], fix_coord[1], altitude_fix_estimation)

#%% DATA PROCESSING: GENERATE FLIGHTS, FEASIBLE TIMES, ENTRY COORDINATES, MAX VELOCITY MEASUREMENTS, FEASIBLE METERING FIXES

# For example purposes, create a fake set of flights. In this case they're actually all duplicates of the same flight.
def generate_exmpl_flights(acID, num_flights, preTRA_entry_coord, preTRA_entry_time, hist_TRA_entry_time):
	flights = []
	for flight_ind in range(num_flights):
		flights.append(acID + "_" + str(flight_ind))
	
	# Create dictionaries for flight times and coordinates
	preTRA_entry_coords = {}
	preTRA_entry_datetimes = {}
	TRA_entry_datetimes = {}
	for flight in flights:
		preTRA_entry_coords[flight] = preTRA_entry_coord
		preTRA_entry_datetimes[flight] = preTRA_entry_time
		TRA_entry_datetimes[flight] = hist_TRA_entry_time
		
	return flights, preTRA_entry_coords, preTRA_entry_datetimes, TRA_entry_datetimes

# Create timesteps which represent a particular interval of time (code taken from my old FAA ZOB work)
def create_timesteps(secs_per_interval, slack_time, flights, preTRA_entry_datetimes, TRA_entry_datetimes):
	# set "starting_time" to the earliest departure time - mins_per_interval (avoid creating flights with a departure time of 0)
	earliest_entry_time = pd.Timestamp.max # Arbitrarily far in the future
	for flight in flights:
		if preTRA_entry_datetimes[flight] < earliest_entry_time:
			earliest_entry_time = preTRA_entry_datetimes[flight]
	
	sim_starting_time = earliest_entry_time - pd.Timedelta(seconds=2*secs_per_interval) # Ensuring no flights arrive exactly at sim_starting_time
	
	# set "ending_time" to the last arrival time + slack_time
	latest_arrival_time = pd.Timestamp.min
	for flight in flights:
		if TRA_entry_datetimes[flight] > latest_arrival_time:
			latest_arrival_time = TRA_entry_datetimes[flight]
	
	sim_ending_time = latest_arrival_time + slack_time + pd.Timedelta(seconds=2*secs_per_interval) # The last term is extra padding just to be safe
	
	total_time = int((sim_ending_time - sim_starting_time) / pd.Timedelta(seconds=secs_per_interval))
	times = range(total_time) # zero-indexed
	
	return times, sim_starting_time, sim_ending_time

# Convert entry and exit times to timesteps (rename variables)
def datetime2int(timestamp, starting_time, secs_per_interval):
	rounded_timestamp = timestamp.round(freq = f'{secs_per_interval}s') # Round this time to the nearest 5 minute interval
	
	timedelta_since_start = rounded_timestamp - starting_time
	timesteps_since_start = timedelta_since_start / pd.Timedelta(seconds=secs_per_interval) # Divide this timedelta by our interval size to get the number of intervals since midnight

	return(int(timesteps_since_start))

def timedelta2int(timedelta, secs_per_interval):
    rounded_timedelta = timedelta.round(freq = f'{secs_per_interval}s') # round to nearest interval size
    timesteps = None
    # Make sure we didn't round down to 0 (we can't have a duration of 0)
    if rounded_timedelta > pd.Timedelta(0):
        timesteps = rounded_timedelta / pd.Timedelta(seconds=secs_per_interval) # Dividing a timedelta by another timedelta returns a number
    else:
        timesteps = 1

    return int(timesteps)

def generate_ent_ex_timesteps(secs_per_interval, slack_time, sim_starting_time, flights, preTRA_entry_datetimes, TRA_entry_datetimes):
	entry_times = {}
	hist_exit_times = {}
	latest_feasible_exit = {}
	slack_timesteps = timedelta2int(slack_time, secs_per_interval) # For now, we'll set this as two hours late (time intervals are 5 minutes)
	for flight in flights:
		# Set the Entry Times
		# Note that flights CANNOT depart at time 0, as this will break the indexing of the model
		entry_times[flight] = datetime2int(preTRA_entry_datetimes[flight], sim_starting_time, secs_per_interval)
	
		# Set the Exit Times
		hist_exit_times[flight] = datetime2int(TRA_entry_datetimes[flight], sim_starting_time, secs_per_interval)
	
		# Set a "Latest Feasible Exit Time" for each flight
		latest_feasible_exit[flight] = hist_exit_times[flight] + slack_timesteps
		
	return entry_times, hist_exit_times, latest_feasible_exit

def generate_feas_times(flights, entry_times, latest_feasible_exit):
	feas_times = {}
	for flight in flights:
		feas_times[flight] = range(entry_times[flight], latest_feasible_exit[flight] + 1) # +1 because we want the last feasible time to be included in the range
		
	return feas_times

def assign_entry_coords(flights):
	entry_lon = {}
	entry_lat = {}
	entry_alt = {}
	for flight in flights:
		entry_lon[flight] = preTRA_entry_coord[0] # Every flight has the same entry coordinate for now
		entry_lat[flight] = preTRA_entry_coord[1]
		entry_alt[flight] = preTRA_entry_coord[2]
		
	return entry_lon, entry_lat, entry_alt

def assign_max_veloc(flights):
	max_velocity_lon = {}
	max_velocity_lat = {}
	max_velocity_alt = {}
	for flight in flights:
		max_velocity_lon[flight] = max_velocityEst_lon # Using estimates for now, will use data eventually based on aircraft type
		max_velocity_lat[flight] = max_velocityEst_lat
		max_velocity_alt[flight] = max_velocityEst_alt
		
	return max_velocity_lon, max_velocity_lat, max_velocity_alt
		
def assign_metering_fixes(flights):
	feas_metering_fixes = {}
	for flight in flights:
		feas_metering_fixes[flight] = metering_fix_coords # Flights can choose any metering fix for now
	
	return feas_metering_fixes

# 3 INPUT
num_flights = 2
secs_per_interval = 30
slack_time = pd.Timedelta(minutes=5) # Amount of time that a flight can feasibly be "late," or later than historical exit from preTRA

# FLIGHTS
# Generate example flights
flights, preTRA_entry_coords, preTRA_entry_datetimes, TRA_entry_datetimes = generate_exmpl_flights(acID, num_flights, preTRA_entry_coord, preTRA_entry_datetime, hist_TRA_entry_datetime)

# TIMES
# Create timesteps
times, sim_starting_time, sim_ending_time = create_timesteps(secs_per_interval, slack_time, flights, preTRA_entry_datetimes, TRA_entry_datetimes)
# Convert all datetime inputs to timesteps
entry_times, hist_exit_times, latest_feasible_exit = generate_ent_ex_timesteps(secs_per_interval, slack_time, sim_starting_time, flights, preTRA_entry_datetimes, TRA_entry_datetimes)
# Create the set of feasible pre-TRACON times for each flight
feas_times = generate_feas_times(flights, entry_times, latest_feasible_exit)

# ENTRY COORDINATES
entry_lon, entry_lat, entry_alt = assign_entry_coords(flights)

# VELOCITY
# Assign maximum velocity for each flight
max_velocity_lon, max_velocity_lat, max_velocity_alt = assign_max_veloc(flights)

# METERING FIXES
feas_metering_fixes = assign_metering_fixes(flights)

#%% RE-ROUTING OPTIMIZATION

	# GENERATE DECISION VARIABLES
def generate_decision_vars(preTRA_model, flights, times, feas_times, feas_metering_fixes, entry_alt):
	# Coordinates at each time interval
	traj_lon = {}
	traj_lat = {}
	traj_alt = {}
	for flight in flights:
		flight_entry_time = feas_times[flight][0]
		traj_lon[flight] = {}
		traj_lat[flight] = {}
		traj_alt[flight] = {}
		# Fixing initial altitude for now
		traj_alt[flight][flight_entry_time] = entry_alt[flight]
		for time in feas_times[flight]:
			traj_lon[flight][time] = preTRA_model.addVar(name = f"traj_lon[{flight}][{time}]", vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY)
			traj_lat[flight][time] = preTRA_model.addVar(name = f"traj_lat[{flight}][{time}]", vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY)
			# Altitude is fixed (not a decision variable) at the entry time
			if time > flight_entry_time:
				traj_alt[flight][time] = preTRA_model.addVar(name = f"traj_alt[{flight}][{time}]", vtype = GRB.CONTINUOUS, lb = 0)
	# Binary selector between lat-lon seraration and altitude separation
		# 1 if enforcing altitude separation, 0 if enforcing lat-lon separation
	separation_direc = {}
	for flight in flights:
		separation_direc[flight] = {}
		for time in feas_times[flight]:
			separation_direc[flight][time] = {}
			for intruder_flight in flights:
				if flight != intruder_flight:
					separation_direc[flight][time][intruder_flight] = preTRA_model.addVar(name = f"separation_direc[{flight}][{time}][{intruder_flight}]", vtype = GRB.BINARY)
	# Binary selector for choosing a metering fix
		# 1 if metering fix is chosen by flight, 0 if not
	fix_selected = {}
	for flight in flights:
		fix_selected[flight] = {}
		for metering_fix in feas_metering_fixes[flight]:
			fix_selected[flight][metering_fix] = preTRA_model.addVar(name = f"fix_selected[{flight}][{metering_fix}]", vtype = GRB.BINARY)
	# Binary indicator that a flight is assigned to arrived BY a certain time (meaning arrived at or before)
	arrived_by = {}
	for flight in flights:
		arrived_by[flight] = []
		last_feasible_index = len(feas_times[flight]) - 1
		for index, time in enumerate(feas_times[flight]):
			# Can't arrive earlier than feasible
			if index == 0:
				for early_time in range(time):
					arrived_by[flight].append(0)
			if index != last_feasible_index:
				arrived_by[flight].append(preTRA_model.addVar(name = f"arrived_by[{flight}][{time}]", vtype = GRB.BINARY))
			# We must have arrived by the last feasible time
			else:
				for late_time in range(time, times[-1]):
					arrived_by[flight].append(1)
	# Represeting the absolute value of velocity in each direction, AKA the speed
	lon_speed = {}
	lat_speed = {}
	alt_speed = {}
	for flight in flights:
		lon_speed[flight] = {}
		lat_speed[flight] = {}
		alt_speed[flight] = {}
		for time in feas_times[flight]:
			lon_speed[flight][time] = preTRA_model.addVar(name = f"lon_speed[{flight}][{time}]", vtype = GRB.CONTINUOUS)
			lat_speed[flight][time] = preTRA_model.addVar(name = f"lat_speed[{flight}][{time}]", vtype = GRB.CONTINUOUS)
			alt_speed[flight][time] = preTRA_model.addVar(name = f"alt_speed[{flight}][{time}]", vtype = GRB.CONTINUOUS)
	# The (positive) distance from this flight to every other flight at the given time
	lon_separ_sign = {}
	lat_separ_sign = {}
	alt_separ_sign = {}
	# NOTE: this could be more efficient if there's only one decision variable for the separation, regardless of order
	for flight in flights:
		lon_separ_sign[flight] = {}
		lat_separ_sign[flight] = {}
		alt_separ_sign[flight] = {}
		for time in feas_times[flight]:
			lon_separ_sign[flight][time] = {}
			lat_separ_sign[flight][time] = {}
			alt_separ_sign[flight][time] = {}
			for intruder_flight in flights:
				if flight != intruder_flight:
					lon_separ_sign[flight][time][intruder_flight] = preTRA_model.addVar(name = f"lon_separ_sign[{flight}][{time}][{intruder_flight}]", vtype = GRB.BINARY)
					lat_separ_sign[flight][time][intruder_flight] = preTRA_model.addVar(name = f"lat_separ_sign[{flight}][{time}][{intruder_flight}]", vtype = GRB.BINARY)
					alt_separ_sign[flight][time][intruder_flight] = preTRA_model.addVar(name = f"alt_separ_sign[{flight}][{time}][{intruder_flight}]", vtype = GRB.BINARY)
	# 
					
	return traj_lon, traj_lat, traj_alt, separation_direc, fix_selected, arrived_by, lon_speed, lat_speed, alt_speed, lon_separ_sign, lat_separ_sign, alt_separ_sign

	# DEFINED QUANTITIES
def velocity(coord, next_coord, secs_per_interval):
	mins_per_interval = secs_per_interval / 60
	difference = next_coord - coord
	velocity = difference / mins_per_interval
	
	return velocity

def exit_time(arrived_by_flight, feas_times_flight):
	preTRA_exit_time = gp.LinExpr()
	for time in feas_times_flight:
		preTRA_exit_time += (arrived_by_flight[time] - arrived_by_flight[time - 1]) * time
		
	return preTRA_exit_time

	# GENERATE CONSTRAINTS
	# Physics and Collision-Avoidance Constraints
def generate_phys_collision_constrs(preTRA_model, arrived_by, traj_lon, traj_lat, traj_alt, separation_direc, max_velocity_lon, max_velocity_lat, max_velocity_alt, min_horiz_dist, min_vert_dist, entry_lon, entry_lat, entry_margin, lon_speed, lat_speed, alt_speed, lon_separ_sign, lat_separ_sign, alt_separ_sign, secs_per_interval):
	# Define Speed, Enforce Maximum Velocity
	for flight in flights:
		for time in feas_times[flight][:-1]:
			# Create linear expressions for the velocities
			V_lon_expr = velocity(traj_lon[flight][time], traj_lon[flight][time + 1], secs_per_interval)
			V_lat_expr = velocity(traj_lat[flight][time], traj_lat[flight][time + 1], secs_per_interval)
			V_alt_expr = velocity(traj_alt[flight][time], traj_alt[flight][time + 1], secs_per_interval)
			
			# Define Speed (absolute value of velocity)
			preTRA_model.addConstr(lon_speed[flight][time] >= V_lon_expr, name = f"def_lon_speed_pos[{flight}][{time}]")
			preTRA_model.addConstr(-lon_speed[flight][time] <= V_lon_expr, name = f"def_lon_speed_neg[{flight}][{time}]")
			preTRA_model.addConstr(lat_speed[flight][time] >= V_lat_expr, name = f"def_lat_speed_pos[{flight}][{time}]")
			preTRA_model.addConstr(-lat_speed[flight][time] <= V_lat_expr, name = f"def_lat_speed_neg[{flight}][{time}]")
			preTRA_model.addConstr(alt_speed[flight][time] >= V_alt_expr, name = f"def_alt_speed_pos[{flight}][{time}]")
			preTRA_model.addConstr(-alt_speed[flight][time] <= V_alt_expr, name = f"def_alt_speed_neg[{flight}][{time}]")
			
			# Enforce maximum velocity
				# Indicator Constraints (constraints only are valid if the flight hasn't yet arrived)
				# might need to turn this into a standard big-M constraint
			preTRA_model.addConstr((arrived_by[flight][time] == 0) >> (lon_speed[flight][time] <= max_velocity_lon[flight]), name = f"enf_max__Vlon[{flight}][{time}]")
			preTRA_model.addConstr((arrived_by[flight][time] == 0) >> (lat_speed[flight][time] <= max_velocity_lat[flight]), name = f"enf_max__Vlat[{flight}][{time}]")
			preTRA_model.addConstr((arrived_by[flight][time] == 0) >> (alt_speed[flight][time] <= max_velocity_alt[flight]), name = f"enf_max__Valt[{flight}][{time}]")
			
	# Define Distance, Enforce Separation
	bigM_horiz_sep = 2*min_horiz_dist # Setting big M as a safe upper bound for the constraint's right hand side value
	bigM_vert_sep = 2*min_vert_dist
	for flight in flights:
		for time in feas_times[flight]:
			for intruder_flight in flights:
				if flight != intruder_flight:
					lon_dist = traj_lon[flight][time] - traj_lon[intruder_flight][time]
					lat_dist = traj_lat[flight][time] - traj_lat[intruder_flight][time]
					alt_dist = traj_alt[flight][time] - traj_alt[intruder_flight][time]
					
					# Longitudidal and Latitudinal Separation
					preTRA_model.addConstr(bigM_horiz_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + separation_direc[flight][time][intruder_flight] + lon_separ_sign[flight][time][intruder_flight]) + lon_dist >= min_horiz_dist, name = f"enf_lon_sep_pos[{flight}][{time}][{intruder_flight}]")
					preTRA_model.addConstr(-bigM_horiz_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + separation_direc[flight][time][intruder_flight] + 1 - lon_separ_sign[flight][time][intruder_flight]) + lon_dist <= -min_horiz_dist, name = f"enf_lon_sep_neg[{flight}][{time}][{intruder_flight}]")
					preTRA_model.addConstr(bigM_horiz_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + separation_direc[flight][time][intruder_flight] + lat_separ_sign[flight][time][intruder_flight]) + lat_dist >= min_horiz_dist, name = f"enf_lat_sep_pos[{flight}][{time}][{intruder_flight}]")
					preTRA_model.addConstr(-bigM_horiz_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + separation_direc[flight][time][intruder_flight] + 1 - lat_separ_sign[flight][time][intruder_flight]) + lat_dist <= -min_horiz_dist, name = f"enf_lat_sep_neg[{flight}][{time}][{intruder_flight}]")
					
					# Vertical Separation
					preTRA_model.addConstr(bigM_vert_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + 1 - separation_direc[flight][time][intruder_flight] + alt_separ_sign[flight][time][intruder_flight]) + alt_dist >= min_vert_dist, name = f"enf_vert_sep_pos[{flight}][{time}][{intruder_flight}]")
					preTRA_model.addConstr(-bigM_vert_sep*(arrived_by[flight][time] + arrived_by[intruder_flight][time] + 1 - separation_direc[flight][time][intruder_flight] + 1 - alt_separ_sign[flight][time][intruder_flight]) + alt_dist <= -min_vert_dist, name = f"enf_vert_sep_neg[{flight}][{time}][{intruder_flight}]")
					
	# Entry-point Margin
	for flight in flights:
		flight_entry_time = feas_times[flight][0]
		lon_dist = traj_lon[flight][flight_entry_time] - entry_lon[flight]
		lat_dist = traj_lat[flight][flight_entry_time] - entry_lat[flight]
		
		# Initial 2D coordiante must be within some distance of the historical entry coordinate
		preTRA_model.addConstr(lon_dist <= entry_margin, name = f"lon_entry_pos[{flight}]")
		preTRA_model.addConstr(lon_dist >= -entry_margin, name = f"lon_entry_neg[{flight}]")
		preTRA_model.addConstr(lat_dist <= entry_margin, name = f"lat_entry_pos[{flight}]")
		preTRA_model.addConstr(lat_dist >= -entry_margin, name = f"lat_entry_neg[{flight}]")
	
def generate_reRouting_constraints(preTRA_model, arrived_by, traj_lon, traj_lat, traj_alt, flights, feas_times, feas_metering_fixes, fix_selected):
	# METERING FIX MUST BE REACHED AT EXIT TIME
	for flight in flights:
		# LEQ bigM: set to be larger than the largest coordinate
		max_fixLon_coord = max(feas_metering_fixes[flight], key = lambda x: x[0])[0]
		max_fixLat_coord = max(feas_metering_fixes[flight], key = lambda x: x[1])[1]
		max_fixAlt_coord = max(feas_metering_fixes[flight], key = lambda x: x[2])[2]
		bigM_fixLonLEQ = max_fixLon_coord - (dtw_lon - preTRACON_radius) + 10 # Setting big M as a safe upper bound for the metering fix coordinates
		bigM_fixLatLEQ = max_fixLat_coord - (dtw_lat - preTRACON_radius) + 10
		bigM_fixAltLEQ = max_fixAlt_coord + 10
		# GEQ bigM: set to be smaller than the smallest coordinate
		min_fixLon_coord = min(feas_metering_fixes[flight], key = lambda x: x[0])[0]
		min_fixLat_coord = min(feas_metering_fixes[flight], key = lambda x: x[1])[1]
		min_fixAlt_coord = min(feas_metering_fixes[flight], key = lambda x: x[2])[2]
		bigM_fixLonGEQ = (dtw_lon + preTRACON_radius) - min_fixLon_coord + 10 # Setting big M as a safe upper bound for the metering fix coordinates
		bigM_fixLatGEQ = (dtw_lat + preTRACON_radius) - min_fixLat_coord + 10
		bigM_fixAltGEQ = 50000 - min_fixAlt_coord # Very conservative upper bound on altitude
		for time in feas_times[flight]:
			chosen_fix_lon = gp.LinExpr()
			chosen_fix_lat = gp.LinExpr()
			chosen_fix_alt = gp.LinExpr()
			for metering_fix in feas_metering_fixes[flight]:
				chosen_fix_lon += fix_selected[flight][metering_fix] * metering_fix[0]
				chosen_fix_lat += fix_selected[flight][metering_fix] * metering_fix[1]
				chosen_fix_alt += fix_selected[flight][metering_fix] * metering_fix[2]
			
				# Enforce the longitude coordinate
			preTRA_model.addConstr(chosen_fix_lon <= traj_lon[flight][time] + bigM_fixLonLEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixLon_coordLEQ[{flight}][{time}]")
			preTRA_model.addConstr(chosen_fix_lon >= traj_lon[flight][time] - bigM_fixLonGEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixLon_coordGEQ[{flight}][{time}]")
			
				# Enforce the latitude coordinate
			preTRA_model.addConstr(chosen_fix_lat <= traj_lat[flight][time] + bigM_fixLatLEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixLat_coordLEQ[{flight}][{time}]")
			preTRA_model.addConstr(chosen_fix_lat >= traj_lat[flight][time] - bigM_fixLatGEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixLat_coordGEQ[{flight}][{time}]")
			
				# Enforce the altitude coordinate
			preTRA_model.addConstr(chosen_fix_alt <= traj_alt[flight][time] + bigM_fixAltLEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixAlt_coordLEQ[{flight}][{time}]")
			preTRA_model.addConstr(chosen_fix_alt >= traj_alt[flight][time] - bigM_fixAltGEQ*(1 - (arrived_by[flight][time] - arrived_by[flight][time - 1])), name = f"enf_fixAlt_coordGEQ[{flight}][{time}]")
			
	# FLIGHTS MUST CHOOSE ONE METERING FIX
	for flight in flights:
		sum_fix_sel = gp.LinExpr()
		for metering_fix in feas_metering_fixes[flight]:
			sum_fix_sel += fix_selected[flight][metering_fix]
			
		preTRA_model.addConstr(sum_fix_sel == 1, name = f"req_one_meterFix[{flight}]")
		
	# DEFINE "arrived_by"
	for flight in flights:
		for time in feas_times[flight]:
			preTRA_model.addConstr(arrived_by[flight][time] - arrived_by[flight][time - 1] >= 0, name = "def_arrived_by[{flight}][{time}]")
	
	# CREATE THE MODEL IN GUROBI
def create_preTRA_routing_model(flights, times, feas_times, feas_metering_fixes, max_velocity_lon, max_velocity_lat, max_velocity_alt, delay_weight, min_horiz_dist, min_vert_dist, entry_lon, entry_lat, entry_alt, entry_margin):
		# MODEL
	preTRA_model = gp.Model("preTRA_model")
	
		# DECISION VARIABLES
	traj_lon, traj_lat, traj_alt, separation_direc, fix_selected, arrived_by, lon_speed, lat_speed, alt_speed, lon_separ_sign, lat_separ_sign, alt_separ_sign = generate_decision_vars(preTRA_model, flights, times, feas_times, feas_metering_fixes, entry_alt)
	
		# OBJECTIVE FUNCTION
	objective = gp.LinExpr() # Empty experssion to fill
	for flight in flights:
		preTRA_exit_time = exit_time(arrived_by[flight], feas_times[flight])
		for time in feas_times[flight][:-1]:
			objective += lon_speed[flight][time] + lat_speed[flight][time] + alt_speed[flight][time]
			objective += delay_weight*(preTRA_exit_time - feas_times[flight][0])
	preTRA_model.setObjective(objective, GRB.MINIMIZE)
			
		# CONSTRAINTS
	generate_phys_collision_constrs(preTRA_model, arrived_by, traj_lon, traj_lat, traj_alt, separation_direc, max_velocity_lon, max_velocity_lat, max_velocity_alt, min_horiz_dist, min_vert_dist, entry_lon, entry_lat, entry_margin, lon_speed, lat_speed, alt_speed, lon_separ_sign, lat_separ_sign, alt_separ_sign, secs_per_interval)
	generate_reRouting_constraints(preTRA_model, arrived_by, traj_lon, traj_lat, traj_alt, flights, feas_times, feas_metering_fixes, fix_selected)
	
	return preTRA_model, traj_lon, traj_lat, traj_alt
	
	# MAIN METHOD
def reRoute_flights(flights, feas_times, preTRA_model, traj_lon, traj_lat, traj_alt):
	# Run the optimization
	preTRA_model.optimize()
	
	# Construct the trajectories
	if preTRA_model.status == GRB.OPTIMAL:
		traj_reRoutes = {}
		for flight in flights:
			traj_reRoutes[flight] = []
			for time in feas_times[flight]:
				if type(traj_alt[flight][time]) != gp.Var:
					current_coord = (traj_lon[flight][time].x, traj_lat[flight][time].x, traj_alt[flight][time])
				else:
					current_coord = (traj_lon[flight][time].x, traj_lat[flight][time].x, traj_alt[flight][time].x)
				traj_reRoutes[flight].append((time, current_coord))
				
		return traj_reRoutes
	else:
		print("NO SOLUTION FOUND")
		return None
	
def print_reRoutes(flights, traj_reRoutes):
	for flight in flights:
		print(flight, "Re-Routed Trajectory")
		for coord in traj_reRoutes[flight]:
			print("Timestep", coord[0], "Coordinate", coord[1]) # TODO print this as an actual datetime using conversion function

# RUN MODEL
delay_weight = 1 # TODO No idea how to set this
entry_margin = min_horiz_dist # TODO Set a reasonable entry margin
#TODO exit altitude margin
preTRA_model, traj_lon, traj_lat, traj_alt = create_preTRA_routing_model(flights, times, feas_times, feas_metering_fixes, max_velocity_lon, max_velocity_lat, max_velocity_alt, delay_weight, min_horiz_dist, min_vert_dist, entry_lon, entry_lat, entry_alt, entry_margin)
traj_reRoutes = reRoute_flights(flights, feas_times, preTRA_model, traj_lon, traj_lat, traj_alt)
print_reRoutes(flights, traj_reRoutes)

#TODO Better output (trajectory stops at exit time), visualization

#%% DEBUGGING
# Print constraints by name
# flight = 'SKW4010_KIADtoKDTW_0'
# time = 4
# intruder_flight = 'SKW4010_KIADtoKDTW_1'
# constraint_names = [f"def_lon_separ_pos[{flight}][{time}][{intruder_flight}]", f"def_lon_separ_neg[{flight}][{time}][{intruder_flight}]"]

# for constraint_name in constraint_names:
#     constraint = preTRA_model.getConstrByName(constraint_name)
#     constraint_LinExpr = preTRA_model.getRow(constraint)
    
#     # Print the constraint name and constraint formulation
#     print(constraint.ConstrName + ":\n", f"{constraint_LinExpr} {constraint.Sense} {constraint.RHS}")

#     # Print the constraint values at optimality
#     print("", end = " ") # Adds a single space indent
#     for varIndex in range(constraint_LinExpr.size() - 1):
#         print(constraint_LinExpr.getCoeff(varIndex) * constraint_LinExpr.getVar(varIndex).x, end = " + ")
#     print(constraint_LinExpr.getCoeff(constraint_LinExpr.size() - 1) * constraint_LinExpr.getVar(constraint_LinExpr.size() - 1).x, end = " ")
#     print(constraint.Sense, constraint.RHS)
#     print("")

# # 	# DEBUGGING
# preTRA_model.computeIIS()
# # Print out the IIS constraints and variables
# # From https://support.gurobi.com/hc/en-us/articles/15656630439441-How-do-I-use-compute-IIS-to-find-a-subset-of-constraints-that-are-causing-model-infeasibility
# print('\nThe following constraints and variables are in the IIS:')
# for c in preTRA_model.getConstrs():
#     if c.IISConstr: print(f'\n\t{c.constrname}: {preTRA_model.getRow(c)} {c.Sense} {c.RHS}')

# print("\n")
# for v in preTRA_model.getVars():
#     if v.IISLB: print(f'\t{v.varname} ≥ {v.LB}')
#     if v.IISUB: print(f'\t{v.varname} ≤ {v.UB}')