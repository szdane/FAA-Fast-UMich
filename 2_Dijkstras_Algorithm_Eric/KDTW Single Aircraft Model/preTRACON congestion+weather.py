# -*- coding: utf-8 -*-
"""
Created on Sat May 17 12:08:53 2025

@author: Eric
"""

#%% IMPORT PACKAGES
import pandas as pd
import math
import networkx as nx
import ast
from shapely.geometry import Polygon, LineString, Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import openap
#import numpy as np

#%% READ IN OPTIMAL NODES, FIX NODES, AND CONGESTION REGIONS
def read_in_nodes(df_path):
	df = pd.read_csv(df_path)
	nodes = [(row["0"], row["1"]) for _, row in df.iterrows()]
	
	return nodes

#%% READ IN FLIGHT INFORMATION
def read_in_flight_info(df_path, acID):
	df = pd.read_csv(df_path) #'Yuwei 4-18\\flight_entry_times_with_coords.csv'
	df.set_index('acId', inplace = True)
	
	flight_info = df.loc[acID]
	entry_coord = ast.literal_eval(flight_info['enter_pre_tracon_coord']) # String to tuple
	entry_time = pd.to_datetime(flight_info['enter_pre_tracon'])
	duration_sec = pd.Timedelta(seconds = flight_info['duration_seconds'])
	
	return entry_coord, entry_time, duration_sec

# Find the pre-TRACON closest entry node to the entry coordinate of the flight we read in
def find_entry_node(entry_coord, possible_entry_nodes):
	closest_node = None
	closest_dist = float('inf')
	for current_node in possible_entry_nodes:
		current_dist = math.dist(current_node, entry_coord)
		if current_dist < closest_dist:
			closest_node = current_node
			closest_dist = current_dist
			
	return closest_node

#%% GENERATE THE PRE-TRACON NETWORK, CREATING AN EDGE BETWEEN THE "NUM_NEIGHBORS" CLOSEST NODES
def get_adj_list(nodes, fix_nodes, edge_gen_dist):
	adjacency_list = {}
	neighbor_dist_adjacency = {}
	
	num_nodes = len(nodes)
	for node1_ind in range(num_nodes):
		node1 = nodes[node1_ind]
		if node1 not in fix_nodes: # don't make edges to the TRACON yet
			node1_dist2TRACON = dist_to_nearest_fix(node1, fix_nodes)
			neighbors = []
			neighbor_distances = []
			for node2_ind in range(num_nodes):
				if node2_ind != node1_ind:
					node2 = nodes[node2_ind]
					if not edge_intersects_TRACON(node1, node2, fix_nodes):
						node2_dist2TRACON = dist_to_nearest_fix(node2, fix_nodes)
						if node2_dist2TRACON <= node1_dist2TRACON: # We only want neighbors that are closer to the TRACON than us
							dist = math.dist(node1, node2)
							if dist <= edge_gen_dist:
								neighbors.append(node2)
								neighbor_distances.append(dist)
			adjacency_list[node1] = neighbors.copy()
			neighbor_dist_adjacency[node1] = neighbor_distances.copy()
						
	# Generate Edges from the TRACON (center) to the metering fixes (no cost edges)
	for metering_fix in fix_nodes[:-1]:
		adjacency_list[metering_fix] = []
		neighbor_dist_adjacency[metering_fix] = []
		adjacency_list[metering_fix].append(TRACON_node)
		neighbor_dist_adjacency[metering_fix].append(0)
		
	return adjacency_list, neighbor_dist_adjacency

def dist_to_nearest_fix(node, fix_nodes):
	nearest_fix_dist = float('inf')
	for metering_fix in fix_nodes[:-1]: # not including the TRACON node
		dist_to_fix = math.dist(node, metering_fix)
		if dist_to_fix < nearest_fix_dist:
			nearest_fix_dist = dist_to_fix
			
	return nearest_fix_dist

def edge_intersects_TRACON(node1, node2, fix_nodes):
	tracon_polygon = Polygon(fix_nodes[:-1])
	edge_line = LineString([node1, node2])
	# Touching the boundary is okay, as long as the line doesn't pass through the interior
	touches_boundary = tracon_polygon.touches(edge_line)
	no_overlap = tracon_polygon.disjoint(edge_line)
	
	return not (touches_boundary or no_overlap)

def generate_preTRACON_grid(nodes, fix_nodes, minimum_seperation, edge_gen_dist, preTRACON_radius_deg):
	adjacency_list, neighbor_dist_adjacency = get_adj_list(nodes, fix_nodes, edge_gen_dist)
	
	# Create a list of edges
	edge_list = []
	for current_node in nodes[:-1]: # -1 to ensure we don't include the TRACON node
		for index, neighbor_node in enumerate(adjacency_list[current_node]):
			edge = (current_node, neighbor_node, neighbor_dist_adjacency[current_node][index])
			edge_list.append(edge)
			
	# Import data into networkx
	traversal_grid = nx.Graph()
	traversal_grid.add_weighted_edges_from(edge_list, weight = 'dist')

	# Visualize
	traversal_grid.pos = {(x,y):(x,y) for x,y in traversal_grid.nodes()} # Sets node positions in a grid layout
	
	# List the outer ring of nodes
	possible_entry_nodes = find_outermost_nodes(nodes, TRACON_node, minimum_seperation, preTRACON_radius_deg)
	
	return traversal_grid, possible_entry_nodes # Returns a networkx graph generated with the given parameters, plus start and end points

def find_outermost_nodes(grid_squares, center, step_size, preTRACON_radius_deg):
	# Get a list of the outermost ring of nodes
	possible_entry_nodes = []
	for current_lon, current_lat in grid_squares:
		dist = math.dist((current_lon, current_lat), center) # Distance, or num lateral edges between this potential node and the center
		if dist > preTRACON_radius_deg - step_size:
			possible_entry_nodes.append((current_lon, current_lat))
	
	return possible_entry_nodes

#%% GET CONGESTION COUNTS AND ASSIGN CONGESTION EDGE WEIGHTS
# READ IN AND CALCULATE THE CONGESTION COUNTS, BASED ON FLIGHT INFO
def get_conges_counts(df_path, entry_time, preTRACON_duration):
	# Read in the dataset and Convert the timestamps to pandas timestamp objects
	df = pd.read_csv(df_path)
	label_columns = list(df.columns[:5])
	df.columns = label_columns + pd.to_datetime(df.columns[5:]).tolist()
	
	# Convert the label node coordinates from string to tuple
	df['coord1'] = df['coord1'].apply(ast.literal_eval)
	df['coord2'] = df['coord2'].apply(ast.literal_eval)
	    
	# Assume entry_time input is a pandas datetime and preTRACON_duration is a pandas timedelta
	exit_time = entry_time + preTRACON_duration
	    
	# Get rid of observations that occur before the entry_time or after the exit_time
	timestamp_columns = df.columns[5:]
	kept_columns = [col for col in timestamp_columns if ((col >= entry_time) and (col <= exit_time))]
	kept_columns = label_columns + kept_columns
	df = df.loc[:, kept_columns]
    # Test: df.to_csv('filtered_file.csv', index=False)
    
    # Average the population of each congestion zone over the given duration
	conges_counts = pd.DataFrame()
	conges_avg = df.iloc[:,5:].mean(axis=1) # Outputs a 1D "series"
	conges_counts['edge'] = list(zip(df["coord1"], df["coord2"])) # new column with the edge label
	conges_counts['conges_avg'] = conges_avg # new column with the congestion average for that edge
	conges_counts.set_index('edge', inplace = True) # Use the edge label as the index
	
	return conges_counts.to_dict()['conges_avg'] # Key: edge label, Value: average congestion over duration

def get_conges_levels(conges_counts, lowest_level, num_levs, lev_interval):
	conges_levels = {}
	for edge, traf_count in conges_counts.items():
		for lev in range(1, num_levs + 1):
			if (lev - 1)*lev_interval <= traf_count < lev*lev_interval: # Figure out which traffic bin this zone is in, set level accordingly
				conges_levels[edge] = lowest_level + (lev - 1)
			elif lev == num_levs:
				if lev*lev_interval <= traf_count:
					conges_levels[edge] = lowest_level + (lev - 1)
	return conges_levels

# SET CONGESTION EDGE WEIGHTS
def set_conges_weights(traversal_grid, conges_counts):
	conges_levels = get_conges_levels(conges_counts, lowest_conges_lev, num_conges_levs, conges_lev_interval)
	
	nx.set_edge_attributes(traversal_grid, 0, "conges_weight") # Create an edge attribute for the congestion weight
	nx.set_edge_attributes(traversal_grid, 0, "conges_level")
	for edge in traversal_grid.edges:
		if TRACON_node not in edge:
			# Reconstruct edges as python floats to avoid np.float64 index error, and account for Yuwei's lat,lon instead of lon,lat
			f_edge = ((float(edge[0][1]), float(edge[0][0])), (float(edge[1][1]), float(edge[1][0])))
			
			traversal_grid.edges[edge]['conges_weight'] = traversal_grid.edges[edge]['dist']*conges_levels[f_edge]
			traversal_grid.edges[edge]['conges_level'] = conges_levels[f_edge]
		
#%% GET WEATHER-SEVERITY LEVEL AND ASSIGN WEATHER EDGE WEIGHTS
def get_weather_counts(df_path, entry_time, preTRACON_duration):
	# Read in the dataset and Convert the timestamps in the first column to pandas timestamp objects
	df = pd.read_csv(df_path)
	df['time'] = pd.to_datetime(df['time'])
	    
	# Assume entry_time input is a pandas datetime and preTRACON_duration is a pandas timedelta
	exit_time = entry_time + preTRACON_duration
	    
	# Get rid of observations that occur before the entry_time or after the exit_time
	kept_rows = (df['time'] >= entry_time) & (df['time'] <= exit_time)
	df = df[kept_rows]
    # Test: df.to_csv('filtered_file.csv', index=False)
    
	conges_avg = df.mean()
	# TODO: No peanalty for a level 1 or 2 Map: 1 or 2 as 0, 3 as 1, 4 as 2, 5 as 3
	# TODO: Create dictionary of edge -> average weather level
	weather_levels = conges_avg.to_list()
	
	return weather_levels

def set_weather_weights(traversal_grid, weather_levels):
	nx.set_edge_attributes(traversal_grid, 0, "weather_level")
	for edge in traversal_grid.edges:
		if TRACON_node not in edge:
			# Reconstruct edges as python floats to avoid np.float64 index error, and account for Yuwei's lat,lon instead of lon,lat
			f_edge = ((float(edge[0][1]), float(edge[0][0])), (float(edge[1][1]), float(edge[1][0])))
			
			traversal_grid.edges[edge]['weather_level'] = weather_levels[f_edge]
			
#%% COMBINE CONGESTION AND WEATHER INTO A SINGLE EDGE WEIGHT
# congestion + weather multiplied by distance
def set_overall_weights(traversal_grid, weather_proportion):
	for edge in traversal_grid.edges:
		traversal_grid.edges[edge]['overall_weight'] = (weather_proportion*traversal_grid.edges[edge]['weather_level'] + (1 - weather_proportion)*traversal_grid.edges[edge]['conges_level'])*traversal_grid.edges[edge]['dist']

#%% TRACON ROUTING
def route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute):
	# For now, just use the built-in default function, and the distance proportion as weights
	flight_trajectory = nx.shortest_path(traversal_grid, source = entry_node, target = TRACON_node, weight = weight_attribute)
	#print([(str(lon), str(lat)) for lon, lat in flight_trajectory])
	assigned_metering_fix = flight_trajectory[len(flight_trajectory) - 2]
	trajectory_edges = [(flight_trajectory[i], flight_trajectory[i+1]) for i in range(len(flight_trajectory) - 2)] # We don't care about the last edge because it's inside of the TRACON
	
	return flight_trajectory, trajectory_edges, assigned_metering_fix

#%% — GENERATE VISUALIZATION OF PRE-TRACON AND FLIGHT TRAJECTORY —
def observe(traversal_grid, trajectory_conges, trajectory_noConges, TRACON_node, metering_fix_nodes, radius):
	traversal_grid.pos = {(x,y):(x,y) for x,y in traversal_grid.nodes()} # Sets node positions in a grid layout
	
	# SET NODE COLOR AND SIZE
	# Set node colors for visualization
	nx.set_node_attributes(traversal_grid, "c", "color") # Most nodes have a color attribute of cyan for now
	traversal_grid.nodes[TRACON_node]['color'] = 'red' # TRACON is red
	for metering_fix in metering_fix_nodes:
	 	traversal_grid.nodes[metering_fix]['color'] = 'y' # Metering fixes are yellow
# 	for ent in possible_entry_nodes:
# 		 traversal_grid.nodes[ent]['color'] = 'g'
	node_colorList = [traversal_grid.nodes[current_node]['color'] for current_node in traversal_grid.nodes]
	
	# # Set node sizes for visualization
	base_node_size = 50000/radius**2 # Inverse proportional node size based on grid length
	nx.set_node_attributes(traversal_grid, 0, "size") #base_node_size, "size")
	traversal_grid.nodes[TRACON_node]['size'] = base_node_size*(radius**.9) # TRACON node relative size grows slowly based on grid size
	for metering_fix in metering_fix_nodes:
	 	traversal_grid.nodes[metering_fix]['size'] = base_node_size*(radius**.5)
# 	for ent in possible_entry_nodes:
# 		 traversal_grid.nodes[ent]['size'] = base_node_size
	node_sizeList = [traversal_grid.nodes[current_node]['size'] for current_node in traversal_grid.nodes]
	
	# DRAW EDGE COLOR BASED ON WEIGHT
	edge_weights = [traversal_grid.edges[current_edge]['conges_level'] for current_edge in traversal_grid.edges]
	weight_colorMap = plt.cm.Reds
	vmin = lowest_conges_lev - 1
	vmax = num_conges_levs
	
	# PLOT
	# Draw only the TRACON and metering fix nodes
	nx.draw_networkx_nodes(traversal_grid, pos = traversal_grid.pos, node_size = node_sizeList, node_color = node_colorList, edgecolors = "grey")
	# nx.draw_networkx_labels(traversal_grid, pos = traversal_grid.pos, font_size = 8, font_color = "black")
	# Draw the lattice grid semi-transparantly
	nx.draw_networkx_edges(traversal_grid, pos = traversal_grid.pos, edge_color = edge_weights, edge_cmap = weight_colorMap, edge_vmin=vmin, edge_vmax=vmax, width = .3, style = '-')#, alpha = .2)
	# Draw the congestion and no-congestion trajectories
	no_conges_color = 'c'
	conges_color = 'limegreen'
	nx.draw_networkx_edges(traversal_grid, pos = traversal_grid.pos, edgelist = trajectory_noConges, width = 1, edge_color = no_conges_color, style = '-', alpha = 1)
	nx.draw_networkx_edges(traversal_grid, pos = traversal_grid.pos, edgelist = trajectory_conges, width = 1, edge_color = conges_color, style = '-')
	plt.axis("off")
	fig = plt.gcf()
	# Plot the legends
	make_conges_legend(lowest_conges_lev, num_conges_levs, conges_lev_interval, fig, weight_colorMap)
	make_traj_legend(no_conges_color, conges_color, fig)
	# Figure width and height
	fig.set_figwidth(8)
	fig.set_figheight(8)
	# Show the plot
	plt.show()
	
def make_conges_legend(lowest_conges_lev, num_conges_levs, conges_lev_interval, fig, weight_colorMap):
	prox_artists = []
	for lev in range(1,num_conges_levs + 1):
		normalized_lev = (lev - lowest_conges_lev)/(num_conges_levs-1)
		lev_color = weight_colorMap(normalized_lev)
		
		lev_min = (lev - 1)*conges_lev_interval
		lev_max = lev*conges_lev_interval
		
		# Create an empty Line2D or mpatches patch "artist" object that stores this level and color
		if lev == num_conges_levs:
			#artist = plt.Line2D([], [], color = lev_color, label = f'Congestion Level {lev}: {lev_min} or more aircraft', marker="s", linewidth=0, markersize=10)
			artist = mpatches.Patch(color = lev_color, label = f'Congestion Level {lev}: {lev_min} or more aircraft')
		else:
			#artist = plt.Line2D([], [], color = lev_color, label = f'Congestion Level {lev}: {lev_min} to {lev_max} aircraft', marker="s", linewidth=0, markersize=10)
			artist = mpatches.Patch(color = lev_color, label = f'Congestion Level {lev}: {lev_min} to {lev_max} aircraft')
		prox_artists.append(artist)
		
	fig.legend(handles = prox_artists, title = 'Congestion Levels')
	
def make_traj_legend(no_conges_color, conges_color, fig):
	conges_artist = mpatches.Patch(color = conges_color, label = "Re-route Accounting for Congestion")
	noConges_artist = mpatches.Patch(color = no_conges_color, label = "Re-route NOT Accounting for Congestion")
	prox_artists = [conges_artist, noConges_artist]
		
	fig.legend(handles = prox_artists, title = 'Trajectory Labels', loc = 'upper left')

#%% PARAMETERS
minimum_seperation = .4 # The minimum seperation between the nodes we generated in the upstream optimization model
preTRACON_radius_deg = 3

# Settings for binning the congestion levels
lowest_conges_lev = 1
num_conges_levs = 3
conges_lev_interval = 3

# Internal parameters
edge_gen_dist = 1.4

#%% READ IN INPUTS
nodes = read_in_nodes("optimal_sparse_nodes.csv")
fix_nodes = read_in_nodes("mand_nodes.csv")
TRACON_node = fix_nodes[-1]

#%% PROOF OF CONCEPT: CHOOSE AN AIRCRAFT ENTRY POINT AND ROUTE IT INTO THE TRACON

# Generate the grid and obtain the list of possible entry points
traversal_grid, possible_entry_nodes = generate_preTRACON_grid(nodes, fix_nodes, minimum_seperation, edge_gen_dist, preTRACON_radius_deg)

# Read in flight info
flight_df_path = 'Yuwei 5-12\\flight_entry_times_with_coords.csv'
acID = 'SKW4010_KIADtoKDTW'
preTRA_entry_coord, preTRA_entry_time, preTRA_duration = read_in_flight_info(flight_df_path, acID)
# For now, just use a made up entry time and duration
# preTRA_entry_time = pd.Timestamp('2023-12-22 12:00:00')
# preTRA_duration = pd.Timedelta(seconds = 1000)

# Set the edge weights based on congestion
conges_df_path = 'Yuwei 6-22\\weighted_edges.csv'
conges_counts = get_conges_counts(conges_df_path, preTRA_entry_time, preTRA_duration)
set_conges_weights(traversal_grid, conges_counts)

# Function: find the closest "possible entry node" to the actual entry point of the input flight
entry_node = find_entry_node(preTRA_entry_coord, possible_entry_nodes)
# For this example, choose an entry node at random to be our start node
#entry_node = random.choice(possible_entry_nodes)

# Now, find the shortest path between our entry node and the TRACON node
trajectory_conges, trajectory_conges_edges, assigned_conges_metering_fix = route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute = 'conges_weight')
trajectory_noConges, trajectory_noConges_edges, assigned_noConges_metering_fix = route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute = 'dist')


#%% VISUALIZE
observe(traversal_grid, trajectory_conges_edges, trajectory_noConges_edges, TRACON_node, fix_nodes[:-1], preTRACON_radius_deg*13)

#%% OUTPUT THE GENERATED EDGE PAIRS
def get_edge_pairs(traversal_grid, TRACON_node):
	edge_list = []
	for edge in traversal_grid.edges:
		if TRACON_node not in edge: # Don't export the TRACON edges
			edge_list.append((tuple(map(float, edge[0])), tuple(map(float, edge[1])))) # Convert to floats to avoid the output containing "np.float64()"
			
	pd.DataFrame(edge_list).to_csv("sparse_edges.csv")
	
#get_edge_pairs(traversal_grid, TRACON_node)

#%% OUTPUT TRAJECTORY

# Convert from degrees to meters (from Kuang)
def proj_with_defined_origin(node, TRACON_node, inverse=False): #lon0, lat0 are the coordinates of the selected origin
	if not inverse: # from lat, lon to x, y
		lon = node[0]
		lat = node[1]
		lon0 = TRACON_node[0]
		lat0 = TRACON_node[1]
	
		bearings = openap.aero.bearing(lat0, lon0, lat, lon) / 180 * 3.14159
		distances = openap.aero.distance(lat0, lon0, lat, lon)
		x = distances * math.sin(bearings)
		y = distances * math.cos(bearings)

		return x, y

def output_reRoute_trajec(trajectory, TRACON_node, trajec_name):
	traj_list = []
	for node in trajectory:
		x, y = proj_with_defined_origin(node, TRACON_node)
		traj_list.append((x, y))
		
	pd.DataFrame(traj_list).to_csv(f"{trajec_name}.csv")
	
#output_reRoute_trajec(trajectory_conges, TRACON_node, "trajectory_w_congestion")