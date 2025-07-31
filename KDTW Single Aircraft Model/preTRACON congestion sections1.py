# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:15:10 2025

@author: Eric
"""
#%% IMPORT PACKAGES
import networkx as nx
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon, box, Point
import random
import pandas as pd
import ast

#%% INPUTS
# DTW 2D COORDINATES, pre-TRACON RADIUS
dtw_lat, dtw_lon = 42.2125, -83.3534

star_fixes = {
    "BONZZ": (-82.7972, 41.7483), "CRAKN": (-82.9405, 41.6730), "CUUGR": (-83.0975, 42.3643),
    "FERRL": (-82.6093, 42.4165), "GRAYT": (-83.6020, 42.9150), "HANBL": (-84.1773, 41.7375),
    "HAYLL": (-84.2975, 41.9662), "HTROD": (-83.3442, 42.0278), "KKISS": (-83.7620, 42.5443),
    "KLYNK": (-82.9888, 41.8793), "LAYKS": (-83.5498, 42.8532), "LECTR": (-84.0217, 41.9183),
    "RKCTY": (-83.9603, 42.6869), "VCTRZ": (-84.0670, 41.9878)
}

#%% SET UP TRACON AND PRE-TRACON, PRE-TRACON CONGESTION MAP
def get_fix_coords(tracon_polygon):
	return tracon_polygon.exterior.coords

def preTRACON_bounds(preTRACON_radius_deg):
	minx = dtw_lon - preTRACON_radius_deg
	miny = dtw_lat - preTRACON_radius_deg
	maxx = dtw_lon + preTRACON_radius_deg
	maxy = dtw_lat + preTRACON_radius_deg
	return (minx, miny, maxx, maxy)

def create_conges_map(box_unit_len, preTRACON_radius_deg, tracon_polygon):
	# Generate a circle "polygon" that encapsulates the pre-TRACON
	theta = np.linspace(0, 2 * np.pi, 400)
	circle_coords = [(dtw_lon + preTRACON_radius_deg * np.cos(t), dtw_lat + preTRACON_radius_deg * np.sin(t)) for t in theta]
	circle_polygon = Polygon(circle_coords)
	# Get the max and min bounds
	minx, miny, maxx, maxy = preTRACON_bounds(preTRACON_radius_deg)
	
	# Create the boxes in the map, ensuring that boundary-boxes only represent the intersection of the box and pre-TRACON
	grid_cells = []
	clipped_cells = []
	for lon in np.arange(minx, maxx, box_unit_len):
	    for lat in np.arange(miny, maxy, box_unit_len):
	        cell = box(lon, lat, lon + box_unit_len, lat + box_unit_len)
	        if circle_polygon.intersects(cell) and not tracon_polygon.contains(cell):
	            clipped = cell.intersection(circle_polygon)
	            if not clipped.is_empty:
	                grid_cells.append(cell)
	                clipped_cells.append(clipped)
					
	return clipped_cells
#%% # — PRE-TRACON GRID GENERATION —
def generate_preTRACON_grid(preTRACON_radius_deg, tracon_polygon):
	# — PRE-TRACON GRID GENERATION —
	radius = preTRACON_radius_deg * nodes_per_deg
	grid_density = radius*2 + 1 # This is the number of nodes in the horizontal diameter, based on the given radius
	center = (dtw_lon, dtw_lat)

	# NOTE: Each grid square in the traversal grid corresponds to a node that an aircraft can visit. We draw a network on top of the grid.
	#       Therefore, the terms "grid square" and "node" can be used interchangably in our vocabulary
	
	# Get values for the longitudinal and latitudinal boundaries of the preTRACON
	minx, miny, maxx, maxy = preTRACON_bounds(preTRACON_radius_deg)
	
	# Create node/grid-square identifiers: The names of the nodes will be their (lon, lat) coordinate within the grid
	grid_squares = []
		# Create grid longitudes and latitudes. Rounding is to prevent precision errors when adding/subtracting.
	step_size = (maxx - minx) / (grid_density - 1) # Subtract 1 to account for the leading 0 in the linspace
	step_size = np.trunc(step_size*10000)/10000
	
	longitudes = np.round(np.arange(minx, maxx, step_size), decimals = 4)
	latitudes = np.round(np.arange(miny, maxy, step_size), decimals = 4)
	for lon in longitudes: # These loops go through every row and column to enumerate all of the grid squares/nodes
		for lat in latitudes:
			dist = math.dist((lon, lat), (dtw_lon, dtw_lat)) # Distance in lateral degrees between this potential node and the center
			in_TRACON = tracon_polygon.contains(Point(lon, lat))
			if (dist <= preTRACON_radius_deg) and (not in_TRACON):
				grid_squares.append((lon, lat))

	# TRACON AND METERING FIXES
	# For each actual metering fix, find the closest node in the grid. This will be a "stand-in" for the actual metering fix.
	actual_fix_coords = get_fix_coords(tracon_polygon)[:-1] # Don't include the duplicate "closure point" at the end
	metering_fix_nodes = []
	all_nodes = grid_squares.copy()
	for actual_fix in actual_fix_coords:
		# Find the closest node
		closest_node = None
		closest_dist = float('inf')
		for current_node in all_nodes:
			current_dist = math.dist(current_node, actual_fix)
			if current_dist < closest_dist:
				closest_dist = current_dist
				closest_node = current_node
		# Add it to the list of metering fix nodes
		metering_fix_nodes.append(closest_node)
		# Remove it from the list of all nodes so we don't choose it twice
		all_nodes.remove(closest_node)
		
	# Now, add the center "airport" node
	grid_squares.append(center)

	# Now we need to make edges. We will be using a Moore neighborhood here, meaning that aircraft can travel to any of the 8 surrounding grid squares (octilinear motion).
	# We can do this by looping through every "neighborhood" of every grid square, creating an adjacency list (AKA list of neighbors of each node)
	moore_adjacency = {}
	neighbor_dist_adjacency = {}
	
	for current_lon, current_lat in grid_squares:
		# Create an entry in our adjacency list for this node. The dictionary key is the node's coordinate and the value is a list of neighbors
		moore_adjacency[(current_lon, current_lat)] = []
		neighbor_dist_adjacency[(current_lon, current_lat)] = []
		
		# Add neighbor nodes to the adjacency list
		for neighbor_lon in np.round([current_lon - step_size, current_lon, current_lon + step_size], decimals=4):
			for neighbor_lat in np.round([current_lat - step_size, current_lat, current_lat + step_size], decimals=4):
				# This conditional ensures that we don't create edges to rows and columns that are out of bounds (ex. no such thing as negative rows)
				if (maxx > neighbor_lon >= minx) and (maxy > neighbor_lat >= miny):
					# This conditional ensures that the neighbor is within the set radius
					dist = math.dist((neighbor_lon, neighbor_lat), center) # Distance, or num lateral edges between this potential node and the center
					in_TRACON = tracon_polygon.contains(Point(neighbor_lon, neighbor_lat))
					if (dist <= preTRACON_radius_deg) and (not in_TRACON):
						# This one ensures that nodes don't count themselves as neighbors
						if (current_lon, current_lat) != (neighbor_lon, neighbor_lat):
# 							if (neighbor_lon, neighbor_lat) not in grid_squares:
# 								print("FLOATING POINT ERROR")
							moore_adjacency[(current_lon, current_lat)].append((neighbor_lon, neighbor_lat))
							
							# Set the base edge distance proportions
							if (neighbor_lon != current_lon) and (neighbor_lat != current_lat): # This is a diagonal edge, base weight is sqrt(2)
								neighbor_dist_adjacency[(current_lon, current_lat)].append(math.sqrt(2))
							else:
								neighbor_dist_adjacency[(current_lon, current_lat)].append(1)
								
	# Generate Edges from the TRACON (center) to the metering fixes (no cost edges)
	for metering_fix in metering_fix_nodes:
		moore_adjacency[center].append(metering_fix)
		neighbor_dist_adjacency[center].append(0)
					
	# Create an edge list for networkx input
	edge_list = []
	for current_node in grid_squares:
		for index, neighbor_node in enumerate(moore_adjacency[current_node]):
			edge_list.append((current_node, neighbor_node, neighbor_dist_adjacency[current_node][index])) # weighted edge list, third coordiante is the distance proportion
			
	# Import data into networkx
	traversal_grid = nx.Graph()
	traversal_grid.add_weighted_edges_from(edge_list, weight = 'dist_propor')

	# Visualize
	traversal_grid.pos = {(x,y):(x,y) for x,y in traversal_grid.nodes()} # Sets node positions in a grid layout
	
	# List the outer ring of nodes
	possible_entry_nodes = find_outermost_nodes(grid_squares, center, step_size)
	
	return traversal_grid, possible_entry_nodes, metering_fix_nodes # Returns a networkx graph generated with the given parameters, plus start and end points

def find_outermost_nodes(grid_squares, center, step_size):
	# Get a list of the outermost ring of nodes
	possible_entry_nodes = []
	for current_lon, current_lat in grid_squares:
		dist = math.dist((current_lon, current_lat), center) # Distance, or num lateral edges between this potential node and the center
		if dist > preTRACON_radius_deg - step_size:
			possible_entry_nodes.append((current_lon, current_lat))
	
	return possible_entry_nodes

#%% — SET WEIGHTS BASED ON CONGESTION —
				
# READ IN AND CALCULATE THE CONGESTION COUNTS, BASED ON FLIGHT INFO
def get_conges_counts(df_path, entry_time, preTRACON_duration):
	# Read in the dataset and Convert the timestamps in the first column to pandas timestamp objects
	df = pd.read_csv(df_path)
	df['Time'] = pd.to_datetime(df['Time'])
	df.set_index('Time', inplace = True)
	    
	# Assume entry_time input is a pandas datetime and preTRACON_duration is a pandas timedelta
	exit_time = entry_time + preTRACON_duration
	    
	# Get rid of observations that occur before the entry_time or after the exit_time
	kept_rows = (df.index >= entry_time) & (df.index <= exit_time)
	df = df[kept_rows]
    # Test: df.to_csv('filtered_file.csv', index=False)
    
    # Average the population of each congestion zone over the given duration
	conges_avg = df.mean()
	conges_counts = conges_avg.to_list()
	
	return conges_counts

def get_conges_levels(conges_counts, lowest_level, num_levs, lev_interval):
	conges_levels = []
	for traf_count in conges_counts:
		for lev in range(1, num_levs + 1):
			if (lev - 1)*lev_interval <= traf_count < lev*lev_interval: # Figure out which traffic bin this zone is in, set level accordingly
				conges_levels.append(lowest_level + (lev - 1))
			elif lev == num_levs:
				if lev*lev_interval <= traf_count:
					conges_levels.append(lowest_level + (lev - 1))
	return conges_levels

# SET CONGESTION EDGE WEIGHTS
def set_conges_weights(traversal_grid, conges_counts, box_unit_len, preTRACON_radius_deg, tracon_polygon):
	conges_cells = create_conges_map(box_unit_len, preTRACON_radius_deg, tracon_polygon)
	conges_levels = get_conges_levels(conges_counts, lowest_conges_lev, num_conges_levs, conges_lev_interval)
	
	nx.set_edge_attributes(traversal_grid, 0, "conges_weight") # Create an edge attribute for the congestion weight
	for edge in traversal_grid.edges:
		for cell_ID, conges_cell in enumerate(conges_cells):
			if conges_cell.contains(Point(edge[0])): # The edge will be in the same congestion zone as its origin node
				traversal_grid.edges[edge]['conges_weight'] = traversal_grid.edges[edge]['dist_propor']*conges_levels[cell_ID]
		
#%% Read in flight information
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

#%% pre-TRACON Routing
def route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute):
	# For now, just use the built-in default function, and the distance proportion as weights
	flight_trajectory = nx.shortest_path(traversal_grid, source = entry_node, target = (dtw_lon, dtw_lat), weight = weight_attribute)
	#print([(str(lon), str(lat)) for lon, lat in flight_trajectory])
	assigned_metering_fix = flight_trajectory[len(flight_trajectory) - 2]
	trajectory_edges = [(flight_trajectory[i], flight_trajectory[i+1]) for i in range(len(flight_trajectory) - 2)] # We don't care about the last edge because it's inside of the TRACON
	
	return trajectory_edges, assigned_metering_fix

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
	edge_weights = [traversal_grid.edges[current_edge]['conges_weight'] for current_edge in traversal_grid.edges]
	weight_colorMap = plt.cm.Reds
	vmin = lowest_conges_lev
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

#%% PARAMETERS (USER INPUTS)
preTRACON_radius_deg = 3 # Radius of the pre-TRACON in latitudinal degrees

nodes_per_deg = 13 # Number of lateral nodes in one latitudinal degree (grid density)

box_unit_len = .7 # The size of the boxes in the congestion map

# Settings for binning the congestion levels
lowest_conges_lev = 1
num_conges_levs = 3
conges_lev_interval = 3

#%% INTERNAL VARIABLES

radius = preTRACON_radius_deg * nodes_per_deg

tracon_polygon = Polygon(star_fixes.values()).convex_hull

TRACON_node = (dtw_lon, dtw_lat)

#%% PROOF OF CONCEPT: CHOOSE AN AIRCRAFT ENTRY POINT AND ROUTE IT INTO THE TRACON

# Generate the grid and obtain the list of possible entry points
traversal_grid, possible_entry_nodes, metering_fix_nodes = generate_preTRACON_grid(preTRACON_radius_deg, tracon_polygon)

# Read in flight info
flight_df_path = 'Yuwei 5-12\\flight_entry_times_with_coords.csv'
acID = 'DAL1433_KMCOtoKDTW'
preTRA_entry_coord, preTRA_entry_time, preTRA_duration = read_in_flight_info(flight_df_path, acID)
# For now, just use a made up entry time and duration
# preTRA_entry_time = pd.Timestamp('2023-12-22 12:00:00')
# preTRA_duration = pd.Timedelta(seconds = 1000)

# Set the edge weights based on congestion
conges_df_path = 'Yuwei 5-12\\grid_counts.csv'
conges_counts = get_conges_counts(conges_df_path, preTRA_entry_time, preTRA_duration)
set_conges_weights(traversal_grid, conges_counts, box_unit_len, preTRACON_radius_deg, tracon_polygon)

# Function: find the closest "possible entry node" to the actual entry point of the input flight
entry_node = find_entry_node(preTRA_entry_coord, possible_entry_nodes)
# For this example, choose an entry node at random to be our start node
#entry_node = random.choice(possible_entry_nodes)

# Now, find the shortest path between our entry node and the TRACON node
trajectory_conges, assigned_conges_metering_fix = route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute = 'conges_weight')
trajectory_noConges, assigned_noConges_metering_fix = route_to_TRACON(traversal_grid, entry_node, TRACON_node, weight_attribute = 'dist_propor')

#%% Visualize
observe(traversal_grid, trajectory_conges, trajectory_noConges, TRACON_node, metering_fix_nodes, radius)

#%% EXPORT NODES AND CONGESTION CELLS
# pd.DataFrame(traversal_grid.nodes).to_csv("nodes.csv")

# mand_nodes = metering_fix_nodes.copy()
# mand_nodes.append(TRACON_node)
# pd.DataFrame(mand_nodes).to_csv("mand_nodes.csv")

# conges_cells = create_conges_map(box_unit_len, preTRACON_radius_deg, tracon_polygon)
# conges_cells_coords = [list(cell.exterior.coords) for cell in conges_cells]
# pd.DataFrame(conges_cells_coords).to_csv("conges_cells.csv")