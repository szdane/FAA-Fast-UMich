# -*- coding: utf-8 -*-
"""
Created on Fri May 16 09:46:44 2025

@author: Eric
"""

#%% IMPORT PACKAGES
from shapely.geometry import Polygon, Point
import pandas as pd
import ast
import math
import matplotlib.pyplot as plt

#%% PARAMETERS
minumum_seperation = .4


#%% READ-IN THE SET OF WAYPOINTS/NODES AND CONGESTION CELLS
def read_in_nodes(df_path):
	df = pd.read_csv(df_path)
	nodes = [(row["0"], row["1"]) for _, row in df.iterrows()]
	
	return nodes

def read_in_conges_cells(df_path):
	df = pd.read_csv(df_path)
	df.set_index(df.iloc[:,0], inplace = True)
	conges_cells = []
	for _, cell in df.iterrows():
		cell_coords = []
		for coord in cell.values[1:]:
			if pd.isna(coord):
				break
			cell_coords.append(ast.literal_eval(coord))
		cell_coords = tuple(cell_coords)
		conges_cells.append(cell_coords)
	
	return conges_cells

nodes = read_in_nodes("nodes.csv")
fix_nodes = 	read_in_nodes("mand_nodes.csv")
conges_cells = read_in_conges_cells("conges_cells.csv")

#%% CALCULATE THE SET OF NODES IN EACH CONGESTION CELL
def sort_conges_cell_nodes(nodes, conges_cells):
	nodes_in_cell = {}
	for cell_coords in conges_cells:
		nodes_in_cell[cell_coords] = []
		cell_polygon = Polygon(cell_coords)
		for node in nodes:
			node_coord = Point(node)
			if cell_polygon.contains(node_coord):
				nodes_in_cell[cell_coords].append(node)
				
	return nodes_in_cell

nodes_in_cell = sort_conges_cell_nodes(nodes, conges_cells)

#%% CALCULATE THE DISTANCE BETWEEN EACH PAIR OF NODES
# Only store the pairs for which distance is less than the minimum
def find_close_pairs(nodes, minumum_seperation):
	close_pairs = []
	num_nodes = len(nodes)
	for node1_ind in range(num_nodes):
		node1 = nodes[node1_ind]
		for node2_ind in range(node1_ind + 1, num_nodes):
			node2 = nodes[node2_ind]
			if not (node1 in fix_nodes and node2 in fix_nodes): # Don't count distance between fix nodes
				dist = math.dist(node1, node2)
				if dist < minumum_seperation:
					close_pairs.append((node1, node2))
				
	return close_pairs

close_pairs = find_close_pairs(nodes, minumum_seperation)

#%% READ IN THE MOST UTILIZED NODES FROM PREVIOUS EXPERIMENT
def read_in_node_popularity(df_path, nodes):
	df = pd.read_csv(df_path)
	df.set_index(df.iloc[:,0], inplace = True)
	df["0"] = df["0"].apply(eval)
	
	node_popularity = {}
	for _, row_series in df.iterrows():
		node = row_series["0"]
		popularity = row_series["1"]
		node_popularity[node] = popularity
		
	# If we don't have data on a node, the popularity is 0
	for node in nodes:
		if node not in node_popularity.keys():
			node_popularity[node] = 0
		
	return node_popularity
	
node_popularity = read_in_node_popularity("popular_nodes.csv", nodes)

#%% FIND THE OPTIMAL NODE LIST (RUN THE INTEGER PROGRAM)

import gurobipy as gp
from gurobipy import GRB

def create_wp_selector_model(nodes, fix_nodes, close_pairs, conges_cells, nodes_in_cell, node_popularity):
	# Create the model
	wp_selector = gp.Model("wp_selector")
	
	# Create the decision variables
	is_chosen = {}
	for node in nodes:
		is_chosen[node] = wp_selector.addVar(name = f"is_chosen[{node}]", vtype = GRB.BINARY)
		
	# Objective function
	wp_selector.setObjective(gp.quicksum(node_popularity[node]*is_chosen[node] for node in nodes))
	
	# Constraints
		# Enforce minumum distance
	for node1, node2 in close_pairs:
		wp_selector.addConstr(is_chosen[node1] + is_chosen[node2] <= 1, name = f"dist_enf[{node1, node2}]")
		
		# Choose at least one waypoint per congestion region
	for index, cell in enumerate(conges_cells):
		cell_nodes = nodes_in_cell[cell]
		if len(cell_nodes) != 0: # Account for tiny edge-case cells that contain 0 nodes
			sum_chosen = gp.LinExpr()
			for node in cell_nodes:
				sum_chosen += is_chosen[node]
			wp_selector.addConstr(sum_chosen >= 1, name = f"enf_cell_usage[{index}]")
		
		# We must choose the TRACON node and the metering fix nodes
	for node in fix_nodes:
		wp_selector.addConstr(is_chosen[node] == 1, name = f"mand_node[{node}]")
		
	return wp_selector, is_chosen

def get_optimal_nodes(wp_selector, is_chosen):
	wp_selector.optimize()
	if wp_selector.status == GRB.OPTIMAL:
		optimal_node_list = []
		for node in nodes:
			if is_chosen[node].x == 1:
				optimal_node_list.append(node)
				
		return optimal_node_list
	else:
		print("NO SOLUTION FOUND")
		return None

wp_selector, is_chosen = create_wp_selector_model(nodes, fix_nodes, close_pairs, conges_cells, nodes_in_cell, node_popularity)
optimal_node_list = get_optimal_nodes(wp_selector, is_chosen)

#%% EXPORT THE LIST OF OPTIMAL NODES
#pd.DataFrame(optimal_node_list).to_csv("optimal_sparse_nodes.csv")

#%% VISUALIZE THE OPTIMAL POINTS
x = [node[0] for node in optimal_node_list]
y = [node[1] for node in optimal_node_list]
plt.scatter(x,y)